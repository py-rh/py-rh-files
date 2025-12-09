def train_fashion_mnist():
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, DistributedSampler
    from torchvision import datasets, transforms

    # Define the PyTorch CNN model to be trained
    from model import SimpleNN 

    # Use NCCL if a GPU is available, otherwise use Gloo as communication backend.
    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    print(f"Using Device: {device}, Backend: {backend}")

    # Setup PyTorch distributed.
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    dist.init_process_group(backend=backend)
    print(
        "Distributed Training for WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}".format(
            dist.get_world_size(),
            dist.get_rank(),
            local_rank,
        )
    )

    # Create the model and load it into the device.
    device = torch.device(f"{device}:{local_rank}")
    model = nn.parallel.DistributedDataParallel(SimpleNN().to(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    
    # Download FashionMNIST dataset only on local_rank=0 process.
    if local_rank == 0:
        dataset = datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    dist.barrier()
    dataset = datasets.FashionMNIST(
        "./data",
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )


    # Shard the dataset accross workers.
    train_loader = DataLoader(
        dataset,
        batch_size=100,
        sampler=DistributedSampler(dataset)
    )

    # TODO(astefanutti): add parameters to the training function
    dist.barrier()
    for epoch in range(1, 3):
        model.train()

        # Iterate over mini-batches from the training set
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Copy the data to the GPU device if available
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and dist.get_rank() == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    # Wait for the distributed training to complete
    dist.barrier()
    if dist.get_rank() == 0:
        print("Training is finished")

    # Finally clean up PyTorch distributed
    dist.destroy_process_group()


if __name__ == "__main__":
    from kubeflow.trainer import CustomTrainer, TrainerClient
    
    client = TrainerClient()
    for runtime in client.list_runtimes():
        print(runtime)
        if runtime.name == "torch-distributed":
            torch_runtime = runtime

    job_name = client.train(
        trainer=CustomTrainer(
            func=train_fashion_mnist,
            # Set how many PyTorch nodes you want to use for distributed training.
            num_nodes=3,
            # Set the resources for each PyTorch node.
            resources_per_node={
                "cpu": 2,
                "memory": "7Gi",
                "nvidia.com/gpu": 1,
            },
        ),
        runtime=torch_runtime,
    )
    client.wait_for_job_status(name=job_name, status={"Running"})

    for logline in client.get_job_logs(job_name, follow=True):
        print(logline)
