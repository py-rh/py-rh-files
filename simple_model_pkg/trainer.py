import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from model import SimpleNN


class FashionMNISTTrainer:
    def __init__(self, epochs=3, batch_size=100, lr=0.1, momentum=0.9):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.device = None
        self.local_rank = None
        self.model = None
        self.optimizer = None
        self.train_loader = None

    def setup_distributed(self):
        # Use NCCL if a GPU is available, otherwise use Gloo as communication backend.
        device_type, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
        print(f"Using Device: {device_type}, Backend: {backend}")

        # Setup PyTorch distributed.
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        dist.init_process_group(backend=backend)
        print(
            "Distributed Training for WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}".format(
                dist.get_world_size(),
                dist.get_rank(),
                self.local_rank,
            )
        )

        # Create the device.
        self.device = torch.device(f"{device_type}:{self.local_rank}")

    def setup_model(self):
        # Create the model and load it into the device.
        self.model = nn.parallel.DistributedDataParallel(SimpleNN().to(self.device))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    def setup_data(self):
        # Download FashionMNIST dataset only on local_rank=0 process.
        if self.local_rank == 0:
            datasets.FashionMNIST(
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

        # Shard the dataset across workers.
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(dataset)
        )

    def train(self):
        dist.barrier()
        for epoch in range(1, self.epochs + 1):
            self.model.train()

            # Iterate over mini-batches from the training set
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                # Copy the data to the GPU device if available
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Forward pass
                outputs = self.model(inputs)
                loss = F.nll_loss(outputs, labels)
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 10 == 0 and dist.get_rank() == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(inputs),
                            len(self.train_loader.dataset),
                            100.0 * batch_idx / len(self.train_loader),
                            loss.item(),
                        )
                    )

        # Wait for the distributed training to complete
        dist.barrier()
        if dist.get_rank() == 0:
            print("Training is finished")

    def cleanup(self):
        dist.destroy_process_group()
