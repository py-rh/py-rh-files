def train_fashion_mnist():
    from trainer import FashionMNISTTrainer

    trainer = FashionMNISTTrainer(epochs=3, batch_size=100, lr=0.1, momentum=0.9)
    trainer.setup_distributed()
    trainer.setup_model()
    trainer.setup_data()
    trainer.train()
    trainer.cleanup()


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
            packages_to_install=[
                "https://github.com/py-rh/py-rh-files/archive/py/kubeflow-testing.tar.gz#subdirectory=simple_model_pkg"
            ],
        ),
        runtime=torch_runtime,
    )
    client.wait_for_job_status(name=job_name, status={"Running"})

    for logline in client.get_job_logs(job_name, follow=True):
        print(logline)
