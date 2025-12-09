import argparse
import os 

def train_vhr10(
    start_time,
    epochs,
    batch_size,
    lr,
    data_root,
    checkpoint_dir,
    model_name,
    freeze_backbone,
    num_classes,
    threshold, 
):
    import time

    from trainer import VHR10Trainer

    trainer = VHR10Trainer(data_root=data_root, checkpoint_dir=checkpoint_dir)
    print("Time to first activity:", time.time() - start_time) # 287.35 from cold, 55.33 from warm

    trainer.setup(
        lr=lr,
        freeze_backbone=freeze_backbone,
        model_name=model_name,
        num_classes=num_classes,
    )
    print("Time to setup:", time.time() - start_time)

    data_start = time.time()
    trainer.load_data(batch_size, num_workers=0)
    print("Time to load data:", time.time() - data_start)
    print("Time to start training:", time.time() - start_time) # 297.81, 66 from warm cached image

    trainer.train(num_epochs=epochs, threshold=threshold)
    print("Training complete, total time:", time.time() - start_time)


if __name__ == "__main__":
    import time

    start_time = time.time()

    parser = argparse.ArgumentParser(description="VHR10 Classification with DINOv3")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--data-root", type=str, default="./data", help="data root directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vitl16",
        choices=["vitl16", "vit7b16"],
        help="DINOv3 model variant (vitl16: distilled, vit7b16: original)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="freeze backbone weights",
    )
    parser.add_argument(
        "--workers", type=int, default=3, help="number of distributed workers"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="number of classes (VHR10 uses labels 1-10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="probability to use for binary classification",
    )

    args = parser.parse_args()

    from kubeflow.trainer import CustomTrainer, TrainerClient

    client = TrainerClient()
    for runtime in client.list_runtimes():
        print(runtime)
        if runtime.name == "torch-distributed":
            torch_runtime = runtime

    job_name = client.train(
        trainer=CustomTrainer(
            func=train_vhr10,
            func_args={
                "start_time": start_time,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "data_root": args.data_root,
                "checkpoint_dir": args.checkpoint_dir,
                "model_name": args.model_name,
                "freeze_backbone": args.freeze_backbone,
                "num_classes": args.num_classes,
                "threshold": args.threshold, 
            },
            num_nodes=args.workers,
            resources_per_node={
                # "cpu": 2,
                # "memory": "7Gi",
                "nvidia.com/gpu": 1,
            },
            packages_to_install=[
                "https://github.com/py-rh/py-rh-files/archive/py/kubeflow-testing.tar.gz#subdirectory=vhr10",
                "torchgeo[datasets,models]",
                "transformers",
                "torchvision",
                "pillow",
                "soxr",
            ],
            env={"HF_TOKEN": os.environ["HF_TOKEN"]}
        ),
        runtime=torch_runtime,
    )
    client.wait_for_job_status(name=job_name, status={"Running"})

    for logline in client.get_job_logs(job_name, follow=True):
        print(logline)

    print("Training took:", time.time() - start_time) # 342.5 for cold start, 112s for warm start
