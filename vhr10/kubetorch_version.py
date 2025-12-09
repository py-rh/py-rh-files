import argparse
import time

import kubetorch as kt

from trainer import VHR10Trainer


def main():
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

    # Define compute configuration
    img = kt.Image(image_id="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime").pip_install(
        [
            "torchgeo[datasets,models]",
            "transformers",
            "torchvision",
            "pillow",
            "soxr",  # Required by transformers for audio_utils
        ]
    )

    gpu_compute = kt.Compute(
        gpus=1,
        image=img,
        launch_timeout=600,
        inactivity_ttl="2h",
        secrets=["huggingface"],
    ).distribute("pytorch", workers=args.workers)

    # Initialize trainer arguments
    init_args = dict(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Dispatch trainer class to remote GPUs
    remote_trainer = kt.cls(VHR10Trainer).to(gpu_compute, init_args=init_args)
    print("Time to first activity:", time.time() - start_time)

    # Run distributed training
    remote_trainer.setup(
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )
    print("Time to setup:", time.time() - start_time) # 16.46 on 2nd run, 274 seconds on first run, cold start on image too

    data_start = time.time()
    remote_trainer.load_data(args.batch_size)
    print("Time to load data:", time.time() - data_start) # 1.6794 seconds on 2nd run, 5.88 seconds on 1st run
    print("Time to start training:", time.time() - start_time) # 18.14 seconds on 2nd+ run, 280 seconds on 1st run

    remote_trainer.train(num_epochs=args.epochs, threshold=args.threshold)
    print("Training complete, total time:", time.time() - start_time) # 161 seconds on 2nd+ run, 429s on first run

if __name__ == "__main__":
    main()
