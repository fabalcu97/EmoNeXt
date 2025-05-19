import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train EmoNeXt model")

    parser.add_argument(
        "--dataset-path", type=str, help="Path to the dataset", default="FER2013"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Path where the best model will be saved",
    )
    parser.add_argument(
        "--patience", type=int, default=12, help="Number of patience before giving up"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--scheduler-max-lr", type=float, default=1e-4, help="Scheduler max learning rate")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--in_22k", action="store_true", default=False)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating the model weights",
    )
    parser.add_argument(
        '--ema-decay',
        type=float,
        default=0.9999,
        help='Exponential moving average decay rate for the model weights',
    )
    parser.add_argument(
        '--ema-update-every',
        type=int,
        default=1,
        help='Number of steps to wait before updating the EMA model weights',
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="The number of subprocesses to use for data loading."
        "0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file for resuming training or performing inference",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "base", "large", "xlarge"],
        default="tiny",
        help="Choose the size of the model: tiny, small, base, large, or xlarge",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--use-cbam",
        type=bool,
        required=False,
        default=False,
        help="Use CBAM or SE. True for CBAM, False for SE",
    )

    parser.add_argument(
        "--show-summary",
        type=bool,
        required=False,
        default=False,
        help="Show the model summary and exit",
    )

    return parser.parse_args()
