import argparse
from ckPlus import CkPlus
from fer2013 import Fer2013
from rafDB import RafDB


datasets = {
    "fer2013": Fer2013,
    "ck+": CkPlus,
    "rafdb": RafDB,
}

parser = argparse.ArgumentParser(description="Download and prepare emotion datasets.")
parser.add_argument(
    "--dataset",
    "-d",
    choices=list(datasets.keys()),
    required=True,
    help="Dataset to download and prepare (fer2013, ck+, rafdb)",
)
args = parser.parse_args()


Dataset = datasets.get(args.dataset, None)
if Dataset is None:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

dataset = Dataset()
dataset.download_and_move()
dataset.post_processing()
dataset.delete_downloaded_files()
