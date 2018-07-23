"""Author: Brandon Trabucco.
Load the image files and build the dataset.
"""


import argparse


from image_annotations.dataset.dataset_annotator import DatasetAnnotator
from image_annotations.dataset.dataset_adapter import DatasetAdaptor


if __name__ == "__main__":
    """Program entry point.
    """

    parser = argparse.ArgumentParser("Video annotator.")
    parser.add_argument("--video_dir", type=str, default="./data/")
    parser.add_argument("--train_dir", type=str, default="./train/")
    args = parser.parse_args()

    # Serialize the dataset to shard files.
    annotations = DatasetAnnotator(args.video_dir, 1).start()
    DatasetAdaptor(annotations, args.train_dir).start()