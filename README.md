# Image Annotations: Training On Custom Datasets

This package enables you to train deep learning models on custom set of videos that you can annotate efficiently in real time.

## Installation

You may install this repository for use by running the following commans in your terminal.

```
git clone http://github.com/brandontrabucco/image_annotations'
cd ./image_annotations
pip install -r requirements.txt
```

## Usage

You may copy your videos into the data folder and begin the dataset generation process by the following.

```
python build.py --video_dir="./data/" --output_dir="./train/"
```
