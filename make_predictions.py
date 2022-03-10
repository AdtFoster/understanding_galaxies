import os
import logging
import glob
import pandas as pd
from pathlib import Path
import argparse

import tensorflow as tf

from zoobot import label_metadata, schemas
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess
from zoobot.tensorflow.predictions import predict_on_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', dest='input_dir', type=str)
    parser.add_argument('--checkpoint-loc', dest='checkpoint_loc', type=str)
    parser.add_argument('--save-loc', dest='save_loc', type=str)
    parser.add_argument('--batch-size', dest='batch_size', default=128, type=int)

    args = parser.parse_args()

    tf.get_logger().setLevel('ERROR')
    logging.basicConfig(level=logging.INFO)

    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    """
    List the images to make predictions on.
    """
    file_format = 'png'

    # utility function to easily list the images in a folder.
    unordered_image_paths = predict_on_dataset.paths_in_folder(Path(args.input_dir), file_format=file_format, recursive=False)

    ## or maybe you already have a list from a catalog?
    # unordered_image_paths = df['paths']

    assert len(unordered_image_paths) > 0
    assert os.path.isfile(unordered_image_paths[0])

    """
    Load the images as a tf.dataset, just as for training
    """
    initial_size = 424  # 300 for paper, from tfrecord or from png (png will be resized when loaded, before preprocessing)
    batch_size = args.batch_size  # 128 for paper, you'll need a very good GPU. 8 for debugging, 64 for RTX 2070, 256 for A100
    raw_image_ds = image_datasets.get_image_dataset([str(x) for x in unordered_image_paths], file_format, initial_size, batch_size)

    preprocessing_config = preprocess.PreprocessingConfig(
        label_cols=[],  # no labels are needed, we're only doing predictions
        input_size=initial_size,
        make_greyscale=True,
        normalise_from_uint8=True  # False for tfrecords with 0-1 floats, True for png/jpg with 0-255 uints
    )
    image_ds = preprocess.preprocess_dataset(raw_image_ds, preprocessing_config)
    # image_ds will give batches of (images, paths) when label_cols=[]

    
    """
    Define the model and load the weights.
    You must define the model exactly the same way as when you trained it.
    """
    crop_size = int(initial_size * 0.75)
    resize_size = 224  # 224 for paper
    channels = 3

    """
    If you're just using the full pretrained Galaxy Zoo model, without finetuning, you can just use include_top=True.
    """

    model = define_model.load_model(
        checkpoint_loc=args.checkpoint_loc,
        include_top=True,
        input_size=initial_size,
        crop_size=crop_size,
        resize_size=resize_size,
        expect_partial=True  # optimiser state will not load as we're not using it for predictions
    )

    label_cols = label_metadata.decals_label_cols  

    """
    If you have done finetuning, use include_top=False and replace the output layers exactly as you did when training.
    For example, below is how to load the model in finetune_minimal.py.
    """

    n_samples = 1
    predict_on_dataset.predict(image_ds, model, n_samples, label_cols, args.save_loc)
