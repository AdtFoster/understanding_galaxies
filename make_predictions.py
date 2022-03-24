import os
import logging
from pathlib import Path
import argparse

import tensorflow as tf

from zoobot.shared import label_metadata
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess
from zoobot.tensorflow.predictions import predict_on_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', dest='image_dir', type=str)
    parser.add_argument('--checkpoint-loc', dest='checkpoint_loc', type=str)
    parser.add_argument('--save-dir', dest='save_dir', default='results/latest_scaled_predictions', type=str)
    parser.add_argument('--batch-size', dest='batch_size', default=128, type=int)
    parser.add_argument('--overwrite', dest='overwrite', default=False, action='store_true')

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
    file_format = 'jpeg'

    # utility function to easily list the images in a folder.
    all_image_paths = predict_on_dataset.paths_in_folder(Path(args.image_dir), file_format=file_format, recursive=False)

    requested_img_size_after_loading = 300  # 300 for paper, from tfrecord or from png (png will be resized when loaded, before preprocessing)
    batch_size = args.batch_size  # 128 for paper, you'll need a very good GPU. 8 for debugging, 64 for RTX 2070, 256 for A100
    crop_size = int(requested_img_size_after_loading * 0.75)
    resize_size = 224  # 224 for paper
    channels = 3
    label_cols = label_metadata.decals_all_campaigns_ortho_label_cols  

    assert len(all_image_paths) > 0
    assert os.path.isfile(all_image_paths[0])
    logging.info('Total images to predict on: {}'.format(len(all_image_paths)))

 
    """
    Define the model and load the weights.
    You must define the model exactly the same way as when you trained it.
    """
    """
    If you're just using the full pretrained Galaxy Zoo model, without finetuning, you can just use include_top=True.
    """

    model = define_model.load_model(
        checkpoint_loc=args.checkpoint_loc,
        output_dim=len(label_cols),
        include_top=True,
        input_size=requested_img_size_after_loading,
        crop_size=crop_size,
        resize_size=resize_size,
        expect_partial=True  # optimiser state will not load as we're not using it for predictions
    )


    preprocessing_config = preprocess.PreprocessingConfig(
        label_cols=[],  # no labels are needed, we're only doing predictions
        input_size=requested_img_size_after_loading,
        make_greyscale=True,
        normalise_from_uint8=True  # False for tfrecords with 0-1 floats, True for png/jpg with 0-255 uints
    )

    png_batch_size = 10000
    png_start_index = 0
    n_samples = 1
    while png_start_index < len(all_image_paths):
        
        save_loc = os.path.join(args.save_dir, '{}.hdf5'.format(png_start_index))
        if not os.path.isfile(save_loc) or args.overwrite:

            this_loop_image_paths = all_image_paths[png_start_index:png_start_index+png_batch_size]

            raw_image_ds = image_datasets.get_image_dataset(
                image_paths=[str(x) for x in this_loop_image_paths],
                file_format=file_format,
                requested_img_size=requested_img_size_after_loading,
                batch_size=batch_size
            )

            image_ds = preprocess.preprocess_dataset(raw_image_ds, preprocessing_config)
            # image_ds will give batches of (images, paths) when label_cols=[]

            predict_on_dataset.predict(image_ds, model, n_samples, label_cols, save_loc)

        png_start_index += png_batch_size
