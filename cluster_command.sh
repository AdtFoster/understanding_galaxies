#!/bin/bash
#SBATCH --job-name=understand                       # Job name
#SBATCH --output=understand_%A.log 
#SBATCH --mem=32gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node

pwd; hostname; date

# nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

ZOOBOT_DIR=/share/nas/walml/repos/understanding_galaxies
PYTHON=/share/nas/walml/miniconda3/envs/zoobot/bin/python

FITS_DIR=/share/nas/walml/galaxy_zoo/decals/dr5/fits_native/J000

SCALE_FACTOR=1.2
SCALED_IMG_DIR=/share/nas/walml/repos/understanding_galaxies/scaled_$SCALE_FACTOR

$PYTHON $ZOOBOT_DIR/creating_image_main.py \
    --fits-dir $FITS_DIR \
    --scale-factor $SCALE_FACTOR \
    --save-dir $SCALED_IMG_DIR
    
$PYTHON $ZOOBOT_DIR/make_predictions.py \
    --batch-size 128 \
    --input-dir $SCALED_IMG_DIR \
    --checkpoint-loc /share/nas/walml/repos/zoobot_test/data/pretrained_models/decals_dr_train_set_only_replicated/checkpoint \
    --save-loc /share/nas/walml/repos/understanding_galaxies/results/scaled_image_predictions_$SCALE_FACTOR.csv
    
  #Split here
  
  #SBATCH --job-name=understand                       # Job name
#SBATCH --output=understand_%A.log 
#SBATCH --mem=32gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node

pwd; hostname; date

# nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

ZOOBOT_DIR=/share/nas/walml/repos/understanding_galaxies
PYTHON=/share/nas/walml/miniconda3/envs/zoobot/bin/python

FITS_DIR=/share/nas/walml/galaxy_zoo/decals/dr5/fits_native/J000

SCALED_IMG_DIR=/share/nas/walml/repos/understanding_galaxies/scaled

MIN_GAL = 100
MAX_GAL = 105

$PYTHON $ZOOBOT_DIR/creating_images_semester_two.py \
    --fits-dir $FITS_DIR \
    --save-dir $SCALED_IMG_DIR \
    --max-redshift 0.15 \
    --step-szie 0.002
    
$PYTHON $ZOOBOT_DIR/make_predictions.py \
    --batch-size 128 \
    --input-dir $SCALED_IMG_DIR \
    --checkpoint-loc /share/nas/walml/repos/zoobot_test/data/pretrained_models/decals_dr_train_set_only_replicated/checkpoint \
    --save-loc /share/nas/walml/repos/understanding_galaxies/results/scaled_image_predictions.csv
    
$PYTHON $ZOOBOT_DIR/create_dataframe.py \
    --file-name /share/nas/walml/repos/understanding_galaxies/results/scaled_image_predictions.csv

$PYTHON $ZOOBOT_DIR/sampling_galaxies.py \
    --min-gal $MIN_GAL \
    --max-gal $MAX_GAL \
    --min-delta-z 0.005 \
    --max-delta-z 0.007 \
    --step-delta-z 0.001 \
    --min-delta-p 0.015 \
    --max-delta-p 0.017 \
    --step-delta-p 0.001 \
    --min-delta-mag 0.4 \
    --max-delta-mag 0.6 \
    --step-delta-mag 0.1
    
$PYTHON $ZOOBOT_DIR/plotting.py \

$PYTHON $ZOOBOT_DIR/cluster_conf_matrix_code.py \
    --min-gal $MIN_GAL \
    --max-gal $MAX_GAL \
    --update-interval 50 \
    --pred-z 0.1 \
    --threshold-val 0.8 \
    --delta-z 0.006 \
    --delta-p 0.016 \
    --delta-mag 0.5
