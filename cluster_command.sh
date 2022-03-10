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

MIN_GAL_SQUID = 100
MAX_GAL_SQUID = 105

PERCENT = 66

PRED_Z = 0.03
MAX_Z = 0.12
STEP_SIZE = 0.002

MIN_DELTA_Z = 0.005
MAX_DELTA_Z = 0.007
STEP_DELTA_Z = 0.001
MIN_DELTA_P = 0.015
MAX_DELTA_P = 0.017
STEP_DELTA_P = 0.001
MIN_DELTA_MAG = 0.4
MAX_DELTA_MAG = 0.6
STEP_DELTA_MAG = 0.1
MIN_DELTA_MASS = 0.9
MAX_DELTA_MASS = 1.1
STEP_DELTA_MASS = 0.1

UPDATE_INTERVAL = 50
THRESHOLD_VAL = 0.8

DELTA_Z = 0.006
DELTA_P = 0.016
DELTA_MAG = 0.5

$PYTHON $ZOOBOT_DIR/creating_images_semester_two.py \
    --fits-dir $FITS_DIR \
    --save-dir $SCALED_IMG_DIR \
    --max-redshift $MAX_Z \
    --step-size $STEP_SIZE
    
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
    --min-delta-z $MIN_DELTA_Z \
    --max-delta-z $MAX_DELTA_Z \
    --step-delta-z $STEP_DELTA_Z \
    --min-delta-p $MIN_DELTA_P \
    --max-delta-p $MAX_DELTA_P \
    --step-delta-p $STEP_DELTA_P \
    --min-delta-mag $MIN_DELTA_MAG \
    --max-delta-mag $MAX_DELTA_MAG \
    --step-delta-mag $STEP_DELTA_MAG \
    --min-delta-mass $MIN_DELTA_MASS \
    --max-delta-mass $MAX_DELTA_MASS \
    --step-delta-mass $STEP_DELTA_MASS
    
$PYTHON $ZOOBOT_DIR/plotting.py \

$PYTHON $ZOOBOT_DIR/cluster_conf_matrix_code.py \
    --min-gal $MIN_GAL \
    --max-gal $MAX_GAL \
    --update-interval $UPDATE_INTERVAL \
    --pred-z $PRED_Z \
    --threshold-val $THRESHOLD_VAL \
    --delta-z $DELTA_Z \
    --delta-p $DELTA_P \
    --delta-mag $DELTA_MAG \
    --delta-mag $DELTA_MASS

$PYTHON $ZOOBOT_DIR/squid_diagrams.py \
    --min-gal $MIN_GAL_SQUID \
    --max-gal $MAX_GAL_SQUID \
    --delta-z $DELTA_Z \
    --delta-p $DELTA_P \
    --delta-mag $DELTA_MAG \
    --delta-mass $DELTA_MASS \
    --min_z $PRED_Z \
    --percent $PERCENT
