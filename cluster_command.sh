#!/bin/bash
#SBATCH --job-name=understand                       # Job name
#SBATCH --output=understand_%A.log 
#SBATCH --mem=0   
#SBATCH -c 24                                    # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node
 
pwd; hostname; date

nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

THIS_REPO_DIR=/share/nas2/walml/repos/understanding_galaxies
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

# TODO crank it up
FITS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr5/fits_native/J000

SCALED_IMG_DIR=/share/nas2/walml/repos/understanding_galaxies/scaled

# TODO crank it up
MIN_GAL=0
MAX_GAL=5

MIN_GAL_SQUID=0
MAX_GAL_SQUID=5

PERCENT=66

PRED_Z=0.03
MAX_Z=0.12
STEP_SIZE=0.002

MIN_ALLOW_Z=0.05
MAX_ALLOW_Z=0.02

MIN_DELTA_Z=0.004
MAX_DELTA_Z=0.015
STEP_DELTA_Z=0.001
MIN_DELTA_P=0.014
MAX_DELTA_P=0.024
STEP_DELTA_P=0.001
MIN_DELTA_MAG=0.3
MAX_DELTA_MAG=1.3
STEP_DELTA_MAG=0.1
MIN_DELTA_MASS=0.5
MAX_DELTA_MASS=1.5
STEP_DELTA_MASS=0.1

UPDATE_INTERVAL=50
THRESHOLD_VAL=0.8

DELTA_Z=0.006
DELTA_P=0.016
DELTA_MAG=0.5
# TODO specify DELTA_MASS

# $PYTHON $THIS_REPO_DIR/creating_images_semester_two.py \
#     --fits-dir $FITS_DIR \
#     --save-dir $SCALED_IMG_DIR \
#     --max-redshift $MAX_Z \
#     --step-size $STEP_SIZE

# $PYTHON $THIS_REPO_DIR/make_predictions.py \
#     --batch-size 256 \
#     --input-dir $SCALED_IMG_DIR \
#     --checkpoint-loc /share/nas2/walml/repos/gz-decals-classifiers/results/tensorflow/all_campaigns_ortho_v2_train_only_m0/checkpoint \
#     --save-loc /share/nas2/walml/repos/understanding_galaxies/results/scaled_image_predictions.csv

# # # load predictions in convenient dataframe
# $PYTHON $THIS_REPO_DIR/create_dataframe.py \
#     --file-name /share/nas2/walml/repos/understanding_galaxies/results/scaled_image_predictions.hdf5 \
#     --min-allow-z $MIN_ALLOW_Z \
#     --max-allow-z $MAX_ALLOW_Z

# # apply debiasing method, to each galaxy, by sampling nearby galaxies
# $PYTHON $THIS_REPO_DIR/sampling_galaxies.py \
#     --min-gal $MIN_GAL \
#     --max-gal $MAX_GAL \
#     --min-delta-z $MIN_DELTA_Z \
#     --max-delta-z $MAX_DELTA_Z \
#     --step-delta-z $STEP_DELTA_Z \
#     --min-delta-p $MIN_DELTA_P \
#     --max-delta-p $MAX_DELTA_P \
#     --step-delta-p $STEP_DELTA_P \
#     --min-delta-mag $MIN_DELTA_MAG \
#     --max-delta-mag $MAX_DELTA_MAG \
#     --step-delta-mag $STEP_DELTA_MAG \
#     --min-delta-mass $MIN_DELTA_MASS \
#     --max-delta-mass $MAX_DELTA_MASS \
#     --step-delta-mass $STEP_DELTA_MASS
    
# $PYTHON $THIS_REPO_DIR/plotting.py

$PYTHON $THIS_REPO_DIR/cluster_conf_matrix_code.py \
    --min-gal $MIN_GAL \
    --max-gal $MAX_GAL \
    --update-interval $UPDATE_INTERVAL \
    --pred-z $PRED_Z \
    --threshold-val $THRESHOLD_VAL \
    --delta-z $DELTA_Z \
    --delta-p $DELTA_P \
    --delta-mag $DELTA_MAG 

# # evolution tracks
# TODO DELTA_MASS needs specifying above
# $PYTHON $THIS_REPO_DIR/squid_diagrams.py \
#     --min-gal $MIN_GAL_SQUID \
#     --max-gal $MAX_GAL_SQUID \
#     --delta-z $DELTA_Z \
#     --delta-p $DELTA_P \
#     --delta-mag $DELTA_MAG \
#     --delta-mass $DELTA_MASS \
#     --min_z $PRED_Z \
#     --percent $PERCENT
