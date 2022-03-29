#!/bin/bash
#SBATCH --job-name=understand                       # Job name
#SBATCH --output=understand_%A.log 
#SBATCH --mem=40gb   
#SBATCH -c 24                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=72:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node
 
pwd; hostname; date

nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

# TODO crank it up
THIS_REPO_DIR=/share/nas2/walml/repos/understanding_galaxies
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python
FITS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr5/fits_native    
SCALED_IMG_DIR=/share/nas2/walml/repos/understanding_galaxies/scaled_with_resizing
PREDICTIONS_DIR=/share/nas2/walml/repos/understanding_galaxies/results/latest_scaled_predictions

#THIS_REPO_DIR=/Users/adamfoster/Documents/MPhysProject/understanding_galaxies
#PYTHON=/Users/adamfoster/opt/anaconda3/envs/ZooBot/bin/python
#PREDICTIONS_DIR=/Users/adamfoster/Documents/MPhysProject/understanding_galaxies/results/latest_scaled_predictions

# TODO crank it up

#Test sample of galaxies to debias for parameter optimisation
MIN_GAL=0
MAX_GAL=250

#Sets the galaxies to be debiased for conf matrix
MIN_GAL_MATRIX=0
MAX_GAL_MATRIX=1000

#Sets the galaxies to be batched for each node when run in parallel (total of 23422 unique galaxies in full_data_1m_with_resizing)
BATCH_GAL_MIN=15000      #if running over multiple nodes, would increase as [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500]
BATCH_GAL_MAX=17500   #if running over multiple nodes, would increase as [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, :]

#hard cap on number of gals being simulated
GALS_TO_SIM=10

#galaxies to plot squid graphs for
MIN_GAL_SQUID=0
MAX_GAL_SQUID=10

#accepeted percentage area under pdf 
PERCENT=66

#decimal to round to
ROUNDING=0.005

#sets target z, maximum sim z and step size up to max_z
PRED_Z=0.03
MAX_Z=0.12
STEP_SIZE=0.004 #could prob make a lot smaller (0.005?)

#cut on which unsimulated galaxies to select (only those with low redshifts)
MIN_ALLOW_Z=0.02
MAX_ALLOW_Z=0.05

#sets of max and min hyperparams with step sizes for selecting optimum
MIN_DELTA_Z=0.005
MAX_DELTA_Z=0.015
STEP_DELTA_Z=0.002
MIN_DELTA_P=0.010
MAX_DELTA_P=0.020
STEP_DELTA_P=0.002
MIN_DELTA_MAG=0.5
MAX_DELTA_MAG=1.5
STEP_DELTA_MAG=0.2
MIN_DELTA_MASS=0.5
MAX_DELTA_MASS=1.5
STEP_DELTA_MASS=0.2
MIN_DELTA_CONC=0.05
MAX_DELTA_CONC=0.15
STEP_DELTA_CONC=0.02

#number of gals iterated per update and threshold for confident prediction
UPDATE_INTERVAL=50
THRESHOLD_VAL=0.8

#N-D box dimensions
DELTA_Z=0.006
DELTA_P=0.016
DELTA_MAG=0.5
DELTA_MASS=1.0
DELTA_CONC=0.1

#Sets the initial constraints wehn tuning hyperparams
INITIAL_DELTA_P=0.016
INITIAL_DELTA_MAG=0.5
INITIAL_DELTA_MASS=1.0
INITIAL_DELTA_CONC=0.1

#define which morphology squid diagrams to produce
MORPHOLOGY='featured-or-disk' #smooth, featured-or-disk, artifact

#$PYTHON $THIS_REPO_DIR/creating_images_semester_two.py \
#    --fits-dir $FITS_DIR \
#    --save-dir $SCALED_IMG_DIR \
#    --max-redshift $MAX_Z \
#    --step-size $STEP_SIZE \
#    --max-gals-to-sim $GALS_TO_SIM

#$PYTHON $THIS_REPO_DIR/make_predictions.py \
#     --batch-size 256 \
#     --image-dir $SCALED_IMG_DIR \
#     --checkpoint-loc /share/nas2/walml/repos/gz-decals-classifiers/results/tensorflow/all_campaigns_ortho_v2_train_only_m0/checkpoint \
#     --save-dir $PREDICTIONS_DIR

#  load predictions in convenient dataframe
#$PYTHON $THIS_REPO_DIR/create_dataframe.py \
#   --predictions-dir $PREDICTIONS_DIR \
#   --max-allow-z $MAX_ALLOW_Z \
#   --min-allow-z $MIN_ALLOW_Z 

# apply debiasing method, to each galaxy, by sampling nearby galaxies
# $PYTHON $THIS_REPO_DIR/sampling_galaxies.py \
#    --max-gal $MAX_GAL \
#    --min-gal $MIN_GAL \
#    --min-delta-z $MIN_DELTA_Z \
#    --max-delta-z $MAX_DELTA_Z \
#    --step-delta-z $STEP_DELTA_Z \
#    --min-delta-p $MIN_DELTA_P \
#    --max-delta-p $MAX_DELTA_P \
#    --step-delta-p $STEP_DELTA_P \
#    --min-delta-mag $MIN_DELTA_MAG \
#    --max-delta-mag $MAX_DELTA_MAG \
#    --step-delta-mag $STEP_DELTA_MAG \
#    --min-delta-mass $MIN_DELTA_MASS \
#    --max-delta-mass $MAX_DELTA_MASS \
#    --step-delta-mass $STEP_DELTA_MASS \
#    --min-delta-conc $MIN_DELTA_CONC \
#    --max-delta-conc $MAX_DELTA_CONC \
#    --step-delta-conc $STEP_DELTA_CONC \
#    --initial-delta-p $INITIAL_DELTA_P \
#    --initial-delta-mag $INITIAL_DELTA_MAG \
#    --initial-delta-mass $INITIAL_DELTA_MASS \
#    --initial-delta-conc $INITIAL_DELTA_CONC
    
# $PYTHON $THIS_REPO_DIR/plotting.py

 $PYTHON $THIS_REPO_DIR/debiasing_predictions.py \
     --batch-gal-min $BATCH_GAL_MIN \
     --batch-gal-max $BATCH_GAL_MAX \
     --update-interval $UPDATE_INTERVAL \
     --pred-z $PRED_Z \
     --delta-z $DELTA_Z \
     --delta-p $DELTA_P \
     --delta-mag $DELTA_MAG \
     --delta-mass $DELTA_MASS \
     --delta-conc $DELTA_CONC

# $PYTHON $THIS_REPO_DIR/conf_matrix_new.py \
#     --min-gal $MIN_GAL_MATRIX \
#     --max-gal $MAX_GAL_MATRIX \
#     --update-interval $UPDATE_INTERVAL \
#     --pred-z $PRED_Z \
#     --threshold-val $THRESHOLD_VAL \
#     --delta-z $DELTA_Z \
#     --delta-p $DELTA_P \
#     --delta-mag $DELTA_MAG \
#     --delta-mass $DELTA_MASS \
#     --delta-conc $DELTA_CONC

# # evolution tracks
# # TODO DELTA_MASS needs specifying above
# $PYTHON $THIS_REPO_DIR/squid_diagrams_new.py \
#     --min-gal $MIN_GAL_SQUID \
#     --max-gal $MAX_GAL_SQUID \
#     --delta-z $DELTA_Z \
#     --delta-p $DELTA_P \
#     --delta-mag $DELTA_MAG \
#     --delta-mass $DELTA_MASS \
#     --min-z $PRED_Z \
#     --percent $PERCENT \
#     --morphology $MORPHOLOGY \
#     --max-z $MAX_Z \
#     --delta-conc $DELTA_CONC

# # bamford_plots
# $PYTHON $THIS_REPO_DIR/bamford_plots.py \
#     --update-interval $UPDATE_INTERVAL \
#     --threshold-val $THRESHOLD_VAL \
#     --delta-z $DELTA_Z \
#     --delta-p $DELTA_P \
#     --delta-mag $DELTA_MAG \
#     --delta-mass $DELTA_MASS \
#     --delta-conc $DELTA_CONC \
#     --rounding $ROUNDING

# Testing that the shell script works (Leave this hashed out)
#$PYTHON $THIS_REPO_DIR/test.py \
