#!/bin/bash
#SBATCH --job-name=bamford                       # Job name
#SBATCH --output=bamford_%A_%a.log 
#SBATCH --mem=10gb   
#SBATCH -c 4           
#SBATCH --nodes 1
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=72:00:00
#SBATCH --array=[0-20]  # must match length of PRED_Z_ARRAY

pwd; hostname; date

nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

THIS_REPO_DIR=/share/nas2/walml/repos/understanding_galaxies
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

#Test sample of galaxies to debias for parameter optimisation
MIN_GAL=0
MAX_GAL=250

#sets target z, maximum sim z and step size up to max_z
#PRED_Z=0.03
PRED_Z_ARRAY=($(seq 0.015 0.005 0.115)) # Should give values from 0.02 through to 0.12
PRED_Z=${PRED_Z_ARRAY[$SLURM_ARRAY_TASK_ID]}
echo Using pred_z $PRED_Z

#N-D box dimensions
# chosen from previous optimal search
DELTA_Z=0.005
DELTA_P=0.016
DELTA_MAG=0.5
DELTA_MASS=1.0
DELTA_CONC=0.05

# apply debiasing method, to each galaxy, by sampling nearby galaxies
$PYTHON $THIS_REPO_DIR/debias_to_target_redshift.py \
   --min-gal $MIN_GAL \
   --max-gal $MAX_GAL \
   --delta-z $DELTA_Z \
   --delta-p $DELTA_P \
   --delta-mag $DELTA_MAG \
   --delta-mass $DELTA_MASS \
   --delta-conc $DELTA_CONC \
   --pred-z $PRED_Z
    