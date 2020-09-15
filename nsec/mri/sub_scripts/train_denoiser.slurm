#!/bin/bash
#SBATCH --job-name=denoiser_score_est_train     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=20           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=60:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=denoiser_score_est_train%A_%a.out # nom du fichier de sortie
#SBATCH --error=denoiser_score_est_train%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-2

set -x
cd $WORK/score-estimation-comparison

#!/bin/bash
module purge
module load python/3.7.5 cuda/10.1.2 cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda
conda activate fastmri-tf-2.1.0

export FASTMRI_DATA_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/

opt[0]=""
opt[2]="-nps 60"
opt[1]="-bs 64 -n 5000"

srun python ./nsec/mri/denoiser_training.py ${opt[$SLURM_ARRAY_TASK_ID]}
