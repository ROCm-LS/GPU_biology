#!/bin/bash -l
#SBATCH --account=${PAWSEY_PROJECT}-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=boltz_prediction

# Load required modules
module load singularity/3.11.4-mpi

# Set container
containerImage=boltzv2.1.1_rocm6.4.sif
# Set input dir
INPUTDIR=$MYSCRATCH/boltz/inputs

# Set output directory
OUTDIR=$MYSCRATCH/boltz/${SLURM_JOB_ID}
mkdir -p ${OUTDIR}

# Set cache directory
CACHEDIR=$MYSCRATCH/boltz/cache
mkdir -p ${CACHEDIR}


srun -N 1 -n 1 -c 8 --gres=gpu:1 \
    singularity exec \
    ${containerImage} boltz predict \
    ${INPUTDIR}/test.yaml \
    --cache ${CACHEDIR} \
    --out_dir ${OUTDIR} \
    --no_kernels