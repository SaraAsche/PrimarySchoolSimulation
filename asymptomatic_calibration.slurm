#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=nv-ibt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -c16
#SBATCH --mem=16000
#SBATCH --ntasks-per-node=16
#SBATCH --job-name="asympt"
#SBATCH --output=test-srun.out
#SBATCH --mail-user="saraj.asche@gmail.com"
#SBATCH --mail-type=ALL

WORKDIR=/cluster/work/saraja/PrimarySchoolSimulation
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module --ignore-cache load Anaconda3/2020.07
python disease_transmission.py

uname -a