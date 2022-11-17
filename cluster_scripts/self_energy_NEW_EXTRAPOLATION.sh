#!/bin/bash
#SBATCH -A def-vidotto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --job-name=self_energy
#SBATCH --output=self_energy.log
#SBATCH --error=self_energy.err


# folders

ROOT_DIR=/home/frisus95/projects/def-vidotto/frisus95
JULIA_DIR=${ROOT_DIR}/julia-1.8.0
SL2CFOAM_DIR=${ROOT_DIR}/sl2cfoam-next-dev
FASTWIG_TABLES_PATH=${SL2CFOAM_DIR}/data_sl2cfoam

export LD_LIBRARY_PATH="${SL2CFOAM_DIR}/lib":$LD_LIBRARY_PATH
export JULIA_LOAD_PATH="${SL2CFOAM_DIR}/julia":$JULIA_LOAD_PATH

setrpaths.sh --path ${JULIA_DIR} [--add_origin]
setrpaths.sh --path ${SL2CFOAM_DIR} [--add_origin]

# otherwise libmpc.so.3 is not found
export LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/gentoo/2020/usr/lib":$LD_LIBRARY_PATH

echo "Running on: $SLURM_NODELIST"
echo


# parameters

JULIA_DIR=${JULIA_DIR}
BASE_DIR=${ROOT_DIR}/Monte_Carlo_spinfoams
SL2CFOAM_DATA_DIR=${SLURM_TMPDIR}
CUTOFF=10
JB=0.5
DL_MIN=0
DL_MAX=10
IMMIRZI=0.1
STORE_FOLDER=${BASE_DIR}
NUMBER_OF_TRIALS=20


# loading amplitudes and fastwig tables

echo "Extracting all vertex fulltensors to: $SLURM_TMPDIR ..."
echo

tar -xvf ${ROOT_DIR}/EPRL_vertex_precomputation_SHELL_MIN_0_SHELL_MAX_10_IMMIRZI_${IMMIRZI}_CUTOFF_10.tar.gz -C $SLURM_TMPDIR

echo "Copying fastwig tables to: $SLURM_TMPDIR ..."
echo

cp ${FASTWIG_TABLES_PATH}/* $SLURM_TMPDIR/



# running codes

#echo "Computing exact amplitudes..."
#echo

#${JULIA_DIR}/bin/julia -p $SLURM_TASKS_PER_NODE ${BASE_DIR}/src/self_energy_EPRL.jl ${SL2CFOAM_DATA_DIR} ${CUTOFF} ${JB} ${DL_MIN} ${DL_MAX} ${IMMIRZI} ${STORE_FOLDER}


echo "Computing amplitudes with Monte Carlo..."
echo

for MONTE_CARLO_ITERATIONS in 1000 10000 100000
do
${JULIA_DIR}/bin/julia -p $SLURM_TASKS_PER_NODE ${BASE_DIR}/src/self_energy_EPRL_MC_NEW_EXTRAPOLATION.jl ${SL2CFOAM_DATA_DIR} ${CUTOFF} ${JB} ${DL_MAX} ${IMMIRZI} ${STORE_FOLDER} ${MONTE_CARLO_ITERATIONS} ${NUMBER_OF_TRIALS}
done


echo "Completed"
echo
