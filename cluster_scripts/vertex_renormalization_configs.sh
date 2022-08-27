#!/bin/bash
#SBATCH -A def-vidotto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-0:00:00
#SBATCH --job-name=vertex_renorm_configs_number
#SBATCH --output=vertex_renorm_configs_number.log
#SBATCH --error=vertex_renorm_configs_number.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=pfrisoni@uwo.ca



# folders

ROOT_DIR=/home/frisus95/projects/def-vidotto/frisus95
JULIA_DIR=${ROOT_DIR}/julia-*
SL2CFOAM_DIR=${ROOT_DIR}/sl2cfoam*
FASTWIG_TABLES_PATH=${SL2CFOAM_DIR}/data_sl2cfoam

export LD_LIBRARY_PATH="${SL2CFOAM_DIR}/lib":$LD_LIBRARY_PATH
export JULIA_LOAD_PATH="${SL2CFOAM_DIR}/julia":$JULIA_LOAD_PATH

setrpaths.sh --path ${JULIA_DIR} [--add_origin]
setrpaths.sh --path ${SL2CFOAM_DIR} [--add_origin]

# otherwise libmpc.so.3 is not found
export LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/gentoo/2020/usr/lib":$LD_LIBRARY_PATH



# parameters

JULIA_DIR=${JULIA_DIR}
BASE_DIR=${ROOT_DIR}/Monte_Carlo_spinfoams
CODE_TO_RUN=vertex_renormalization_BF_MC
SL2CFOAM_DATA_DIR=${SL2CFOAM_DIR}/data_sl2cfoam
CUTOFF=10
JB=0.5
STORE_FOLDER=${BASE_DIR}
MONTE_CARLO_ITERATIONS=1000
COMPUTE_NUMBER_SPINS_CONFIGURATIONS=true
COMPUTE_MC_INDICES=false



# running code

echo "Running on: $SLURM_NODELIST"
echo

echo "Running: ${CODE_TO_RUN}"
echo

${JULIA_DIR}/bin/julia ${BASE_DIR}/src/${CODE_TO_RUN}.jl ${SL2CFOAM_DATA_DIR} ${CUTOFF} ${JB} ${STORE_FOLDER} ${MONTE_CARLO_ITERATIONS} ${COMPUTE_NUMBER_SPINS_CONFIGURATIONS} ${COMPUTE_MC_INDICES}

echo "Completed"
echo
