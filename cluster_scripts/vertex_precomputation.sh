#!/bin/bash
#SBATCH -A def-vidotto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=0-00:40:00
#SBATCH --job-name=EPRL_vertex_precomputation
#SBATCH --output=EPRL_vertex_precomputation.log
#SBATCH --error=EPRL_vertex_precomputation.err


# folders

ROOT_DIR=/home/frisus95/projects/def-vidotto/frisus95
JULIA_DIR=${ROOT_DIR}/julia-1.7.2
SL2CFOAM_DIR=${ROOT_DIR}/sl2cfoam-next-dev
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
CODE_TO_RUN=EPRL_vertex_precomputation
SL2CFOAM_DATA_DIR=${SLURM_TMPDIR}/localscratch/frisus95.42647778.0
CUTOFF=10
JB=0.5
DL_MIN=0
DL_MAX=0
IMMIRZI=0.1
STORE_FOLDER=${BASE_DIR}
COMPUTE_SPINS_CONFIGURATIONS=true

BOOSTER_DIR=${BASE_DIR}


# booster extraction

echo "Extracting previous boosters to: $SLURM_TMPDIR ..."
echo

tar -xvf ${BASE_DIR}/self_energy_EPRL_MC_flying_sampling_BOOSTERS_SHELL_MIN_${DL_MIN}_SHELL_MAX_10_IMMIRZI_${IMMIRZI}_CUTOFF_${CUTOFF}.tar.gz -C $SLURM_TMPDIR




# fastwig tables copy

echo "Copying fastwig tables to: $SLURM_TMPDIR ..."
echo

cp ${FASTWIG_TABLES_PATH}/* $SLURM_TMPDIR/localscratch/frisus95.42647778.0





# running code

echo "Running on: $SLURM_NODELIST"
echo

echo "Running: ${CODE_TO_RUN}"
echo

${JULIA_DIR}/bin/julia -p $SLURM_TASKS_PER_NODE ${BASE_DIR}/src/${CODE_TO_RUN}.jl ${SL2CFOAM_DATA_DIR} ${CUTOFF} ${JB} ${DL_MIN} ${DL_MAX} ${IMMIRZI} ${STORE_FOLDER} ${COMPUTE_SPINS_CONFIGURATIONS}



echo "Compressing and copying computed amplitudes to ${BOOSTER_DIR}..."
echo

cd $SLURM_TMPDIR/localscratch/frisus95.42647778.0

tar -czvf ${BOOSTER_DIR}/${CODE_TO_RUN}_SHELL_MIN_${DL_MIN}_SHELL_MAX_${DL_MAX}_IMMIRZI_${IMMIRZI}_CUTOFF_${CUTOFF}.tar.gz vertex/


echo "Completed"
echo
