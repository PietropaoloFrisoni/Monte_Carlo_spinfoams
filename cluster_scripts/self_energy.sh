#!/bin/bash
#SBATCH -A def-vidotto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=0-0:15:00
#SBATCH --job-name=self_energy
#SBATCH --output=self_energy.log
#SBATCH --error=self_energy.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=pfrisoni@uwo.ca



# folders

ROOT_DIR=/home/frisus95/projects/def-vidotto/frisus95
JULIA_DIR=${ROOT_DIR}/julia-1.7.2
SL2CFOAM_DIR=${ROOT_DIR}/sl2cfoam_next
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
CODE_TO_RUN=self_energy_EPRL_MC_flying_sampling
SL2CFOAM_DATA_DIR=${SLURM_TMPDIR}
CUTOFF=10
JB=0.5
DL_MIN=0
DL_MAX=0
IMMIRZI=0.1
STORE_FOLDER=${BASE_DIR}
MONTE_CARLO_ITERATIONS=1000
COMPUTE_MC_SPINS=true

BOOSTER_DIR=${BASE_DIR}



echo "Copying fastwig tables to: $SLURM_TMPDIR ..."
echo

cp ${FASTWIG_TABLES_PATH}/* $SLURM_TMPDIR/

# booster extraction

#echo "Extracting previous boosters to: $SLURM_TMPDIR ..."
#echo

#tar -xvf ${BOOSTER_DIR}/${CODE_TO_RUN}_BOOSTERS_SHELL_MIN_${DL_MIN}_SHELL_MAX_${DL_MAX}_IMMIRZI_${IMMIRZI}_CUTOFF_${CUTOFF}.tar.gz -C $SLURM_TMPDIR/


# running code

echo "Running on: $SLURM_NODELIST"
echo

echo "Running: ${CODE_TO_RUN}"
echo

${JULIA_DIR}/bin/julia -p $SLURM_TASKS_PER_NODE ${BASE_DIR}/src/${CODE_TO_RUN}.jl ${SL2CFOAM_DATA_DIR} ${CUTOFF} ${JB} ${DL_MIN} ${DL_MAX} ${IMMIRZI} ${STORE_FOLDER} ${MONTE_CARLO_ITERATIONS} ${COMPUTE_MC_SPINS}



echo "Compressing and copying computed boosters to ${BOOSTER_DIR}..."
echo

tar -czvf ${BOOSTER_DIR}/${CODE_TO_RUN}_BOOSTERS_SHELL_MIN_${DL_MIN}_SHELL_MAX_${DL_MAX}_IMMIRZI_${IMMIRZI}_CUTOFF_${CUTOFF}.tar.gz $SLURM_TMPDIR/vertex


echo "Completed"
echo
