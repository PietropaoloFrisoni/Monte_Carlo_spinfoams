# Monte Carlo divergences

**The Julia codes are parallelized on the available cores.** It is therefore advisable for the performance to parallelize the codes keeping into account the number of CPU cores present on the system.

A full list of the employed Julia packages can be found in `./inc/pkgs.jl`. **Before executing the source codes, all packages must be installed.**

**The Julia's Just-in-Time compiler is such that the first execution of functions is considerably slower that following ones, and it also allocates much more memory**. To avoid this, you can use the [DaemonMode package](https://github.com/dmolina/DaemonMode.jl).

## Usage

To execute the Julia codes (on a single machine with the synthax below) you can run the following command:

```
$JULIA_EXECUTABLE_PATH   -p   [N-1]    $JULIA_CODE_PATH   $ARGS
```

where [N-1] is the number of workers. The `ARGS` parameter depends on the specific kind of computation. We show an example below.

### Example: Self energy EPRL with monte carlo sampling

```
ARGS = DATA_SL2CFOAM_FOLDER    CUTOFF    JB    DL_MIN    DL_MAX     IMMIRZI    STORE_FOLDER    MONTE_CARLO_ITERATIONS    NUMBER_OF_TRIALS
```

where:

- `DATA_SL2CFOAM_FOLDER`: folder with fastwigxj tables where boosters (and possibly vertices) are retrieved/stored

- `CUTOFF`: the maximum value of bulks spins

- `JB`: value of boundary spins

- `DL_MIN`: minimum value of truncation parameter over auxiliary spins

- `DL_MAX`: maximum value of truncation parameter over auxiliary spins

- `IMMIRZI`: value of Immirzi parameter

- `STORE_FOLDER`: folder where data are saved

- `MONTE_CARLO_ITERATIONS`: number of monte carlo sampling for each trial

- `NUMBER_OF_TRIALS`: number of trials

Additionally, you can specify the weights $\mu_1, \mu_2 \dots \mu_n$ on bulk faces *inside* the code script, with the vector `FACE_WEIGHTS_VEC`.

Each bulk face with spin $j$ has dimension $(2j+1)^{\mu}$, and the code computes all amplitudes with provided weights.