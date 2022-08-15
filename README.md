# Monte Carlo divergences

**The Julia codes are parallelized on the available cores.** It is therefore advisable for the performance to parallelize the codes keeping into account the number of CPU cores present on the system.

A full list of the employed Julia packages can be found in `./inc/pkgs.jl`. **Before executing the source codes, all packages must be installed.**

**The Julia's Just-in-Time compiler is such that the first execution of functions is considerably slower that following ones, and it also allocates much more memory**. To avoid this, you can use the [DaemonMode package](https://github.com/dmolina/DaemonMode.jl).

## Usage:

To execute the Julia codes (on a single machine with the synthax below) you can run the following command:

```
$JULIA_EXECUTABLE_PATH   -p   [N-1]    $JULIA_CODE_PATH   $ARGS
```

where [N-1] is the number of workers. The `ARGS` parameter depends on the specific kind of computation. We show some examples below.

### Example: BF Monte Carlo

```
ARGS = $CUTOFF    $JB    $STORE_FOLDER    $MONTE_CARLO_ITERATIONS    $COMPUTE_SPINS_CONFIGURATIONS    $COMPUTE_MC_INDICES
```

where:

- `CUTOFF`: the maximum value of bulks spins

- `JB`: value of boundary spins

- `STORE_FOLDER`: folder where data are saved

- `MONTE_CARLO_ITERATIONS`: number of monte carlo sampling

- `COMPUTE_SPINS_CONFIGURATIONS`: if `true` computes the spins configurations, if `false` it doesn't (in this case they must have been previously computed)

- `COMPUTE_MC_INDICES`: if `true` computes the monte carlo indices, if `false` it doesn't (in this case they must have been previously computed)