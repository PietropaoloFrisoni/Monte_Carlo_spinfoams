# VEGAS in spinfoams

**_The Julia codes are parallelized on the available cores. In some cases we adopted a hybrid multilevel parallelization scheme, exploiting the available processes, threads and loop vectorization._** It is therefore advisable for the performance to use a number of workers \* threads equal to or less than the physical number of cores present on the system.

A full list of the employed Julia packages can be found in `./julia_codes/pkgs.jl`. **Before executing the codes, all packages must be installed.**

**The Julia's Just-in-Time compiler is such that the first execution of functions is considerably slower that following ones, and it also allocates much more memory**. To avoid this, you can use the [DaemonMode package](https://github.com/dmolina/DaemonMode.jl).

The [ipe](https://ipe.otfried.org/) file `diagrams_code_notation.ipe` contains the explicit structure of the spinfoams diagrams. The spins labels exactly match the ones implemented in the Julia scripts. We provide the computed data in the `data` folder.

## Usage:

To execute the Julia codes (on a single machine with the synthax below) you can run the following command:

```
$JULIA_EXECUTABLE_PATH   -p   [N-1]   --threads   [T]   $JULIA_CODE_PATH   $ARGS
```

where [N-1] is the number of workers and [T] is the number of threads.
