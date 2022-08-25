available_cpus = length(Sys.cpu_info())
if (number_of_workers > available_cpus)
    printstyled("WARNING: you are using more resources than available cores on this system. Performances will be affected\n\n"; bold=true, color=:red)
end

# initialize library
function init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER::String, Immirzi::Float64)
    isMPI = @ccall SL2Cfoam.clib.sl2cfoam_is_MPI()::Bool
    isMPI && error("MPI version not allowed")
    conf = SL2Cfoam.Config(VerbosityOff, HighAccuracy, 200, 0)
    SL2Cfoam.cinit(DATA_SL2CFOAM_FOLDER, Immirzi, conf)
    # disable C library automatic parallelization
    SL2Cfoam.set_OMP(false)
end