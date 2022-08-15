# initialize library
function init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER::String, Immirzi::Float64)
    isMPI = @ccall SL2Cfoam.clib.sl2cfoam_is_MPI()::Bool
    isMPI && error("MPI version not allowed")
    conf = SL2Cfoam.Config(VerbosityOff, HighAccuracy, 200, 0)
    SL2Cfoam.cinit(DATA_SL2CFOAM_FOLDER, Immirzi, conf)
    # disable C library automatic parallelization
    SL2Cfoam.set_OMP(false)
end

# logging function (flushing needed)
function log(x...)
    println("[ ", now(), " ] - ", join(x, " ")...)
    flush(stdout)
end

# comunicate between processes
macro retrieve_from_process(p, obj, mod=:Main)
    quote
        remotecall_fetch($(esc(p)), $(esc(mod)), $(QuoteNode(obj))) do m, o
            Core.eval(m, o)
        end
    end
end