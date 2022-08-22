using Distributed

number_of_workers = nworkers()

printstyled("\nEPRL vertex precomputation parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 8 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    DL_MIN    DL_MAX     IMMIRZI    STORE_FOLDER    COMPUTE_SPINS_CONFIGURATIONS")

@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
DL_MIN = parse(Int, ARGS[4])
DL_MAX = parse(Int, ARGS[5])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[6]))
@eval STORE_FOLDER = $(ARGS[7])
COMPUTE_SPINS_CONFIGURATIONS = parse(Bool, ARGS[8])

printstyled("precompiling packages and source codes...\n"; bold=true, color=:cyan)
@everywhere begin
    include("../inc/pkgs.jl")
    include("init.jl")
    include("spins_configurations.jl")
end
println("done\n")

CUTOFF_FLOAT = parse(Float64, ARGS[2])
CUTOFF = HalfInt(CUTOFF_FLOAT)

JB_FLOAT = parse(Float64, ARGS[3])
JB = HalfInt(JB_FLOAT)

printstyled("initializing library with immirzi $(IMMIRZI)...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, IMMIRZI)
println("done\n")

SPINS_CONF_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/spins_configurations"

if (COMPUTE_SPINS_CONFIGURATIONS)
    printstyled("computing spins configurations for jb $(JB) up to cutoff $(CUTOFF)...\n\n"; bold=true, color=:cyan)
    mkpath(SPINS_CONF_FOLDER)
    @time self_energy_spins_conf(CUTOFF, JB, SPINS_CONF_FOLDER)
    println("done\n")
end

function EPRL_vertex_precomputation(cutoff, jb::HalfInt, Dl::Int, spins_conf_folder::String, step=half(1))

    result_return = (ret=false, store=true, store_batches=false)

    for pcutoff = 0:step:cutoff

        @load "$(spins_conf_folder)/configs_pcutoff_$(twice(pcutoff)/2).jld2" spins_configurations

        @sync @distributed for spins in spins_configurations

            j23, j24, j25, j34, j35, j45 = spins

            # compute vertex
            vertex_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], Dl; result=result_return)

        end

    end # partial cutoffs loop

end

printstyled("\nStarting computation with jb=$(JB), Dl_min=$(DL_MIN), Dl_max=$(DL_MAX), Immirzi=$(IMMIRZI) up to K=$(CUTOFF)...\n"; bold=true, color=:cyan)

for Dl = DL_MIN:DL_MAX

    printstyled("\nCurrent Dl = $(Dl)...\n"; bold=true, color=:magenta)
    @time EPRL_vertex_precomputation(CUTOFF, JB, Dl, SPINS_CONF_FOLDER)

end

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
