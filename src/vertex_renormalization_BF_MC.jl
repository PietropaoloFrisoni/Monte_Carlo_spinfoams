using Distributed

number_of_workers = nworkers()

printstyled("\nVertex renormalization BF monte carlo divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 6 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    STORE_FOLDER    MONTE_CARLO_ITERATIONS    COMPUTE_NUMBER_SPINS_CONFIGURATIONS    COMPUTE_MC_INDICES")

@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
@eval STORE_FOLDER = $(ARGS[4])
MONTE_CARLO_ITERATIONS = parse(Int, ARGS[5])
COMPUTE_NUMBER_SPINS_CONFIGURATIONS = parse(Bool, ARGS[6])
COMPUTE_MC_INDICES = parse(Bool, ARGS[7])

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

printstyled("initializing library...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, 0.123) # fictitious Immirzi 
println("done\n")

SPINS_CONF_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/spins_configurations"
SPINS_MC_INDICES_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/spins_indices"

if (COMPUTE_NUMBER_SPINS_CONFIGURATIONS)
    printstyled("computing number of spins configurations for jb $(JB) up to cutoff $(CUTOFF)...\n\n"; bold=true, color=:cyan)
    mkpath(SPINS_CONF_FOLDER)
    @time vertex_renormalization_number_spins_configurations(CUTOFF, JB, SPINS_CONF_FOLDER)
    println("done\n")
end

if (COMPUTE_MC_INDICES)
    printstyled("sampling monte carlo spins indices for Nmc $(MONTE_CARLO_ITERATIONS), jb $(JB) up to cutoff $(CUTOFF)...\n"; bold=true, color=:cyan)
    mkpath(SPINS_MC_INDICES_FOLDER)
    @time vertex_renormalization_MC_sampling(CUTOFF, MONTE_CARLO_ITERATIONS, JB, SPINS_MC_INDICES_FOLDER)
    println("done\n")
end

STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/BF"
mkpath(STORE_AMPLS_FOLDER)

function vertex_renormalization_BF()

    # TODO write

end

# TODO write



printstyled("\nCompleted\n\n"; bold=true, color=:blue)
