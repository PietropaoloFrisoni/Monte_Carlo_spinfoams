using Distributed

number_of_workers = nworkers()

printstyled("\nSelf energy EPRL monte carlo divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 10 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    DL_MIN    DL_MAX     IMMIRZI    STORE_FOLDER    MONTE_CARLO_ITERATIONS    NUMBER_OF_TRIALS    OVERWRITE_PREVIOUS_TRIALS")

@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
DL_MIN = parse(Int, ARGS[4])
DL_MAX = parse(Int, ARGS[5])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[6]))
@eval STORE_FOLDER = $(ARGS[7])
MONTE_CARLO_ITERATIONS = parse(Int, ARGS[8])
NUMBER_OF_TRIALS = parse(Int, ARGS[9])
OVERWRITE_PREVIOUS_TRIALS = parse(Bool, ARGS[10])

printstyled("precompiling packages and source codes...\n\n"; bold=true, color=:cyan)
@everywhere begin
    include("../inc/pkgs.jl")
    include("init.jl")
    include("utilities.jl")
    include("spins_configurations.jl")
end

CUTOFF_FLOAT = parse(Float64, ARGS[2])
CUTOFF = HalfInt(CUTOFF_FLOAT)

JB_FLOAT = parse(Float64, ARGS[3])
JB = HalfInt(JB_FLOAT)

printstyled("initializing library with immirzi=$(IMMIRZI)...\n\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, IMMIRZI)

SPINS_CONF_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/spins_configurations"
SPINS_MC_INDICES_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/spins_indices"
STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/EPRL/immirzi_$(IMMIRZI)"
mkpath(STORE_AMPLS_FOLDER)

function self_energy_EPRL_MC(cutoff, jb::HalfInt, Dl::Int, Nmc::Int, vec_number_spins_configurations, spins_mc_folder::String, step=half(1))

    ampls = Float64[]
    stds = Float64[]

    result_return = (ret=true, store=false, store_batches=false)

    # case pcutoff = 0
    # TODO: generalize this to take into account integer case
    push!(ampls, 0.0)
    push!(stds, 0.0)

    for pcutoff = step:step:cutoff

        @load "$(spins_mc_folder)/MC_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_draws

        bulk_ampls = SharedArray{Float64}(Nmc)

        @time @sync @distributed for bulk_ampls_index in eachindex(bulk_ampls)

            j23 = MC_draws[1, bulk_ampls_index]
            j24 = MC_draws[2, bulk_ampls_index]
            j25 = MC_draws[3, bulk_ampls_index]
            j34 = MC_draws[4, bulk_ampls_index]
            j35 = MC_draws[5, bulk_ampls_index]
            j45 = MC_draws[6, bulk_ampls_index]

            # compute vertex
            v = vertex_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], Dl; result=result_return)

            # face dims
            dfj = (2j23 + 1) * (2j24 + 1) * (2j25 + 1) * (2j34 + 1) * (2j35 + 1) * (2j45 + 1)

            # contract
            bulk_ampls[bulk_ampls_index] = dfj * dot(v.a[:, :, :, :, 1], v.a[:, :, :, :, 1])

        end

        tampl = mean(bulk_ampls)

        tampl_var = 0.0
        for i = 1:Nmc
            tampl_var += (bulk_ampls[i] - tampl)^2
        end
        tampl_var /= (Nmc - 1)

        # normalize
        index_cutoff = Int(2 * pcutoff + 1)
        tnconf = vec_number_spins_configurations[index_cutoff] - vec_number_spins_configurations[index_cutoff-1]
        tampl *= tnconf
        tampl_std = sqrt(tampl_var * (tnconf^2) / Nmc)

        if isempty(ampls)
            ampl = tampl
            std = tampl_std
        else
            ampl = ampls[end] + tampl
            std = stds[end] + tampl_std
        end

        log("Amplitude at partial cutoff = $pcutoff: $(ampl)")
        push!(ampls, ampl)
        println("Amplitude std at partial cutoff = $pcutoff: $(std)\n")
        push!(stds, std)

    end # partial cutoffs loop

    ampls, stds

end

if (!isfile("$(SPINS_CONF_FOLDER)/spins_configurations_cutoff_$(CUTOFF_FLOAT).csv"))
    printstyled("computing spins configurations for jb=$(JB) up to K=$(CUTOFF)...\n\n"; bold=true, color=:cyan)
    @time self_energy_spins_conf(CUTOFF, JB, SPINS_CONF_FOLDER)
    println("done\n")
end

printstyled("loading CSV file with number of spins configurations for jb=$(JB) up to K=$(CUTOFF)...\n"; bold=true, color=:cyan)
vec_number_spins_configurations = vec(
    Matrix(
        DataFrame(
            CSV.File(
                "$(SPINS_CONF_FOLDER)/spins_configurations_cutoff_$(CUTOFF_FLOAT).csv"
            ),
        ),
    ),
)

printstyled("\nstarting computation with Nmc=$(MONTE_CARLO_ITERATIONS), n_trials=$(NUMBER_OF_TRIALS), jb=$(JB), Dl_min=$(DL_MIN), Dl_max=$(DL_MAX), Immirzi=$(IMMIRZI) up to K=$(CUTOFF)...\n"; bold=true, color=:cyan)

for Dl = DL_MIN:DL_MAX

    printstyled("\ncurrent Dl = $(Dl)...\n"; bold=true, color=:magenta)

    STORE_AMPLS_FOLDER_DL = "$(STORE_AMPLS_FOLDER)/Dl_$(Dl)"
    mkpath(STORE_AMPLS_FOLDER_DL)

    number_of_previously_stored_trials = 0

    if (!OVERWRITE_PREVIOUS_TRIALS)
        number_of_previously_stored_trials += file_count(STORE_AMPLS_FOLDER_DL)
        printstyled("\n$(number_of_previously_stored_trials) trials have been previously stored, and $(NUMBER_OF_TRIALS) will be added\n"; bold=true, color=:cyan)
    end

    for current_trial = 1:NUMBER_OF_TRIALS

        printstyled("\nsampling $(MONTE_CARLO_ITERATIONS) bulk spins configurations in trial $(current_trial)...\n"; bold=true, color=:bold)
        mkpath(SPINS_MC_INDICES_FOLDER)
        @time self_energy_MC_sampling(CUTOFF, MONTE_CARLO_ITERATIONS, JB, SPINS_MC_INDICES_FOLDER)

        printstyled("\ncomputing amplitudes...\n"; bold=true, color=:blue)
        @time ampls, stds = self_energy_EPRL_MC(CUTOFF, JB, Dl, MONTE_CARLO_ITERATIONS, vec_number_spins_configurations, SPINS_MC_INDICES_FOLDER)

        printstyled("\nsaving dataframe...\n"; bold=true, color=:cyan)
        df = DataFrame([ampls, stds], ["amp", "std"])

        CSV.write("$(STORE_AMPLS_FOLDER_DL)/ampls_cutoff_$(CUTOFF)_ib_0.0_trial_$(number_of_previously_stored_trials + current_trial).csv", df)

    end

end

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
