using Distributed

number_of_workers = nworkers()

printstyled("\nSelf energy EPRL monte carlo divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 9 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    DL_MIN    DL_MAX     IMMIRZI    STORE_FOLDER    MONTE_CARLO_ITERATIONS    NUMBER_OF_TRIALS")

@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
DL_MIN = parse(Int, ARGS[4])
DL_MAX = parse(Int, ARGS[5])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[6]))
@eval STORE_FOLDER = $(ARGS[7])
MONTE_CARLO_ITERATIONS = parse(Int, ARGS[8])
NUMBER_OF_TRIALS = parse(Int, ARGS[9])

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

# vector with weights on internal faces
# each internal face with spin j has dimension (2j+1)^(weight)
@everywhere FACE_WEIGHTS_VEC = [4 / 3, 7 / 6, 1.0, 1 / 3]

printstyled("initializing library with immirzi=$(IMMIRZI)...\n\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, IMMIRZI)

SPINS_CONF_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/spins_configurations"
SPINS_MC_INDICES_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/spins_indices"
STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/EPRL/immirzi_$(IMMIRZI)"
mkpath(STORE_AMPLS_FOLDER)

function self_energy_EPRL_MC(cutoff, jb::HalfInt, Dl::Int, Nmc::Int, vec_number_spins_configurations, spins_mc_folder::String, face_weights_vec, step=half(1))

    result_return = (ret=true, store=false, store_batches=false)

    total_number_of_ampls = Int(2 * cutoff + 1)
    boundary_dim = Int(2 * jb + 1)
    number_of_weights = size(face_weights_vec)[1]

    # tensor with amplitudes and stds
    ampls_tensor = zeros(total_number_of_ampls, number_of_weights, boundary_dim)
    stds_tensor = zeros(total_number_of_ampls, number_of_weights, boundary_dim)

    # case pcutoff = 0 AND amplitude = 0
    # TODO: generalize to take into account integer boundary spin case
    for pcutoff = step:step:cutoff

        index_pcutoff = Int(2 * pcutoff + 1)

        @load "$(spins_mc_folder)/MC_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_draws

        bulk_ampls = SharedArray{Float64}(Nmc, number_of_weights, boundary_dim)

        @time @sync @distributed for bulk_ampls_index = 1:Nmc

            j23 = MC_draws[1, bulk_ampls_index]
            j24 = MC_draws[2, bulk_ampls_index]
            j25 = MC_draws[3, bulk_ampls_index]
            j34 = MC_draws[4, bulk_ampls_index]
            j35 = MC_draws[5, bulk_ampls_index]
            j45 = MC_draws[6, bulk_ampls_index]

            # compute vertex
            v = vertex_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], Dl; result=result_return)

            # face base dims
            dfj = (2j23 + 1) * (2j24 + 1) * (2j25 + 1) * (2j34 + 1) * (2j35 + 1) * (2j45 + 1)

            # contract
            for weight_index = 1:number_of_weights

                weight = face_weights_vec[weight_index]

                for ib_index = 1:boundary_dim
                    bulk_ampls[bulk_ampls_index, weight_index, ib_index] = dfj^(weight) * dot(v.a[:, :, :, :, ib_index], v.a[:, :, :, :, ib_index])
                end

            end

        end

        # volume normalization factor
        total_number_conf = vec_number_spins_configurations[index_pcutoff] - vec_number_spins_configurations[index_pcutoff-1]

        for weight_index = 1:number_of_weights

            for ib_index = 1:boundary_dim

                tampl = mean(bulk_ampls[:, weight_index, ib_index])

                tampl_var = 0.0
                for n = 1:Nmc
                    tampl_var += (bulk_ampls[n, weight_index, ib_index] - tampl)^2
                end
                tampl_var /= (Nmc - 1)

                tampl *= total_number_conf
                tampl_std = sqrt(tampl_var * (total_number_conf^2) / Nmc)

                ampls_tensor[index_pcutoff, weight_index, ib_index] = ampls_tensor[index_pcutoff-1, weight_index, ib_index] + tampl
                stds_tensor[index_pcutoff, weight_index, ib_index] = stds_tensor[index_pcutoff-1, weight_index, ib_index] + tampl_std

            end

        end

        log("\nAt partial cutoff = $pcutoff the ampls matrix is:\n")
        display(ampls_tensor[index_pcutoff, :, :])
        println("\nwhile std matrix is:\n")
        display(stds_tensor[index_pcutoff, :, :])
        println("\n")

    end # partial cutoffs loop

    ampls_tensor, stds_tensor

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

total_number_of_ampls = Int(2 * CUTOFF_FLOAT + 1)
boundary_dim = Int(2 * JB + 1)
number_of_weights = size(FACE_WEIGHTS_VEC)[1]

printstyled("\nstarting computation with Nmc=$(MONTE_CARLO_ITERATIONS), n_trials=$(NUMBER_OF_TRIALS), jb=$(JB), Dl_min=$(DL_MIN), Dl_max=$(DL_MAX), Immirzi=$(IMMIRZI) up to K=$(CUTOFF)...\n"; bold=true, color=:cyan)

for current_trial = 1:NUMBER_OF_TRIALS

    printstyled("\nsampling $(MONTE_CARLO_ITERATIONS) bulk spins configurations in trial $(current_trial)...\n"; bold=true, color=:bold)
    mkpath(SPINS_MC_INDICES_FOLDER)
    @time self_energy_MC_sampling(CUTOFF, MONTE_CARLO_ITERATIONS, JB, SPINS_MC_INDICES_FOLDER)

    for Dl = DL_MIN:DL_MAX

        STORE_AMPLS_FOLDER_DL = "$(STORE_AMPLS_FOLDER)/Dl_$(Dl)"

        printstyled("\ncurrent Dl = $(Dl)...\n"; bold=true, color=:magenta)

        printstyled("\ncomputing amplitudes...\n"; bold=true, color=:blue)
        @time ampls_tensor, stds_tensor = self_energy_EPRL_MC(CUTOFF, JB, Dl, MONTE_CARLO_ITERATIONS, vec_number_spins_configurations, SPINS_MC_INDICES_FOLDER, FACE_WEIGHTS_VEC)

        printstyled("\nsaving dataframe...\n"; bold=true, color=:cyan)

        for weight_index = 1:number_of_weights

            weight = round(FACE_WEIGHTS_VEC[weight_index], digits=3)

            for ib_index = 1:boundary_dim

                STORE_AMPLS_FINAL_FOLDER = "$(STORE_AMPLS_FOLDER_DL)/weight_$(weight)/ib_$(ib_index-1)"
                mkpath(STORE_AMPLS_FINAL_FOLDER)

                ampls = ampls_tensor[:, weight_index, ib_index]
                df = DataFrame([ampls], ["amp"])

                CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/ampls_cutoff_$(CUTOFF).csv", df)

            end

        end

    end

end

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
