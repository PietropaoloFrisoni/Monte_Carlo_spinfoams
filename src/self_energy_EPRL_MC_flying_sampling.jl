using Distributed

number_of_workers = nworkers()

printstyled("\nSelf energy EPRL (flying sampling) monte carlo divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 9 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    DL_MIN    DL_MAX     IMMIRZI    STORE_FOLDER    MONTE_CARLO_ITERATIONS    COMPUTE_MC_INDICES")

@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
DL_MIN = parse(Int, ARGS[4])
DL_MAX = parse(Int, ARGS[5])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[6]))
@eval STORE_FOLDER = $(ARGS[7])
MONTE_CARLO_ITERATIONS = parse(Int, ARGS[8])
COMPUTE_MC_INDICES = parse(Bool, ARGS[9])

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
SPINS_MC_INDICES_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/spins_indices"

if (COMPUTE_MC_INDICES)
    printstyled("sampling monte carlo spins for Nmc $(MONTE_CARLO_ITERATIONS), jb $(JB) up to cutoff $(CUTOFF)...\n"; bold=true, color=:cyan)
    mkpath(SPINS_MC_INDICES_FOLDER)
    @time self_energy_MC_sampling(CUTOFF, MONTE_CARLO_ITERATIONS, JB, SPINS_MC_INDICES_FOLDER)
    println("done\n")
end

STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/EPRL/immirzi_$(IMMIRZI)"
mkpath(STORE_AMPLS_FOLDER)

function self_energy_EPRL(cutoff, jb::HalfInt, Dl::Int, Nmc::Int, vec_number_spins_configurations, spins_mc_indices_folder::String, step=half(1))

    ampls = Float64[]
    stds = Float64[]

    result_return = (ret=true, store=false, store_batches=false)

    # case pcutoff = 0
    push!(ampls, 0.0)
    push!(stds, 0.0)

    for pcutoff = step:step:cutoff

        @load "$(spins_mc_indices_folder)/MC_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_draws

        bulk_ampls = SharedArray{Float64}(Nmc)

        @time @sync @distributed for bulk_ampls_index in eachindex(bulk_ampls)

            j23 = MC_draws[1, bulk_ampls_index]
            j24 = MC_draws[2, bulk_ampls_index]
            j25 = MC_draws[3, bulk_ampls_index]
            j34 = MC_draws[4, bulk_ampls_index]
            j35 = MC_draws[5, bulk_ampls_index]
            j45 = MC_draws[6, bulk_ampls_index]

            # restricted range of intertwiners
            r2, _ = intertwiner_range(jb, j25, j24, j23)
            r3, _ = intertwiner_range(j23, jb, j34, j35)
            r4, _ = intertwiner_range(j34, j24, jb, j45)
            r5, _ = intertwiner_range(j45, j35, j25, jb)
            rm = ((0, 0), r2, r3, r4, r5)

            # compute vertex
            v = vertex_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], Dl, rm; result=result_return)

            # face dims
            dfj = (2j23 + 1) * (2j24 + 1) * (2j25 + 1) * (2j34 + 1) * (2j35 + 1) * (2j45 + 1)

            # contract
            bulk_ampls[bulk_ampls_index] = dfj * dot(v.a, v.a)

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

printstyled("\nLoading CSV file with number of spins configurations for jb $(JB) up to cutoff $(CUTOFF)...\n"; bold=true, color=:cyan)
vec_number_spins_configurations = vec(
    Matrix(
        DataFrame(
            CSV.File(
                "$(SPINS_CONF_FOLDER)/spins_configurations_cutoff_$(CUTOFF_FLOAT).csv"
            ),
        ),
    ),
)

printstyled("\nStarting computation with Nmc $(MONTE_CARLO_ITERATIONS), jb $(JB), Dl_min = $(DL_MIN), Dl_max = $(DL_MAX), Immirzi = $(IMMIRZI) up to cutoff $(CUTOFF)...\n"; bold=true, color=:cyan)

for Dl = DL_MIN:DL_MAX

    printstyled("\nCurrent Dl = $(Dl)...\n"; bold=true, color=:magenta)
    @time ampls, stds = self_energy_EPRL(CUTOFF, JB, Dl, MONTE_CARLO_ITERATIONS, vec_number_spins_configurations, SPINS_MC_INDICES_FOLDER)

    printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
    df = DataFrame([ampls, stds], ["amp", "std"])
    STORE_AMPLS_FOLDER_DL = "$(STORE_AMPLS_FOLDER)/Dl_$(Dl)"
    mkpath(STORE_AMPLS_FOLDER_DL)
    CSV.write("$(STORE_AMPLS_FOLDER_DL)/flying_sampling_ampls_cutoff_$(CUTOFF)_ib_0.0.csv", df)

end

printstyled("\nCompleted\n\n"; bold=true, color=:blue)