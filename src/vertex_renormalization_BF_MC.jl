using Distributed

number_of_workers = nworkers()

printstyled("\nVertex renormalization BF monte carlo divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 6 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    STORE_FOLDER    MONTE_CARLO_ITERATIONS    COMPUTE_NUMBER_SPINS_CONFIGURATIONS    COMPUTE_MC_SPINS")

@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
@eval STORE_FOLDER = $(ARGS[4])
MONTE_CARLO_ITERATIONS = parse(Int, ARGS[5])
COMPUTE_NUMBER_SPINS_CONFIGURATIONS = parse(Bool, ARGS[6])
COMPUTE_MC_SPINS = parse(Bool, ARGS[7])

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
SPINS_MC_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/spins_indices"

if (COMPUTE_NUMBER_SPINS_CONFIGURATIONS)
    printstyled("computing number of spins configurations for jb=$(JB) up to K=$(CUTOFF)...\n\n"; bold=true, color=:cyan)
    mkpath(SPINS_CONF_FOLDER)
    @time vertex_renormalization_number_spins_configurations(CUTOFF, JB, SPINS_CONF_FOLDER)
    println("done\n")
end

if (COMPUTE_MC_SPINS)
    printstyled("sampling monte carlo spins for Nmc=$(MONTE_CARLO_ITERATIONS), jb=$(JB) up to K=$(CUTOFF)...\n"; bold=true, color=:cyan)
    mkpath(SPINS_MC_FOLDER)
    @time vertex_renormalization_MC_sampling(CUTOFF, MONTE_CARLO_ITERATIONS, JB, SPINS_MC_FOLDER)
    println("done\n")
end

STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/BF"
mkpath(STORE_AMPLS_FOLDER)

function vertex_renormalization_BF(cutoff, jb::HalfInt, Nmc::Int, vec_number_spins_configurations, spins_mc_folder::String, step=half(1))

    ampls = Float64[]
    stds = Float64[]

    # case pcutoff = 0
    push!(ampls, 0.0)
    push!(stds, 0.0)

    for pcutoff = step:step:cutoff

        # load MC bulk spins 
        @load "$(spins_mc_folder)/MC_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_draws

        # load MC intertwiners
        @load "$(spins_mc_folder)/MC_right_intertwiners_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_right_intertwiners_draws
        @load "$(spins_mc_folder)/MC_left_intertwiners_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_left_intertwiners_draws
        @load "$(spins_mc_folder)/MC_inner_intertwiners_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_inner_intertwiners_draws

        bulk_ampls = SharedArray{Float64}(Nmc)

        @time @sync @distributed for bulk_ampls_index in eachindex(bulk_ampls)

            jpink = MC_draws[1, bulk_ampls_index]
            jblue = MC_draws[2, bulk_ampls_index]
            jbrightgreen = MC_draws[3, bulk_ampls_index]
            jbrown = MC_draws[4, bulk_ampls_index]
            jdarkgreen = MC_draws[5, bulk_ampls_index]
            jviolet = MC_draws[6, bulk_ampls_index]
            jpurple = MC_draws[7, bulk_ampls_index]
            jred = MC_draws[8, bulk_ampls_index]
            jorange = MC_draws[9, bulk_ampls_index]
            jgrassgreen = MC_draws[10, bulk_ampls_index]

            rABr = MC_right_intertwiners_draws[1, bulk_ampls_index]
            rAEr = MC_right_intertwiners_draws[2, bulk_ampls_index]
            rbr = MC_right_intertwiners_draws[3, bulk_ampls_index]
            rCDr = MC_right_intertwiners_draws[4, bulk_ampls_index]
            rBCr = MC_right_intertwiners_draws[5, bulk_ampls_index]

            rABl = MC_left_intertwiners_draws[1, bulk_ampls_index]
            rAEl = MC_left_intertwiners_draws[2, bulk_ampls_index]
            rbl = MC_left_intertwiners_draws[3, bulk_ampls_index]
            rCDl = MC_left_intertwiners_draws[4, bulk_ampls_index]
            rBCl = MC_left_intertwiners_draws[5, bulk_ampls_index]

            rIu = MC_inner_intertwiners_draws[1, bulk_ampls_index]
            rIul = MC_inner_intertwiners_draws[2, bulk_ampls_index]
            rIbl = MC_inner_intertwiners_draws[3, bulk_ampls_index]
            rIbr = MC_inner_intertwiners_draws[4, bulk_ampls_index]
            rIur = MC_inner_intertwiners_draws[5, bulk_ampls_index]

            # compute vertex up
            r_u = ((0, 0), rABr[1], rIul[1], rIur[1], rBCl[1])
            v_u = vertex_BF_compute([jb, jb, jb, jb, jpink, jblue, jbrightgreen, jpurple, jgrassgreen, jred], r_u;)

            # compute vertex left
            r_l = ((0, 0), rAEr[1], rIbl[1], rIu[1], rABl[1])
            v_l = vertex_BF_compute([jb, jb, jb, jb, jbrown, jdarkgreen, jpink, jorange, jblue, jbrightgreen], r_l;)

            # compute vertex bottom-left
            r_bl = ((0, 0), rbr[1], rIbr[1], rIul[1], rAEl[1])
            v_bl = vertex_BF_compute([jb, jb, jb, jb, jviolet, jpurple, jbrown, jgrassgreen, jdarkgreen, jpink], r_bl;)

            # compute vertex bottom-right
            r_br = ((0, 0), rCDr[1], rIur[1], rIbl[1], rbl[1])
            v_br = vertex_BF_compute([jb, jb, jb, jb, jred, jorange, jviolet, jblue, jpurple, jbrown], r_br;)

            # compute vertex right
            r_r = ((0, 0), rBCr[1], rIu[1], rIbr[1], rCDl[1])
            v_r = vertex_BF_compute([jb, jb, jb, jb, jbrightgreen, jgrassgreen, jred, jdarkgreen, jorange, jviolet], r_r;)


            # outer "left" and "right" have same dimension  
            # TODO: improve efficiency of contraction (@turbo and @simd don't work with this synthax)

            for rABl_index in 1:rABl[2], rAEl_index in 1:rAEl[2], rbl_index in 1:rbl[2], rCDl_index in 1:rCDl[2], rBCl_index in 1:rBCl[2],
                rIu_index in 1:rIu[2], rIul_index in 1:rIul[2], rIbl_index in 1:rIbl[2], rIbr_index in 1:rIbr[2], rIur_index in 1:rIur[2]

                bulk_ampls[bulk_ampls_index] +=
                    v_u.a[rBCl_index, rIur_index, rIul_index, rABl_index, 1] *
                    v_l.a[rABl_index, rIu_index, rIbl_index, rAEl_index, 1] *
                    v_bl.a[rAEl_index, rIul_index, rIbr_index, rbl_index, 1] *
                    v_br.a[rbl_index, rIbl_index, rIur_index, rCDl_index, 1] *
                    v_r.a[rCDl_index, rIbr_index, rIu_index, rBCl_index, 1]

            end

            #= 
            Paranoid check (passed)

            AB right
            rABr = intertwiner_range(
                jpink,
                jblue,
                jbrightgreen,
                jb,
            )

            if (rABr != MC_right_intertwiners_draws[1, bulk_ampls_index])
                println("OPS")
            end

            # AB left
            rABl = intertwiner_range(
                jbrightgreen,
                jblue,
                jpink,
                jb,
            )

            if (rABl != MC_left_intertwiners_draws[1, bulk_ampls_index])
                println("OPS")
            end

            # AE right
            rAEr = intertwiner_range(
                jbrown,
                jdarkgreen,
                jpink,
                jb
            )

            if (rAEr != MC_right_intertwiners_draws[2, bulk_ampls_index])
                println("OPS")
            end

            # AE left
            rAEl = intertwiner_range(
                jpink,
                jdarkgreen,
                jbrown,
                jb
            )

            if (rAEl != MC_left_intertwiners_draws[2, bulk_ampls_index])
                println("OPS")
            end

            # bottom right
            rbr = intertwiner_range(
                jviolet,
                jpurple,
                jbrown,
                jb
            )

            if (rbr != MC_right_intertwiners_draws[3, bulk_ampls_index])
                println("OPS")
            end

            # bottom left
            rbl = intertwiner_range(
                jbrown,
                jpurple,
                jviolet,
                jb
            )

            if (rbl != MC_left_intertwiners_draws[3, bulk_ampls_index])
                println("OPS")
            end

            # CD right
            rCDr = intertwiner_range(
                jred,
                jorange,
                jviolet,
                jb
            )

            if (rCDr != MC_right_intertwiners_draws[4, bulk_ampls_index])
                println("OPS")
            end

            # CD left
            rCDl = intertwiner_range(
                jviolet,
                jorange,
                jred,
                jb
            )

            if (rCDl != MC_left_intertwiners_draws[4, bulk_ampls_index])
                println("OPS")
            end

            # BC right
            rBCr = intertwiner_range(
                jbrightgreen,
                jgrassgreen,
                jred,
                jb
            )

            if (rBCr != MC_right_intertwiners_draws[5, bulk_ampls_index])
                println("OPS")
            end

            # BC left
            rBCl = intertwiner_range(
                jred,
                jgrassgreen,
                jbrightgreen,
                jb
            )

            if (rBCl != MC_left_intertwiners_draws[5, bulk_ampls_index])
                println("OPS")
            end

            =#
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

printstyled("\nLoading CSV file with number of spins configurations for jb=$(JB) up to K=$(CUTOFF)...\n"; bold=true, color=:cyan)
vec_number_spins_configurations = vec(
    Matrix(
        DataFrame(
            CSV.File(
                "$(SPINS_CONF_FOLDER)/spins_configurations_cutoff_$(CUTOFF_FLOAT).csv"
            ),
        ),
    ),
)

printstyled("\nStarting computation with Nmc=$(MONTE_CARLO_ITERATIONS), jb=$(JB) up to K=$(CUTOFF)...\n"; bold=true, color=:cyan)
@time ampls, stds = vertex_renormalization_BF(CUTOFF, JB, MONTE_CARLO_ITERATIONS, vec_number_spins_configurations, SPINS_MC_FOLDER);

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
