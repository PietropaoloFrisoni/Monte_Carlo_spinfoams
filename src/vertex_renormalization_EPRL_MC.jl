using Distributed

number_of_workers = nworkers()

printstyled("\nVertex renormalization EPRL monte carlo divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

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

# vector with weights on internal faces
# each internal face with spin j has dimension (2j+1)^(weight)
@everywhere FACE_WEIGHTS_VEC = [1.1, 1.3, 1.5]

printstyled("initializing library with immirzi=$(IMMIRZI)...\n\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, IMMIRZI)

SPINS_CONF_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/spins_configurations"
SPINS_MC_INDICES_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/spins_indices"
STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/monte_carlo/Nmc_$(MONTE_CARLO_ITERATIONS)/EPRL/immirzi_$(IMMIRZI)"
mkpath(STORE_AMPLS_FOLDER)

# CUTOFF, JB, MONTE_CARLO_ITERATIONS, vec_number_spins_configurations, SPINS_MC_INDICES_FOLDER, current_trial, FACE_WEIGHTS_VEC)

function vertex_renormalization_EPRL(cutoff, jb::HalfInt, Dl::Int, Nmc::Int, vec_number_spins_configurations, spins_mc_folder::String, trial::Int, face_weights_vec, step=half(1))

    total_number_of_ampls = Int(2 * cutoff + 1)
    boundary_dim = Int(2 * jb + 1)
    number_of_weights = size(face_weights_vec)[1]

    # tensor with amplitudes and stds
    ampls_tensor = zeros(total_number_of_ampls, number_of_weights, boundary_dim)
    stds_tensor = zeros(total_number_of_ampls, number_of_weights, boundary_dim)

    result_return = (ret=true, store=false, store_batches=false)

    # case pcutoff = 0
    # TODO: generalize this to take into account integer case
    for pcutoff = step:step:cutoff

        index_pcutoff = Int(2 * pcutoff + 1)

        # load MC bulk spins 
        @load "$(spins_mc_folder)/MC_draws_pcutoff_$(twice(pcutoff)/2)_trial_$(trial).jld2" MC_draws

        # load MC intertwiners
        @load "$(spins_mc_folder)/MC_right_intertwiners_draws_pcutoff_$(twice(pcutoff)/2)_trial_$(trial).jld2" MC_right_intertwiners_draws
        @load "$(spins_mc_folder)/MC_left_intertwiners_draws_pcutoff_$(twice(pcutoff)/2)_trial_$(trial).jld2" MC_left_intertwiners_draws
        @load "$(spins_mc_folder)/MC_inner_intertwiners_draws_pcutoff_$(twice(pcutoff)/2)_trial_$(trial).jld2" MC_inner_intertwiners_draws

        bulk_ampls = SharedArray{Float64}(Nmc, number_of_weights, boundary_dim)
        bulk_ampls[:, :, :] .= 0.0

        @time @sync @distributed for bulk_ampls_index = 1:Nmc

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
            #r_u = ((0, 0), rABr[1], rIul[1], rIur[1], rBCl[1])
            v_u = vertex_compute([jb, jb, jb, jb, jpink, jblue, jbrightgreen, jpurple, jgrassgreen, jred], Dl; result=result_return)

            # compute vertex left
            #r_l = ((0, 0), rAEr[1], rIbl[1], rIu[1], rABl[1])
            v_l = vertex_compute([jb, jb, jb, jb, jbrown, jdarkgreen, jpink, jorange, jblue, jbrightgreen], Dl; result=result_return)

            # compute vertex bottom-left
            #r_bl = ((0, 0), rbr[1], rIbr[1], rIul[1], rAEl[1])
            v_bl = vertex_compute([jb, jb, jb, jb, jviolet, jpurple, jbrown, jgrassgreen, jdarkgreen, jpink], Dl; result=result_return)

            # compute vertex bottom-right
            #r_br = ((0, 0), rCDr[1], rIur[1], rIbl[1], rbl[1])
            v_br = vertex_compute([jb, jb, jb, jb, jred, jorange, jviolet, jblue, jpurple, jbrown], Dl; result=result_return)

            # compute vertex right
            #r_r = ((0, 0), rBCr[1], rIu[1], rIbr[1], rCDl[1])
            v_r = vertex_compute([jb, jb, jb, jb, jbrightgreen, jgrassgreen, jred, jdarkgreen, jorange, jviolet], Dl; result=result_return)

            # face dims
            dfj = (2jpink + 1) * (2jblue + 1) * (2jbrightgreen + 1) * (2jbrown + 1) * (2jdarkgreen + 1) *
                  (2jviolet + 1) * (2jpurple + 1) * (2jred + 1) * (2jorange + 1) * (2jgrassgreen + 1)

            df_phase = (-1)^(2 * (jpink + jblue + jbrightgreen + jbrown + jdarkgreen + jviolet + jpurple + jred + jorange + jgrassgreen))


            # PHASE VERTEX UP

            W6j_matrix = Array{Float64}(undef, rBCl[2], rBCr[2])

            for rBCr_index = 1:rBCr[2]
                rBCr_intertw = from_index_to_intertwiner(rBCr, rBCr_index)
                for rBCl_index = 1:rBCl[2]
                    rBCl_intertw = from_index_to_intertwiner(rBCl, rBCl_index)
                    W6j_matrix[rBCl_index, rBCr_index] =
                        float(wigner6j(jb, jbrightgreen, rBCl_intertw, jgrassgreen, jred, rBCr_intertw)) *
                        sqrt((2rBCl_intertw + 1) * (2rBCr_intertw + 1)) * (-1)^(jb + jbrightgreen + jred - jgrassgreen)
                end
            end

            vertex_up_pre_contracted = zeros(rBCr[2], rIur[2], rIul[2], rABr[2], boundary_dim)
            #check_size(vertex_up_pre_contracted, v_u.a)
            tensor_contraction!(vertex_up_pre_contracted, v_u.a, W6j_matrix)


            # PHASE VERTEX LEFT

            W6j_matrix = Array{Float64}(undef, rABl[2], rABr[2])

            for rABr_index = 1:rABr[2]
                rABr_intertw = from_index_to_intertwiner(rABr, rABr_index)
                for rABl_index = 1:rABl[2]
                    rABl_intertw = from_index_to_intertwiner(rABl, rABl_index)
                    W6j_matrix[rABl_index, rABr_index] =
                        float(wigner6j(jb, jpink, rABl_intertw, jblue, jbrightgreen, rABr_intertw)) *
                        sqrt((2rABl_intertw + 1) * (2rABr_intertw + 1)) * (-1)^(jb + jpink + jbrightgreen - jblue)
                end
            end

            vertex_left_pre_contracted = zeros(rABr[2], rIu[2], rIbl[2], rAEr[2], boundary_dim)
            #check_size(vertex_left_pre_contracted, v_l.a)
            tensor_contraction!(vertex_left_pre_contracted, v_l.a, W6j_matrix)


            # PHASE VERTEX BOTTOM-LEFT

            W6j_matrix = Array{Float64}(undef, rAEl[2], rAEr[2])

            for rAEr_index = 1:rAEr[2]
                rAEr_intertw = from_index_to_intertwiner(rAEr, rAEr_index)
                for rAEl_index = 1:rAEl[2]
                    rAEl_intertw = from_index_to_intertwiner(rAEl, rAEl_index)
                    W6j_matrix[rAEl_index, rAEr_index] =
                        float(wigner6j(jb, jbrown, rAEl_intertw, jdarkgreen, jpink, rAEr_intertw)) *
                        sqrt((2rAEl_intertw + 1) * (2rAEr_intertw + 1)) * (-1)^(jb + jbrown + jpink - jdarkgreen)
                end
            end

            vertex_bottom_left_pre_contracted = zeros(rAEr[2], rIul[2], rIbr[2], rbr[2], boundary_dim)
            tensor_contraction!(vertex_bottom_left_pre_contracted, v_bl.a, W6j_matrix)


            # PHASE VERTEX BOTTOM-RIGHT

            W6j_matrix = Array{Float64}(undef, rbl[2], rbr[2])

            for rbr_index = 1:rbr[2]
                rbr_intertw = from_index_to_intertwiner(rbr, rbr_index)
                for rbl_index = 1:rbl[2]
                    rbl_intertw = from_index_to_intertwiner(rbl, rbl_index)
                    W6j_matrix[rbl_index, rbr_index] =
                        float(wigner6j(jb, jviolet, rbl_intertw, jpurple, jbrown, rbr_intertw)) *
                        sqrt((2rbl_intertw + 1) * (2rbr_intertw + 1)) * (-1)^(jb + jbrown + jviolet - jpurple)
                end
            end

            vertex_bottom_right_pre_contracted = zeros(rbr[2], rIbl[2], rIur[2], rCDr[2], boundary_dim)
            #check_size(vertex_bottom_right_pre_contracted, v_br.a)
            tensor_contraction!(vertex_bottom_right_pre_contracted, v_br.a, W6j_matrix)


            # PHASE VERTEX RIGHT

            W6j_matrix = Array{Float64}(undef, rCDl[2], rCDr[2])

            for rCDr_index = 1:rCDr[2]
                rCDr_intertw = from_index_to_intertwiner(rCDr, rCDr_index)
                for rCDl_index = 1:rCDl[2]
                    rCDl_intertw = from_index_to_intertwiner(rCDl, rCDl_index)
                    W6j_matrix[rCDl_index, rCDr_index] =
                        float(wigner6j(jb, jred, rCDl_intertw, jorange, jviolet, rCDr_intertw)) *
                        sqrt((2rCDl_intertw + 1) * (2rCDr_intertw + 1)) * (-1)^(jb + jviolet + jred - jorange)
                end
            end

            vertex_right_pre_contracted = zeros(rCDr[2], rIbr[2], rIu[2], rBCr[2], boundary_dim)
            #check_size(vertex_right_pre_contracted, v_r.a)
            tensor_contraction!(vertex_right_pre_contracted, v_r.a, W6j_matrix)


            # FINAL INTERTWINER CONTRACTION
            # after pre-contraction, outer left intertwiners don't exist anymore

            # TODO: consider using multiple threads and/or vectorize loops
            for weight_index = 1:number_of_weights

                weight = face_weights_vec[weight_index]

                for ib_index = 1:boundary_dim

                    for rAB_index in 1:rABr[2], rAE_index in 1:rAEr[2], rb_index in 1:rbr[2], rCD_index in 1:rCDr[2], rBC_index in 1:rBCr[2],
                        rIu_index in 1:rIu[2], rIul_index in 1:rIul[2], rIbl_index in 1:rIbl[2], rIbr_index in 1:rIbr[2], rIur_index in 1:rIur[2]

                        rIu_intertw = from_index_to_intertwiner(rIu, rIu_index)
                        rIul_intertw = from_index_to_intertwiner(rIul, rIul_index)
                        rIbl_intertw = from_index_to_intertwiner(rIbl, rIbl_index)
                        rIbr_intertw = from_index_to_intertwiner(rIbr, rIbr_index)
                        rIur_intertw = from_index_to_intertwiner(rIur, rIur_index)

                        bulk_ampls[bulk_ampls_index, weight_index, ib_index] +=
                            vertex_up_pre_contracted[rBC_index, rIur_index, rIul_index, rAB_index, ib_index] * (-1)^(jb + jred + rIur_intertw) *
                            vertex_left_pre_contracted[rAB_index, rIu_index, rIbl_index, rAE_index, ib_index] * (-1)^(jb + jbrightgreen + rIu_intertw) *
                            vertex_bottom_left_pre_contracted[rAE_index, rIul_index, rIbr_index, rb_index, ib_index] * (-1)^(jb + jpink + rIul_intertw) *
                            vertex_bottom_right_pre_contracted[rb_index, rIbl_index, rIur_index, rCD_index, ib_index] * (-1)^(jb + jbrown + rIbl_intertw) *
                            vertex_right_pre_contracted[rCD_index, rIbr_index, rIu_index, rBC_index, ib_index] * (-1)^(jb + jviolet + rIbr_intertw)

                    end

                    # face dims
                    bulk_ampls[bulk_ampls_index, weight_index, ib_index] *= dfj^(weight) * df_phase

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
    @time vertex_renormalization_number_spins_configurations(CUTOFF, JB, SPINS_CONF_FOLDER)
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

printstyled("\nparallel sampling of $(NUMBER_OF_TRIALS) trials, each one with $(MONTE_CARLO_ITERATIONS) bulk spins configurations...\n"; bold=true, color=:bold)
mkpath(SPINS_MC_INDICES_FOLDER)

@time @sync @distributed for current_trial = 1:NUMBER_OF_TRIALS
    vertex_renormalization_MC_sampling(CUTOFF, MONTE_CARLO_ITERATIONS, JB, SPINS_MC_INDICES_FOLDER, current_trial)
end

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

        printstyled("\ncomputing amplitudes in trial $(current_trial)...\n"; bold=true, color=:blue)
        @time ampls_tensor, stds_tensor = vertex_renormalization_EPRL(CUTOFF, JB, Dl, MONTE_CARLO_ITERATIONS, vec_number_spins_configurations, SPINS_MC_INDICES_FOLDER, current_trial, FACE_WEIGHTS_VEC)

        printstyled("\nsaving dataframe...\n"; bold=true, color=:cyan)

        for weight_index = 1:number_of_weights

            weight = round(FACE_WEIGHTS_VEC[weight_index], digits=3)

            for ib_index = 1:boundary_dim

                STORE_AMPLS_FINAL_FOLDER = "$(STORE_AMPLS_FOLDER_DL)/weight_$(weight)/ib_$(ib_index-1)"
                mkpath(STORE_AMPLS_FINAL_FOLDER)

                ampls = ampls_tensor[:, weight_index, ib_index]
                stds = stds_tensor[:, weight_index, ib_index]

                df = DataFrame([ampls, stds], ["amp", "std"])
                CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/ampls_cutoff_$(CUTOFF)_trial_$(current_trial).csv", df)

            end

        end

    end

end

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
