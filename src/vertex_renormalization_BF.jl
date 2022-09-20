using Distributed

number_of_workers = nworkers()

printstyled("\nVertex renormalization BF divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 4 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    STORE_FOLDER")

@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
@eval STORE_FOLDER = $(ARGS[4])

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

printstyled("initializing library...\n\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, 0.123) # fictitious Immirzi 

SPINS_CONF_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/spins_configurations"
mkpath(SPINS_CONF_FOLDER)

if (!isfile("$(SPINS_CONF_FOLDER)/spins_configurations_cutoff_$(CUTOFF_FLOAT).csv"))
    printstyled("computing spins configurations for jb=$(JB) up to K=$(CUTOFF)...\n\n"; bold=true, color=:cyan)
    @time vertex_renormalization_number_spins_configurations(CUTOFF, JB, SPINS_CONF_FOLDER)
    println("done\n")
end

STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/vertex_renormalization/jb_$(JB_FLOAT)/exact/BF"
mkpath(STORE_AMPLS_FOLDER)

function vertex_renormalization_BF(cutoff, jb::HalfInt, spins_conf_folder::String, step=half(1))

    ampls = Float64[]

    for pcutoff = 0:step:cutoff

        # load bulk spins and intertwiners
        @load "$(spins_conf_folder)/configs_pcutoff_$(twice(pcutoff)/2).jld2" spins_configurations

        @load "$(spins_conf_folder)/right_intertwiners_configurations_pcutoff_$(twice(pcutoff)/2).jld2" right_intertwiners_configurations
        @load "$(spins_conf_folder)/left_intertwiners_configurations_pcutoff_$(twice(pcutoff)/2).jld2" left_intertwiners_configurations
        @load "$(spins_conf_folder)/inner_intertwiners_configurations_pcutoff_$(twice(pcutoff)/2).jld2" inner_intertwiners_configurations

        if isempty(spins_configurations)
            push!(ampls, 0.0)
            continue
        end

        @time tampl = @sync @distributed (+) for spins_index in eachindex(spins_configurations)

            jpink, jblue, jbrightgreen, jbrown, jdarkgreen, jviolet, jpurple, jred, jorange, jgrassgreen = spins_configurations[spins_index]

            rABr, rAEr, rbr, rCDr, rBCr = right_intertwiners_configurations[spins_index]
            rABl, rAEl, rbl, rCDl, rBCl = left_intertwiners_configurations[spins_index]
            rIu, rIul, rIbl, rIbr, rIur = inner_intertwiners_configurations[spins_index]

            # compute vertex up
            #r_u = ((0, 0), rABr[1], rIul[1], rIur[1], rBCl[1])
            v_u = vertex_BF_compute([jb, jb, jb, jb, jpink, jblue, jbrightgreen, jpurple, jgrassgreen, jred])

            # compute vertex left
            #r_l = ((0, 0), rAEr[1], rIbl[1], rIu[1], rABl[1])
            v_l = vertex_BF_compute([jb, jb, jb, jb, jbrown, jdarkgreen, jpink, jorange, jblue, jbrightgreen])

            # compute vertex bottom-left
            #r_bl = ((0, 0), rbr[1], rIbr[1], rIul[1], rAEl[1])
            v_bl = vertex_BF_compute([jb, jb, jb, jb, jviolet, jpurple, jbrown, jgrassgreen, jdarkgreen, jpink])

            # compute vertex bottom-right
            #r_br = ((0, 0), rCDr[1], rIur[1], rIbl[1], rbl[1])
            v_br = vertex_BF_compute([jb, jb, jb, jb, jred, jorange, jviolet, jblue, jpurple, jbrown])

            # compute vertex right
            #r_r = ((0, 0), rBCr[1], rIu[1], rIbr[1], rCDl[1])
            v_r = vertex_BF_compute([jb, jb, jb, jb, jbrightgreen, jgrassgreen, jred, jdarkgreen, jorange, jviolet])

            # face dims
            dfj = (2jpink + 1) * (2jblue + 1) * (2jbrightgreen + 1) * (2jbrown + 1) * (2jdarkgreen + 1) *
                  (2jviolet + 1) * (2jpurple + 1) * (2jred + 1) * (2jorange + 1) * (2jgrassgreen + 1)


            ##################################################################################################################################
            ### PRE-CONTRACTION
            ##################################################################################################################################

            # TODO: several optimizations, but for now readability is the priority

            # PHASE VERTEX UP

            W6j_matrix_up = Array{Float64}(undef, rBCl[2], rBCr[2])

            for rBCr_index = 1:rBCr[2]
                rBCr_intertw = from_index_to_intertwiner(rBCr, rBCr_index)
                for rBCl_index = 1:rBCl[2]
                    rBCl_intertw = from_index_to_intertwiner(rBCl, rBCl_index)
                    @inbounds W6j_matrix_up[rBCl_index, rBCr_index] =
                        float(wigner6j(jb, jbrightgreen, rBCl_intertw, jgrassgreen, jred, rBCr_intertw)) *
                        sqrt((2rBCl_intertw + 1) * (2rBCr_intertw + 1)) * (-1)^(jb + jbrightgreen + jred - jgrassgreen)
                end
            end

            vertex_up_pre_contracted = zeros(rBCr[2], rIur[2], rIul[2], rABr[2], 2)
            #check_size(vertex_up_pre_contracted, v_u.a)
            tensor_contraction!(vertex_up_pre_contracted, v_u.a, W6j_matrix_up)

            # PHASE VERTEX LEFT

            W6j_matrix_l = Array{Float64}(undef, rABl[2], rABr[2])

            for rABr_index = 1:rABr[2]
                rABr_intertw = from_index_to_intertwiner(rABr, rABr_index)
                for rABl_index = 1:rABl[2]
                    rABl_intertw = from_index_to_intertwiner(rABl, rABl_index)
                    @inbounds W6j_matrix_l[rABl_index, rABr_index] =
                        float(wigner6j(jb, jpink, rABl_intertw, jblue, jbrightgreen, rABr_intertw)) *
                        sqrt((2rABl_intertw + 1) * (2rABr_intertw + 1)) * (-1)^(jb + jpink + jbrightgreen - jblue)
                end
            end

            vertex_left_pre_contracted = zeros(rABr[2], rIu[2], rIbl[2], rAEr[2], 2)
            #check_size(vertex_left_pre_contracted, v_l.a)
            tensor_contraction!(vertex_left_pre_contracted, v_l.a, W6j_matrix_l)

            #println(vertex_left_pre_contracted)


            # PHASE VERTEX BOTTOM-LEFT

            W6j_matrix_bl = Array{Float64}(undef, rAEl[2], rAEr[2])

            for rAEr_index = 1:rAEr[2]
                rAEr_intertw = from_index_to_intertwiner(rAEr, rAEr_index)
                for rAEl_index = 1:rAEl[2]
                    rAEl_intertw = from_index_to_intertwiner(rAEl, rAEl_index)
                    @inbounds W6j_matrix_bl[rAEl_index, rAEr_index] =
                        float(wigner6j(jb, jbrown, rAEl_intertw, jdarkgreen, jpink, rAEr_intertw)) *
                        sqrt((2rAEl_intertw + 1) * (2rAEr_intertw + 1)) * (-1)^(jb + jbrown + jpink - jdarkgreen)
                end
            end

            vertex_bottom_left_pre_contracted = zeros(rAEr[2], rIul[2], rIbr[2], rbr[2], 2)
            tensor_contraction!(vertex_bottom_left_pre_contracted, v_bl.a, W6j_matrix_bl)


            # PHASE VERTEX BOTTOM-RIGHT

            W6j_matrix_br = Array{Float64}(undef, rbl[2], rbr[2])

            for rbr_index = 1:rbr[2]
                rbr_intertw = from_index_to_intertwiner(rbr, rbr_index)
                for rbl_index = 1:rbl[2]
                    rbl_intertw = from_index_to_intertwiner(rbl, rbl_index)
                    @inbounds W6j_matrix_br[rbl_index, rbr_index] =
                        float(wigner6j(jb, jviolet, rbl_intertw, jpurple, jbrown, rbr_intertw)) *
                        sqrt((2rbl_intertw + 1) * (2rbr_intertw + 1)) * (-1)^(jb + jbrown + jviolet - jpurple)
                end
            end

            vertex_bottom_right_pre_contracted = zeros(rbr[2], rIbl[2], rIur[2], rCDr[2], 2)
            #check_size(vertex_bottom_right_pre_contracted, v_br.a)
            tensor_contraction!(vertex_bottom_right_pre_contracted, v_br.a, W6j_matrix_br)


            # PHASE VERTEX RIGHT

            W6j_matrix_r = Array{Float64}(undef, rCDl[2], rCDr[2])

            for rCDr_index = 1:rCDr[2]
                rCDr_intertw = from_index_to_intertwiner(rCDr, rCDr_index)
                for rCDl_index = 1:rCDl[2]
                    rCDl_intertw = from_index_to_intertwiner(rCDl, rCDl_index)
                    @inbounds W6j_matrix_r[rCDl_index, rCDr_index] =
                        float(wigner6j(jb, jred, rCDl_intertw, jorange, jviolet, rCDr_intertw)) *
                        sqrt((2rCDl_intertw + 1) * (2rCDr_intertw + 1)) * (-1)^(jb + jviolet + jred - jorange)
                end
            end

            vertex_right_pre_contracted = zeros(rCDr[2], rIbr[2], rIu[2], rBCr[2], 2)
            #check_size(vertex_right_pre_contracted, v_r.a)
            tensor_contraction!(vertex_right_pre_contracted, v_r.a, W6j_matrix_r)

            ##################################################################################################################################
            ### FINAL INTERTWINER CONTRACTION
            ##################################################################################################################################

            amp = 0.0

            # after pre-contraction, outer left intertwiners don't exist anymore

            @inbounds  for rAB_index in 1:rABr[2], rAE_index in 1:rAEr[2], rb_index in 1:rbr[2], rCD_index in 1:rCDr[2], rBC_index in 1:rBCr[2],
                rIu_index in 1:rIu[2], rIul_index in 1:rIul[2], rIbl_index in 1:rIbl[2], rIbr_index in 1:rIbr[2], rIur_index in 1:rIur[2]

                rIu_intertw = from_index_to_intertwiner(rIu, rIu_index)
                rIul_intertw = from_index_to_intertwiner(rIul, rIul_index)
                rIbl_intertw = from_index_to_intertwiner(rIbl, rIbl_index)
                rIbr_intertw = from_index_to_intertwiner(rIbr, rIbr_index)
                rIur_intertw = from_index_to_intertwiner(rIur, rIur_index)

               @inbounds amp +=
                    vertex_up_pre_contracted[rBC_index, rIur_index, rIul_index, rAB_index, 1] * (-1)^(jb + jred + rIur_intertw) *
                vertex_left_pre_contracted[rAB_index, rIu_index, rIbl_index, rAE_index, 1] * (-1)^(jb + jbrightgreen + rIu_intertw) *
                vertex_bottom_left_pre_contracted[rAE_index, rIul_index, rIbr_index, rb_index, 1] * (-1)^(jb + jpink + rIul_intertw) *
                vertex_bottom_right_pre_contracted[rb_index, rIbl_index, rIur_index, rCD_index, 1] * (-1)^(jb + jbrown + rIbl_intertw) *
                vertex_right_pre_contracted[rCD_index, rIbr_index, rIu_index, rBC_index, 1] * (-1)^(jb + jviolet + rIbr_intertw) 

            end

            # face dims
            amp *= dfj #* (-1)^(2jpink + 2jblue + 2jbrightgreen + 2jbrown + 2jdarkgreen + 2jviolet + 2jpurple + 2jred + 2jorange + 2jgrassgreen)

            amp

        end

        amp = 0.0

        if isempty(ampls)
            ampl = tampl
        else
            ampl = ampls[end] + tampl
        end

        log("Amplitude at partial cutoff = $pcutoff: $(ampl)\n")
        push!(ampls, ampl)

    end # partial cutoffs loop

    ampls

end

printstyled("\nStarting computation with jb=$(JB) up to K=$(CUTOFF)...\n"; bold=true, color=:cyan)
@time ampls = vertex_renormalization_BF(CUTOFF, JB, SPINS_CONF_FOLDER);

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame([ampls], ["amp"])
CSV.write("$(STORE_AMPLS_FOLDER)/ampls_cutoff_$(CUTOFF_FLOAT)_ib_0.0.csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)