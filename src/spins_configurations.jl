######################################################################################################################
### SELF ENERGY 
######################################################################################################################

# store self-energy spins configurations for all partial cutoffs up to cutoff
function self_energy_spins_conf(cutoff, jb::HalfInt, configs_path::String, step=half(1))

    total_number_spins_configs = Int[]

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate a list of all spins to compute
        spins_configurations = NTuple{6,HalfInt8}[]

        for j23::HalfInt = 0:step:pcutoff, j24::HalfInt = 0:step:pcutoff, j25::HalfInt = 0:step:pcutoff,
            j34::HalfInt = 0:step:pcutoff, j35::HalfInt = 0:step:pcutoff, j45::HalfInt = 0:step:pcutoff

            # skip if computed in lower partial cutoff
            j23 <= (pcutoff - step) && j24 <= (pcutoff - step) &&
                j25 <= (pcutoff - step) && j34 <= (pcutoff - step) &&
                j35 <= (pcutoff - step) && j45 <= (pcutoff - step) && continue

            # skip if any intertwiner range empty
            r2, _ = intertwiner_range(jb, j25, j24, j23)
            r3, _ = intertwiner_range(j23, jb, j34, j35)
            r4, _ = intertwiner_range(j34, j24, jb, j45)
            r5, _ = intertwiner_range(j45, j35, j25, jb)

            isempty(r2) && continue
            isempty(r3) && continue
            isempty(r4) && continue
            isempty(r5) && continue

            # must be computed
            push!(spins_configurations, (j23, j24, j25, j34, j35, j45))

        end

        # store partial spins configurations at pctuoff
        @save "$(configs_path)/configs_pcutoff_$(twice(pcutoff)/2).jld2" spins_configurations

        total_conf = size(spins_configurations)[1]

        if (!isempty(total_number_spins_configs))
            total_conf += total_number_spins_configs[end]
        end

        log("configurations at partial cutoff = $pcutoff: $(total_conf)\n")
        push!(total_number_spins_configs, total_conf)

    end

    # store total spins configurations at total cutoff
    total_number_spins_configs_df = DataFrame(total_spins_configs=total_number_spins_configs)
    CSV.write("$(configs_path)/spins_configurations_cutoff_$(twice(cutoff)/2).csv", total_number_spins_configs_df)

end


# store Monte Carlo self-energy indices for all partial cutoffs up to cutoff
function self_energy_MC_spins_indices(cutoff, Nmc::Int, jb::HalfInt, configs_path::String, MC_configs_path::String, step=half(1))

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        file_to_load = "$(configs_path)/configs_pcutoff_$(twice(pcutoff)/2).jld2"

        if (!isfile(file_to_load))
            error("spins configurations for pcutoff $(twice(pcutoff)/2) must be computed first\n")
        end

        # load list of partial spins to sample from
        @load "$(file_to_load)" spins_configurations

        if (isempty(spins_configurations))
            MC_indices_spins_configurations = Array{Int32}(undef, 0)
            @save "$(MC_configs_path)/indices_pcutoff_$(twice(pcutoff)/2).jld2" MC_indices_spins_configurations
        else

            MC_indices_spins_configurations = Array{Int32}(undef, Nmc)

            number_spins_configs = size(spins_configurations)[1]
            distr = Uniform(1, number_spins_configs + 1)
            draw_float_sample = Array{Float64}(undef, 1)

            for n = 1:Nmc
                rand!(distr, draw_float_sample)
                MC_indices_spins_configurations[n] = round(Int64, floor(draw_float_sample[1]))
            end

            # store MC spins indices 
            @save "$(MC_configs_path)/indices_pcutoff_$(twice(pcutoff)/2).jld2" MC_indices_spins_configurations

        end

    end

end


# store Monte Carlo self-energy spins configurations for all partial cutoffs up to cutoff
function self_energy_MC_sampling(cutoff, Nmc::Int, jb::HalfInt, MC_configs_path::String, step=half(1))

    MC_draws = Array{HalfInt8}(undef, 6, Nmc)
    draw_float_sample = Array{Float64}(undef, 1)

    # loop over partial cutoffs
    for pcutoff = step:step:cutoff

        distr = Uniform(0, Int(2 * pcutoff + 1))

        for n = 1:Nmc

            while true

                # sampling j23, j24, j25 for the 4j with spins [j23, j24, j25, jb]
                for i = 1:3
                    rand!(distr, draw_float_sample)
                    MC_draws[i, n] = half(floor(draw_float_sample[1]))
                end

                # sampling j34, j35 for the 4j with spins [j34, j35, jb, j23]
                for i = 4:5
                    rand!(distr, draw_float_sample)
                    MC_draws[i, n] = half(floor(draw_float_sample[1]))
                end

                # sampling j45 for the 4j with spins [j45, jb, j24, j34]
                for i = 6:6
                    rand!(distr, draw_float_sample)
                    MC_draws[i, n] = half(floor(draw_float_sample[1]))
                end

                # skip if computed in lower partial cutoff
                MC_draws[1, n] <= (pcutoff - step) && MC_draws[2, n] <= (pcutoff - step) &&
                    MC_draws[3, n] <= (pcutoff - step) && MC_draws[4, n] <= (pcutoff - step) &&
                    MC_draws[5, n] <= (pcutoff - step) && MC_draws[6, n] <= (pcutoff - step) && continue

                # check that 4j with spins [j45, jb, j24, j34] satisfies triangular inequalities
                r, _ = intertwiner_range(
                    MC_draws[6, n],
                    jb,
                    MC_draws[2, n],
                    MC_draws[4, n]
                )
                isempty(r) && continue

                # check that 4j with spins [j34, j35, jb, j23] satisfies triangular inequalities
                r, _ = intertwiner_range(
                    MC_draws[4, n],
                    MC_draws[5, n],
                    jb,
                    MC_draws[1, n]
                )
                isempty(r) && continue

                # check that 4j with spins [j23, j24, j25, jb] satisfies triangular inequalities
                r, _ = intertwiner_range(
                    MC_draws[1, n],
                    MC_draws[2, n],
                    MC_draws[3, n],
                    jb
                )
                isempty(r) && continue

                # check that 4j with spins [jb, j25, j35, j45] satisfies triangular inequalities
                r, _ = intertwiner_range(
                    jb,
                    MC_draws[3, n],
                    MC_draws[5, n],
                    MC_draws[6, n],
                )
                isempty(r) && continue

                # bulk spins have passed all tests -> must be computed
                break

            end

        end

        # store MC spins indices 
        @save "$(MC_configs_path)/MC_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_draws

    end

end

######################################################################################################################
### VERTEX RENORMALIZATION 
######################################################################################################################

# store the number of vertex renormalization spins configurations for all partial cutoffs up to cutoff
function vertex_renormalization_number_spins_configurations(cutoff, jb::HalfInt, configs_path::String, step=half(1))

    total_number_spins_configs = Int[]

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        counter_partial_configurations = 0

        for jpink::HalfInt = 0:step:pcutoff, jblue::HalfInt = 0:step:pcutoff, jbrightgreen::HalfInt = 0:step:pcutoff,
            jbrown::HalfInt = 0:step:pcutoff, jdarkgreen::HalfInt = 0:step:pcutoff, jviolet::HalfInt = 0:step:pcutoff,
            jpurple::HalfInt = 0:step:pcutoff, jred::HalfInt = 0:step:pcutoff, jorange::HalfInt = 0:step:pcutoff,
            jgrassgreen::HalfInt = 0:step:pcutoff

            # skip if computed in lower partial cutoff
            jpink <= (pcutoff - step) && jblue <= (pcutoff - step) && jbrightgreen <= (pcutoff - step) &&
                jbrown <= (pcutoff - step) && jdarkgreen <= (pcutoff - step) && jviolet <= (pcutoff - step) &&
                jpurple <= (pcutoff - step) && jred <= (pcutoff - step) && jorange <= (pcutoff - step) &&
                jgrassgreen <= (pcutoff - step) && continue

            # check AB
            r, _ = intertwiner_range(jpink, jblue, jbrightgreen, jb)
            isempty(r) && continue

            # check AE
            r, _ = intertwiner_range(jbrown, jdarkgreen, jpink, jb)
            isempty(r) && continue

            # check bottom
            r, _ = intertwiner_range(jviolet, jpurple, jbrown, jb)
            isempty(r) && continue

            # check CD
            r, _ = intertwiner_range(jred, jorange, jviolet, jb)
            isempty(r) && continue

            # check BC
            r, _ = intertwiner_range(jbrightgreen, jgrassgreen, jred, jb)
            isempty(r) && continue

            # inner check up
            r, _ = intertwiner_range(jorange, jdarkgreen, jb, jbrightgreen)
            isempty(r) && continue

            # inner check up-left
            r, _ = intertwiner_range(jgrassgreen, jpurple, jb, jpink)
            isempty(r) && continue

            # inner check bottom-left
            r, _ = intertwiner_range(jblue, jorange, jb, jbrown)
            isempty(r) && continue

            # inner check bottom-right
            r, _ = intertwiner_range(jdarkgreen, jgrassgreen, jb, jviolet)
            isempty(r) && continue

            # inner check up-right
            r, _ = intertwiner_range(jpurple, jblue, jb, jred)
            isempty(r) && continue

            # must be taken into account
            counter_partial_configurations += 1

        end

        total_conf = counter_partial_configurations

        if (!isempty(total_number_spins_configs))
            total_conf += total_number_spins_configs[end]
        end

        log("configurations at partial cutoff = $pcutoff: $(total_conf)\n")
        push!(total_number_spins_configs, total_conf)

    end

    # store total spins configurations at total cutoff
    total_number_spins_configs_df = DataFrame(total_spins_configs=total_number_spins_configs)
    CSV.write("$(configs_path)/spins_configurations_cutoff_$(twice(cutoff)/2).csv", total_number_spins_configs_df)

end


# store Monte Carlo vertex renormalization spins configurations for all partial cutoffs up to cutoff
function vertex_renormalization_MC_sampling(cutoff, Nmc::Int, jb::HalfInt, MC_configs_path::String, step=half(1))

    MC_draws = Array{HalfInt8}(undef, 10, Nmc)
    MC_right_intertwiners_draws = Array{Tuple{Tuple{HalfInt8,HalfInt8},Int8}}(undef, 5, Nmc)
    MC_left_intertwiners_draws = Array{Tuple{Tuple{HalfInt8,HalfInt8},Int8}}(undef, 5, Nmc)
    MC_inner_intertwiners_draws = Array{Tuple{Tuple{HalfInt8,HalfInt8},Int8}}(undef, 5, Nmc)
    draw_float_sample = Array{Float64}(undef, 1)

    # loop over partial cutoffs
    for pcutoff = step:step:cutoff

        distr = Uniform(0, Int(2 * pcutoff + 1))

        for n = 1:Nmc

            while true

                # sampling jpink, jblue, jbrightgreen
                for i = 1:3
                    rand!(distr, draw_float_sample)
                    MC_draws[i, n] = half(floor(draw_float_sample[1]))
                end

                # sampling jbrown, jdarkgreen
                for i = 4:5
                    rand!(distr, draw_float_sample)
                    MC_draws[i, n] = half(floor(draw_float_sample[1]))
                end

                # sampling jviolet, jpurple
                for i = 6:7
                    rand!(distr, draw_float_sample)
                    MC_draws[i, n] = half(floor(draw_float_sample[1]))
                end

                # sampling jred, jorange
                for i = 8:9
                    rand!(distr, draw_float_sample)
                    MC_draws[i, n] = half(floor(draw_float_sample[1]))
                end

                # sampling jgrassgreen
                for i = 10:10
                    rand!(distr, draw_float_sample)
                    MC_draws[i, n] = half(floor(draw_float_sample[1]))
                end

                # skip if computed in lower partial cutoff
                MC_draws[1, n] <= (pcutoff - step) && MC_draws[2, n] <= (pcutoff - step) &&
                    MC_draws[3, n] <= (pcutoff - step) && MC_draws[4, n] <= (pcutoff - step) &&
                    MC_draws[5, n] <= (pcutoff - step) && MC_draws[6, n] <= (pcutoff - step) &&
                    MC_draws[7, n] <= (pcutoff - step) && MC_draws[8, n] <= (pcutoff - step) &&
                    MC_draws[9, n] <= (pcutoff - step) && MC_draws[10, n] <= (pcutoff - step) && continue

                # we check triangular inequalities computing the range of right intertwiners,
                # as we want to avoid zeros and left and right intertwiners have same dimension

                # AB right check
                rABr = intertwiner_range(
                    MC_draws[1, n],
                    MC_draws[2, n],
                    MC_draws[3, n],
                    jb,
                )
                (rABr[2] == 0) && continue

                # AE right check
                rAEr = intertwiner_range(
                    MC_draws[4, n],
                    MC_draws[5, n],
                    MC_draws[1, n],
                    jb
                )
                (rAEr[2] == 0) && continue

                # bottom right check
                rbr = intertwiner_range(
                    MC_draws[6, n],
                    MC_draws[7, n],
                    MC_draws[4, n],
                    jb
                )
                (rbr[2] == 0) && continue

                # CD right check
                rCDr = intertwiner_range(
                    MC_draws[8, n],
                    MC_draws[9, n],
                    MC_draws[6, n],
                    jb
                )
                (rCDr[2] == 0) && continue

                # BC right check
                rBCr = intertwiner_range(
                    MC_draws[3, n],
                    MC_draws[10, n],
                    MC_draws[8, n],
                    jb
                )
                (rBCr[2] == 0) && continue

                # inner check up
                rIu = intertwiner_range(
                    MC_draws[9, n],
                    MC_draws[5, n],
                    jb,
                    MC_draws[3, n]
                )
                (rIu[2] == 0) && continue

                # inner check up-left
                rIul = intertwiner_range(
                    MC_draws[10, n],
                    MC_draws[7, n],
                    jb,
                    MC_draws[1, n]
                )
                (rIul[2] == 0) && continue

                # inner check bottom-left
                rIbl = intertwiner_range(
                    MC_draws[2, n],
                    MC_draws[9, n],
                    jb,
                    MC_draws[4, n]
                )
                (rIbl[2] == 0) && continue

                # inner check bottom-right
                rIbr = intertwiner_range(
                    MC_draws[5, n],
                    MC_draws[10, n],
                    jb,
                    MC_draws[6, n]
                )
                (rIbr[2] == 0) && continue

                # inner check up-right
                rIur = intertwiner_range(
                    MC_draws[7, n],
                    MC_draws[2, n],
                    jb,
                    MC_draws[8, n]
                )
                (rIur[2] == 0) && continue

                # bulk spins have passed all tests -> must be computed

                # now we compute also the range of left intertwiners,
                # as these will be required during the contraction phase

                # AB left
                rABl = intertwiner_range(
                    MC_draws[3, n],
                    MC_draws[2, n],
                    MC_draws[1, n],
                    jb,
                )

                # AE left
                rAEl = intertwiner_range(
                    MC_draws[1, n],
                    MC_draws[5, n],
                    MC_draws[4, n],
                    jb
                )

                # bottom left
                rbl = intertwiner_range(
                    MC_draws[4, n],
                    MC_draws[7, n],
                    MC_draws[6, n],
                    jb
                )

                # CD left
                rCDl = intertwiner_range(
                    MC_draws[6, n],
                    MC_draws[9, n],
                    MC_draws[8, n],
                    jb
                )

                # BC left
                rBCl = intertwiner_range(
                    MC_draws[8, n],
                    MC_draws[10, n],
                    MC_draws[3, n],
                    jb
                )

                MC_right_intertwiners_draws[1, n] = rABr
                MC_right_intertwiners_draws[2, n] = rAEr
                MC_right_intertwiners_draws[3, n] = rbr
                MC_right_intertwiners_draws[4, n] = rCDr
                MC_right_intertwiners_draws[5, n] = rBCr

                MC_left_intertwiners_draws[1, n] = rABl
                MC_left_intertwiners_draws[2, n] = rAEl
                MC_left_intertwiners_draws[3, n] = rbl
                MC_left_intertwiners_draws[4, n] = rCDl
                MC_left_intertwiners_draws[5, n] = rBCl

                MC_inner_intertwiners_draws[1, n] = rIu
                MC_inner_intertwiners_draws[2, n] = rIul
                MC_inner_intertwiners_draws[3, n] = rIbl
                MC_inner_intertwiners_draws[4, n] = rIbr
                MC_inner_intertwiners_draws[5, n] = rIur

                break

            end

        end

        # store MC bulk spins 
        @save "$(MC_configs_path)/MC_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_draws

        # store MC intertwiners
        @save "$(MC_configs_path)/MC_right_intertwiners_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_right_intertwiners_draws
        @save "$(MC_configs_path)/MC_left_intertwiners_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_left_intertwiners_draws
        @save "$(MC_configs_path)/MC_inner_intertwiners_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_inner_intertwiners_draws



    end

end