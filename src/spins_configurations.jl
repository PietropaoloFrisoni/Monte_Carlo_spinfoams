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

                test = true

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
                if (MC_draws[1, n] <= (pcutoff - step) && MC_draws[2, n] <= (pcutoff - step) &&
                    MC_draws[3, n] <= (pcutoff - step) && MC_draws[4, n] <= (pcutoff - step) &&
                    MC_draws[5, n] <= (pcutoff - step) && MC_draws[6, n] <= (pcutoff - step))
                    test = false
                end

                # check that 4j with spins [j45, jb, j24, j34] satisfies triangular inequalities
                r, _ = intertwiner_range(
                    MC_draws[6, n],
                    jb,
                    MC_draws[2, n],
                    MC_draws[4, n]
                )
                if (isempty(r))
                    test = false
                end

                # check that 4j with spins [j34, j35, jb, j23] satisfies triangular inequalities
                r, _ = intertwiner_range(
                    MC_draws[4, n],
                    MC_draws[5, n],
                    jb,
                    MC_draws[1, n]
                )
                if (isempty(r))
                    test = false
                end

                # check that 4j with spins [j23, j24, j25, jb] satisfies triangular inequalities
                r, _ = intertwiner_range(
                    MC_draws[1, n],
                    MC_draws[2, n],
                    MC_draws[3, n],
                    jb
                )
                if (isempty(r))
                    test = false
                end

                # check that 4j with spins [jb, j25, j35, j45] satisfies triangular inequalities
                r, _ = intertwiner_range(
                    jb,
                    MC_draws[3, n],
                    MC_draws[5, n],
                    MC_draws[6, n],
                )
                if (isempty(r))
                    test = false
                end

                if (test == true)
                    break
                end

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
    draw_float_sample = Array{Float64}(undef, 1)

    # loop over partial cutoffs
    for pcutoff = step:step:cutoff

        distr = Uniform(0, Int(2 * pcutoff + 1))

        for n = 1:Nmc

            while true

                test = true

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
                if (MC_draws[1, n] <= (pcutoff - step) && MC_draws[2, n] <= (pcutoff - step) &&
                    MC_draws[3, n] <= (pcutoff - step) && MC_draws[4, n] <= (pcutoff - step) &&
                    MC_draws[5, n] <= (pcutoff - step) && MC_draws[6, n] <= (pcutoff - step) &&
                    MC_draws[7, n] <= (pcutoff - step) && MC_draws[8, n] <= (pcutoff - step) &&
                    MC_draws[9, n] <= (pcutoff - step) && MC_draws[10, n] <= (pcutoff - step))
                    test = false
                end

                # AB check
                r, _ = intertwiner_range(
                    MC_draws[1, n],
                    MC_draws[2, n],
                    MC_draws[3, n],
                    jb,
                )
                if (isempty(r))
                    test = false
                end

                # AE check
                r, _ = intertwiner_range(
                    MC_draws[4, n],
                    MC_draws[5, n],
                    MC_draws[1, n],
                    jb
                )
                if (isempty(r))
                    test = false
                end

                # bottom check
                r, _ = intertwiner_range(
                    MC_draws[6, n],
                    MC_draws[7, n],
                    MC_draws[4, n],
                    jb
                )
                if (isempty(r))
                    test = false
                end

                # CD check
                r, _ = intertwiner_range(
                    MC_draws[8, n],
                    MC_draws[9, n],
                    MC_draws[6, n],
                    jb
                )
                if (isempty(r))
                    test = false
                end

                # BC check
                r, _ = intertwiner_range(
                    MC_draws[3, n],
                    MC_draws[10, n],
                    MC_draws[8, n],
                    jb
                )
                if (isempty(r))
                    test = false
                end

                # inner check up
                r, _ = intertwiner_range(
                    MC_draws[9, n],
                    MC_draws[5, n],
                    jb,
                    MC_draws[3, n]
                )
                if (isempty(r))
                    test = false
                end

                # inner check up-left
                r, _ = intertwiner_range(
                    MC_draws[10, n],
                    MC_draws[7, n],
                    jb,
                    MC_draws[1, n]
                )
                if (isempty(r))
                    test = false
                end

                # inner check bottom-left
                r, _ = intertwiner_range(
                    MC_draws[2, n],
                    MC_draws[9, n],
                    jb,
                    MC_draws[4, n]
                )
                if (isempty(r))
                    test = false
                end

                # inner check bottom-right
                r, _ = intertwiner_range(
                    MC_draws[5, n],
                    MC_draws[10, n],
                    jb,
                    MC_draws[6, n]
                )
                if (isempty(r))
                    test = false
                end

                # inner check up-right
                r, _ = intertwiner_range(
                    MC_draws[7, n],
                    MC_draws[2, n],
                    jb,
                    MC_draws[8, n]
                )
                if (isempty(r))
                    test = false
                end

                if (test == true)
                    break
                end

            end

        end

        # store MC spins indices 
        @save "$(MC_configs_path)/MC_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_draws

    end

end