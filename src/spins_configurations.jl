# compute self-energy spins configurations for all partial cutoffs up to cutoff
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

        if (pcutoff > 0)
            nconf = total_number_spins_configs[end] + size(spins_configurations)[1]
            push!(total_number_spins_configs, nconf)
        else
            push!(total_number_spins_configs, size(spins_configurations)[1])
        end

    end

    # store total spins configurations at total cutoff
    total_number_spins_configs_df = DataFrame(total_spins_configs=total_number_spins_configs)
    CSV.write("$(configs_path)/spins_configurations_cutoff_$(twice(cutoff)/2).csv", total_number_spins_configs_df)

end


# compute Monte Carlo self-energy indices for all partial cutoffs up to cutoff
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


function self_energy_MC_sampling_BIASED(cutoff, Nmc::Int, jb::HalfInt, MC_configs_path::String, step=half(1))

    MC_draws = Array{HalfInt8}(undef, 6, Nmc)
    draw_float_sample = Array{Float64}(undef, 1)

    # loop over partial cutoffs
    for pcutoff = step:step:cutoff

        distr = Uniform(0, Int(2 * pcutoff + 1))

        for n = 1:Nmc

            while true

                # sampling j23, j24, j25 for the 4j with spins [j23, j24, j25, jb]
                while true
                    for i = 1:3
                        rand!(distr, draw_float_sample)
                        MC_draws[i, n] = half(floor(draw_float_sample[1]))
                    end
                    r, _ = intertwiner_range(
                        MC_draws[1, n],
                        MC_draws[2, n],
                        MC_draws[3, n],
                        jb
                    )
                    isempty(r) || break
                end

                # sampling j34, j35 for the 4j with spins [j34, j35, jb, j23]
                while true
                    for i = 4:5
                        rand!(distr, draw_float_sample)
                        MC_draws[i, n] = half(floor(draw_float_sample[1]))
                    end
                    r, _ = intertwiner_range(
                        MC_draws[4, n],
                        MC_draws[5, n],
                        jb,
                        MC_draws[1, n]
                    )
                    isempty(r) || break
                end

                # sampling j45 for the 4j with spins [j45, jb, j24, j34]
                while true
                    for i = 6:6
                        rand!(distr, draw_float_sample)
                        MC_draws[i, n] = half(floor(draw_float_sample[1]))
                    end
                    r, _ = intertwiner_range(
                        MC_draws[6, n],
                        jb,
                        MC_draws[2, n],
                        MC_draws[4, n]
                    )
                    isempty(r) || break
                end

                # skip if computed in lower partial cutoff
                MC_draws[1, n] <= (pcutoff - step) && MC_draws[2, n] <= (pcutoff - step) &&
                    MC_draws[3, n] <= (pcutoff - step) && MC_draws[4, n] <= (pcutoff - step) &&
                    MC_draws[5, n] <= (pcutoff - step) && MC_draws[6, n] <= (pcutoff - step) && continue

                # check that 4j with spins [jb, j25, j35, j45] satisfies triangular inequalities
                r, _ = intertwiner_range(
                    jb,
                    MC_draws[3, n],
                    MC_draws[5, n],
                    MC_draws[6, n],
                )
                isempty(r) || break

            end

        end

        # store MC spins indices 
        @save "$(MC_configs_path)/MC_draws_pcutoff_$(twice(pcutoff)/2).jld2" MC_draws

    end

end



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