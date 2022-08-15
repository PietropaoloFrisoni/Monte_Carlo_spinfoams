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