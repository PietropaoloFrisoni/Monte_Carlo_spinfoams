# compute the self-energy spins configurations for all partial cutoffs up to cutoff
function self_energy_spins_conf(cutoff, jb::HalfInt, configs_path::String)

    step = onehalf = half(1)

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate a list of all spins to compute
        spins_configurations = NTuple{6,HalfInt8}[]

        for j23::HalfInt = 0:onehalf:pcutoff, j24::HalfInt = 0:onehalf:pcutoff, j25::HalfInt = 0:onehalf:pcutoff,
            j34::HalfInt = 0:onehalf:pcutoff, j35::HalfInt = 0:onehalf:pcutoff, j45::HalfInt = 0:onehalf:pcutoff

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

        # store spins configurations 
        @save "$(configs_path)/spins_configurations_pcutoff_$(twice(pcutoff)/2).jld2" spins_configurations

    end

end