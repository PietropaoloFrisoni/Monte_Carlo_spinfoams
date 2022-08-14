using Distributed

number_of_workers = nworkers()

printstyled("\nSelf-energy BF divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 4 && error("please use these 3 arguments: data_sl2cfoam_next_folder    cutoff    store_folder    M_C_iterations")
@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
CUTOFF = parse(Int, ARGS[2])
@eval STORE_FOLDER = $(ARGS[3])
MONTE_CARLO_ITERATIONS = parse(Int, ARGS[4])

printstyled("precompiling packages...\n"; bold=true, color=:cyan)
@everywhere begin
    include("../inc/pkgs.jl")
    include("init.jl")
end
println("done\n")

CUTOFF_FLOAT = parse(Float64, ARGS[2])
CUTOFF = HalfInt(CUTOFF_FLOAT)

if (CUTOFF <= 1)
    error("please provide a larger cutoff")
end

STORE_FOLDER = "$(STORE_FOLDER)/data_MC/BF/cutoff_$(CUTOFF_FLOAT)/MC_iterations_$(MONTE_CARLO_ITERATIONS)"
mkpath(STORE_FOLDER)

printstyled("initializing library...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, 0.123) # fictitious Immirzi 
println("done\n")


function self_energy(cutoff, Nmc, number_of_workers)

    # set boundary
    step = onehalf = half(1)
    jb = half(1)

    ampls = Float64[]
    stds = Float64[]
    nconfs = Int64[]

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate a list of all spins
        spins_all = NTuple{6,HalfInt}[]

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
            push!(spins_all, (j23, j24, j25, j34, j35, j45))

        end

        if isempty(spins_all)
            push!(ampls, 0.0)
            push!(stds, 0.0)
            push!(nconfs, 0)
            continue
        end

        tnconf = size(spins_all)[1]

        distr = Uniform(1, tnconf + 1)

        draw_float_sample = Array{Float64}(undef, 1)

        bulk_ampls = SharedArray{Float64}(Nmc)
        bulk_ampls[:] .= 0

        @time @sync @distributed for bulk_ampls_index in eachindex(bulk_ampls)

            rand!(distr, draw_float_sample)

            index_MC = round(Int64, floor(draw_float_sample[1]))

            j23, j24, j25, j34, j35, j45 = spins_all[index_MC]

            # restricted range of intertwiners
            r2, _ = intertwiner_range(jb, j25, j24, j23)
            r3, _ = intertwiner_range(j23, jb, j34, j35)
            r4, _ = intertwiner_range(j34, j24, jb, j45)
            r5, _ = intertwiner_range(j45, j35, j25, jb)
            rm = ((0, 0), r2, r3, r4, r5)

            # compute vertex
            v = vertex_BF_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], rm;)

            # contract
            dfj = (2j23 + 1) * (2j24 + 1) * (2j25 + 1) * (2j34 + 1) * (2j35 + 1) * (2j45 + 1)

            bulk_ampls[bulk_ampls_index] = dfj * dot(v.a, v.a)

        end

        tampl = mean(bulk_ampls)

        tampl_var = 0.0

        for i = 1:Nmc
            tampl_var += (bulk_ampls[i] - tampl)^2
        end

        tampl_var /= (Nmc - 1)

        # normalize
        tampl = tampl * tnconf
        tampl_std = sqrt(tampl_var * (tnconf^2) / Nmc)

        ampl = ampls[end] + tampl
        std = stds[end] + tampl_std
        nconf = nconfs[end] + tnconf 

        log("Amplitude at partial cutoff = $pcutoff: $(ampl)")
        push!(ampls, ampl)
        log("Amplitude std at partial cutoff = $pcutoff: $(std)")
        push!(stds, std)
        log("Bulk spins configs at partial cutoff = $pcutoff: $(nconf)")
        push!(nconfs, nconf)

    end # partial cutoffs loop

    nconfs, ampls, stds

end

printstyled("Pre-compiling the function...\n"; bold=true, color=:cyan)
@time self_energy(1, 10, 1);
println("done\n")
sleep(1)

printstyled("\nStarting computation with K = $(CUTOFF)...\n"; bold=true, color=:cyan)
@time nconfs, ampls, stds = self_energy(CUTOFF, MONTE_CARLO_ITERATIONS, number_of_workers);

column_labels = ["n_confs", "amp", "std"]

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame([nconfs, ampls, stds], column_labels)

CSV.write("$(STORE_FOLDER)/self_energy.csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
