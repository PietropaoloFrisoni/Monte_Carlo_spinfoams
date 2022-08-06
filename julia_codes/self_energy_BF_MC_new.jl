using Distributed

printstyled("\nSelf-energy BF divergence parallelized on $(nworkers()) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 4 && error("please use these 3 arguments: data_sl2cfoam_next_folder    cutoff    store_folder    M_C_iterations")
@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
CUTOFF = parse(Int, ARGS[2])
@eval STORE_FOLDER = $(ARGS[3])
MONTE_CARLO_ITERATIONS = parse(Int, ARGS[4])

printstyled("precompiling packages...\n"; bold=true, color=:cyan)
@everywhere begin
    include("pkgs.jl")
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


function self_energy(cutoff, Nmc)

    number_of_threads = Threads.nthreads()

    # set boundary
    step = onehalf = half(1)
    jb = half(1)

    ampls = Float64[]

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
            continue
        end

        number_bulk_spins = size(spins_all)[1]

        indices_spins_MC = rand(1:number_bulk_spins, Nmc)

        @time tampl = @sync @distributed (+) for index_MC in indices_spins_MC

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

            dfj * dot(v.a, v.a)

        end

        # normalize the amplitude to get MC estimator
        tampl = tampl * number_bulk_spins / Nmc

        ampl = ampls[end] + tampl

        log("Amplitude at partial cutoff = $pcutoff: $(ampl)")
        push!(ampls, ampl)

    end # partial cutoffs loop

    ampls

end

printstyled("Pre-compiling the function...\n"; bold=true, color=:cyan)
@time self_energy(1, 10);
println("done\n")
sleep(1)

printstyled("\nStarting computation with K = $(CUTOFF)...\n"; bold=true, color=:cyan)
@time ampls = self_energy(CUTOFF, MONTE_CARLO_ITERATIONS);

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame(amplitudes=ampls)
CSV.write("$(STORE_FOLDER)/self_energy.csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)