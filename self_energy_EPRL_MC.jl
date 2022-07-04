using Distributed

number_of_workers = nworkers()
number_of_processes = nprocs()
number_of_threads = Threads.nthreads()
available_cpus = length(Sys.cpu_info())

printstyled("\nSelf energy EPRL divergence parallelized on $(number_of_workers) worker(s) and $(number_of_threads) thread(s)\n\n"; bold=true, color=:blue)

if (number_of_workers * number_of_threads > available_cpus)
    printstyled("WARNING: you are using more resources than available cores on this system. Performances will be affected\n\n"; bold=true, color=:red)
end

length(ARGS) < 6 && error("please use these 6 arguments: data_sl2cfoam_next_folder    cutoff    shell_min    shell_max     Immirzi    store_folder")
@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
SHELL_MIN = parse(Int, ARGS[3])
SHELL_MAX = parse(Int, ARGS[4])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[5]))
@eval STORE_FOLDER = $(ARGS[6])
MONTE_CARLO_ITERATIONS = parse(Int, ARGS[7])

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

STORE_FOLDER = "$(STORE_FOLDER)/data_MC/EPRL/immirzi_$(IMMIRZI)/divergence/cutoff_$(CUTOFF_FLOAT)"
mkpath(STORE_FOLDER)

printstyled("initializing library...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, IMMIRZI)
println("done\n")


function self_energy_EPRL(cutoff, shells, Nmc)

    vec = [8, 73, 286, 758, 1728, 3399, 6242, 10564, 17164, 26453, 39666, 57306, 81164, 111811, 151726, 201512, 264480, 341217, 436022, 549406, 686824, 848639, 1041642, 1265964]

    number_of_threads = Threads.nthreads()

    # set boundary
    step = onehalf = half(1)
    jb = half(1)

    ampls = Float64[]

    draw_float_sample = Array{Float64}(undef, 1)
    spins_draw = Array{HalfInt8}(undef, 6)

    result_return = (ret=true, store=false, store_batches=false)

    cucu = 0

    # loop over partial cutoffs
    for pcutoff = onehalf:step:cutoff

        cucu += 1

        Uniform_distribution = Uniform(0, pcutoff)

        # generate a list of all spins to compute
        spins_all = NTuple{6,HalfInt}[]

        counter = 0

        while (counter < Nmc)

            # sampling j23, j24, j25 for the draw [j23, j24, j25, jb]
            while true

                for i = 1:3
                    rand!(Uniform_distribution, draw_float_sample)
                    draw_float_sample[1] = round(Int64, 2 * draw_float_sample[1])
                    spins_draw[i] = half(draw_float_sample[1])
                end

                r, _ = intertwiner_range(spins_draw[1], spins_draw[2], spins_draw[3], jb)
                isempty(r) || break

            end

            # sampling j34, j35 for the draw [j34, j35, jb, j23]
            while true

                for i = 4:5
                    rand!(Uniform_distribution, draw_float_sample)
                    draw_float_sample[1] = round(Int64, 2 * draw_float_sample[1])
                    spins_draw[i] = half(draw_float_sample[1])
                end

                r, _ = intertwiner_range(spins_draw[4], spins_draw[5], jb, spins_draw[1])
                isempty(r) || break

            end

            # sampling j45 for the draw [j45, jb, j24, j34]
            while true

                for i = 6:6
                    rand!(Uniform_distribution, draw_float_sample)
                    draw_float_sample[1] = round(Int64, 2 * draw_float_sample[1])
                    spins_draw[i] = half(draw_float_sample[1])
                end

                r, _ = intertwiner_range(spins_draw[6], jb, spins_draw[2], spins_draw[4])
                isempty(r) || break

            end

            final_test_1 = false

            # check that draw [jb, j25, j35, j45] satisfies triangular inequalities
            r, _ = intertwiner_range(jb, spins_draw[3], spins_draw[5], spins_draw[6])
            if (!isempty(r))
                final_test_1 = true
            end

            final_test_2 = false

            # check that at least one spin is equal to pcutoff 
            for i = 1:6
                if (spins_draw[i] == pcutoff)
                    final_test_2 = true
                end
            end

            if (final_test_1 == true && final_test_2 == true)

                #if (final_test_1 == true)

                # must be computed
                push!(spins_all, (spins_draw[1], spins_draw[2], spins_draw[3], spins_draw[4], spins_draw[5], spins_draw[6]))
                counter += 1

            end


        end

        #println(spins_all)

        if isempty(spins_all)
            push!(ampls, 0.0)
            continue
        end

        @time tampl = @sync @distributed (+) for spins in spins_all

            j23, j24, j25, j34, j35, j45 = spins

            # restricted range of intertwiners
            r2, _ = intertwiner_range(jb, j25, j24, j23)
            r3, _ = intertwiner_range(j23, jb, j34, j35)
            r4, _ = intertwiner_range(j34, j24, jb, j45)
            r5, _ = intertwiner_range(j45, j35, j25, jb)
            rm = ((0, 0), r2, r3, r4, r5)

            # compute vertex
            v = vertex_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], shells, rm; result=result_return)

            # contract
            dfj = (2j23 + 1) * (2j24 + 1) * (2j25 + 1) * (2j34 + 1) * (2j35 + 1) * (2j45 + 1)

            dfj * dot(v.a, v.a)

        end

        tampl = tampl * vec[cucu] / Nmc

        # if-else for integer spin case
        if isempty(ampls)
            ampl = tampl
            log("Amplitude at partial cutoff = $pcutoff: $(ampl)")
            push!(ampls, ampl)
        else
            ampl = ampls[end] + tampl
            log("Amplitude at partial cutoff = $pcutoff: $(ampl)")
            push!(ampls, ampl)
        end

    end # partial cutoffs loop

    ampls

end

printstyled("Pre-compiling the function...\n"; bold=true, color=:cyan)
@time self_energy_EPRL(1, 0, 10);
println("done\n")
sleep(1)


ampls_matrix = Array{Float64,2}(undef, convert(Int, 2 * CUTOFF), SHELL_MAX - SHELL_MIN + 1)

printstyled("\nStarting computation with K = $(CUTOFF), Dl_min = $(SHELL_MIN), Dl_max = $(SHELL_MAX), Immirzi = $(IMMIRZI)...\n"; bold=true, color=:cyan)

column_labels = String[]

for Dl = SHELL_MIN:SHELL_MAX

    printstyled("\nCurrent Dl = $(Dl)...\n"; bold=true, color=:magenta)
    @time ampls = self_energy_EPRL(CUTOFF, Dl, MONTE_CARLO_ITERATIONS)
    push!(column_labels, "Dl = $(Dl)")
    ampls_matrix[:, Dl-SHELL_MIN+1] = ampls[:]

end

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame(ampls_matrix, column_labels)
CSV.write("$(STORE_FOLDER)/self_energy_Dl_min_$(SHELL_MIN)_Dl_max_$(SHELL_MAX).csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)



