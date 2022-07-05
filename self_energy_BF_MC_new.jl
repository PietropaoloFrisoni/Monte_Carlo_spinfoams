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

STORE_FOLDER = "$(STORE_FOLDER)/data_MC/BF/cutoff_$(CUTOFF_FLOAT)"
mkpath(STORE_FOLDER)

printstyled("initializing library...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, 0.123) # fictitious Immirzi 
println("done\n")

# This function receives 3 spins and set the range bounds for the remaining spins, for which there's at least 1 possible intertwiner

function spin_range!(a::Spin, b::Spin, c::Spin, range)

    if (a == 0)
        if (b == 0)
            if (c == 0)
                range[1] = 0
                range[2] = 0
                return 0
            else # c > 0
                range[1] = c
                range[2] = c
            end # if on c
        else # b > 0
            if (c == 0)
                range[1] = b
                range[2] = b
                return 0
            end # if (c == 0)
            if (c < b)
                range[1] = b - c
                range[2] = b + c
                return 0
            end # if (c < b)
            if (c == b)
                range[1] = 0
                range[2] = 2b
                return 0
            end # if (c == b)
            if (c > b)
                range[1] = c - b
                range[2] = c + b
                return 0
            end # if (c > b)
        end # if on b
    else # a > 0
        if (b == 0)
            if (c == 0)
                range[1] = a
                range[2] = a
                return 0
            end # if (c == 0)
            if (c < a)
                range[1] = a - c
                range[2] = a + c
                return 0
            end # if (c < a)
            if (c == a)
                range[1] = 0
                range[2] = 2a
                return 0
            end # if (c == a)
            if (c > a)
                range[1] = c - a
                range[2] = c + a
                return 0
            end # if (c > a)
        end # if (b == 0)
        if (b < a)
            if (c < (a - b))
                range[1] = a - b - c
                range[2] = a + b + c
                return 0
            end # if (c < (a-b))
            if (c <= (a + b))
                range[1] = 0
                range[2] = a + b + c
                return 0
            end # if (c <= (a+b))
            if (c > (a + b))
                range[1] = -a - b + c
                range[2] = a + b + c
                return 0
            end # if (c > (a+b))
        end # if (b < a)
        if (b == a)
            if (c <= 2a)
                range[1] = 0
                range[2] = 2a + c
                return 0
            end # if (c <= 2a)
            if (c > 2a)
                range[1] = c - 2a
                range[2] = c + 2a
                return 0
            end # if (c > 2a)
        end # if (b == a) 
        if (b > a)
            if (c < (b - a))
                range[1] = -a + b - c
                range[2] = a + b + c
                return 0
            end # if (c < (b-a))
            if (c <= (a + b))
                range[1] = 0
                range[2] = a + b + c
                return 0
            end # if (c <= (a+b))
            if (c > (a + b))
                range[1] = -a - b + c
                range[2] = a + b + c
                return 0
            end # if (c > (a+b))
        end # if (b > a)
    end # if on a

end

function self_energy(cutoff, Nmc)

    number_of_threads = Threads.nthreads()

    range = Vector{HalfInt}(undef, 2)

    # set boundary
    step = onehalf = half(1)
    jb = half(1)

    ampls = Float64[]

    draw_float_sample = Array{Float64}(undef, 1)
    spins_draw = Array{HalfInt}(undef, 7)

    # loop over partial cutoffs
    for pcutoff = onehalf:step:cutoff

        Uniform_distribution = Uniform(0, pcutoff)

        # generate a list of all spins to compute
        spins_all = NTuple{7,HalfInt}[]

        counter = 0

        a = 0.0
        b = 0.0
        c = 0.0

        while (counter < Nmc)

            # sampling j23, j24, j25 for the draw [j23, j24, j25, jb]
            while true

                Uniform_distribution = Uniform(0, pcutoff)

                for i = 1:2
                    rand!(Uniform_distribution, draw_float_sample)
                    draw_float_sample[1] = round(Int64, 2 * draw_float_sample[1])
                    spins_draw[i] = half(draw_float_sample[1])
                end

                spin_range!(spins_draw[1], spins_draw[2], jb, range)

                # println(range)

                if (range[1] == range[2])
                    spins_draw[3] = range[1]

                    a = 1

                    r, _ = intertwiner_range(spins_draw[1], spins_draw[2], spins_draw[3], jb)
                    isempty(r) || break

                else

                    Uniform_distribution = Uniform(range[1], range[2])

                    for i = 3:3
                        rand!(Uniform_distribution, draw_float_sample)
                        draw_float_sample[1] = round(Int64, 2 * draw_float_sample[1])
                        spins_draw[i] = half(draw_float_sample[1])
                    end

                    a = 2 * (range[2] - range[1])

                    r, _ = intertwiner_range(spins_draw[1], spins_draw[2], spins_draw[3], jb)
                    isempty(r) || break

                end

                #break

            end

            # sampling j34, j35 for the draw [j34, j35, jb, j23]
            while true

                Uniform_distribution = Uniform(0, pcutoff)

                for i = 4:4
                    rand!(Uniform_distribution, draw_float_sample)
                    draw_float_sample[1] = round(Int64, 2 * draw_float_sample[1])
                    spins_draw[i] = half(draw_float_sample[1])
                end

                spin_range!(spins_draw[1], jb, spins_draw[4], range)

                if (range[1] == range[2])
                    spins_draw[5] = range[1]

                    b = 1

                    r, _ = intertwiner_range(spins_draw[4], spins_draw[5], jb, spins_draw[1])
                    isempty(r) || break

                else

                    Uniform_distribution = Uniform(range[1], range[2])

                    for i = 5:5
                        rand!(Uniform_distribution, draw_float_sample)
                        draw_float_sample[1] = round(Int64, 2 * draw_float_sample[1])
                        spins_draw[i] = half(draw_float_sample[1])
                    end

                    b = 2 * (range[2] - range[1])

                    r, _ = intertwiner_range(spins_draw[4], spins_draw[5], jb, spins_draw[1])
                    isempty(r) || break

                end

                #break

            end

            # sampling j45 for the draw [j45, jb, j24, j34]
            while true

                spin_range!(spins_draw[4], spins_draw[2], jb, range)

                if (range[1] == range[2])
                    spins_draw[6] = range[1]

                    c = 1

                    r, _ = intertwiner_range(spins_draw[6], jb, spins_draw[2], spins_draw[4])
                    isempty(r) || break

                else

                    Uniform_distribution = Uniform(range[1], range[2])

                    for i = 6:6
                        rand!(Uniform_distribution, draw_float_sample)
                        draw_float_sample[1] = round(Int64, 2 * draw_float_sample[1])
                        spins_draw[i] = half(draw_float_sample[1])
                    end

                    c = 2 * (range[2] - range[1])

                    r, _ = intertwiner_range(spins_draw[6], jb, spins_draw[2], spins_draw[4])
                    isempty(r) || break

                end

                #break

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

                spins_draw[7] = a * b * c * (2 * (pcutoff))^3

                # must be computed
                push!(spins_all, (spins_draw[1], spins_draw[2], spins_draw[3], spins_draw[4], spins_draw[5], spins_draw[6], spins_draw[7]))
                counter += 1

            end


        end

        if isempty(spins_all)
            push!(ampls, 0.0)
            continue
        end

        #println(spins_all)

        @time tampl = @sync @distributed (+) for spins in spins_all

            j23, j24, j25, j34, j35, j45, multeplicity = spins

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

            dfj * dot(v.a, v.a) * multeplicity

        end

        tampl = tampl / Nmc

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
@time self_energy(1, 10);
println("done\n")
sleep(1)

printstyled("\nStarting computation with K = $(CUTOFF)...\n"; bold=true, color=:cyan)
@time ampls = self_energy(CUTOFF, MONTE_CARLO_ITERATIONS);

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame(amplitudes=ampls)
CSV.write("$(STORE_FOLDER)/self_energy_10.csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)


