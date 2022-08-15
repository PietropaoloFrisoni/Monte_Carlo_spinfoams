using Distributed

number_of_workers = nworkers()

printstyled("\nSelf-energy BF divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 5 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    STORE_FOLDER    COMPUTE_SPINS_CONFIGURATIONS")
@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
@eval STORE_FOLDER = $(ARGS[4])
COMPUTE_SPINS_CONFIGURATIONS = parse(Bool, ARGS[5])

printstyled("precompiling packages and source codes...\n"; bold=true, color=:cyan)
@everywhere begin
    include("../inc/pkgs.jl")
    include("init.jl")
    include("spins_configurations.jl")
end
println("done\n")

CUTOFF_FLOAT = parse(Float64, ARGS[2])
CUTOFF = HalfInt(CUTOFF_FLOAT)

JB_FLOAT = parse(Float64, ARGS[3])
JB = HalfInt(JB_FLOAT)

printstyled("initializing library...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, 0.123) # fictitious Immirzi 
println("done\n")

SPINS_CONF_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/spins_configurations"

if (COMPUTE_SPINS_CONFIGURATIONS)
    printstyled("computing spins configurations for jb $(JB) up to cutoff $(CUTOFF)...\n"; bold=true, color=:cyan)
    mkpath(SPINS_CONF_FOLDER)
    @time self_energy_spins_conf(CUTOFF, JB, SPINS_CONF_FOLDER)
    println("done\n")
end

STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/exact/BF"
mkpath(STORE_AMPLS_FOLDER)

function self_energy_BF(cutoff, jb::HalfInt, spins_conf_folder::String, step=half(1))

    ampls = Float64[]

    for pcutoff = 0:step:cutoff

        @load "$(spins_conf_folder)/configs_pcutoff_$(twice(pcutoff)/2).jld2" spins_configurations

        if isempty(spins_configurations)
            push!(ampls, 0.0)
            continue
        end

        @time tampl = @sync @distributed (+) for spins in spins_configurations

            j23, j24, j25, j34, j35, j45 = spins

            # restricted range of intertwiners
            r2, _ = intertwiner_range(jb, j25, j24, j23)
            r3, _ = intertwiner_range(j23, jb, j34, j35)
            r4, _ = intertwiner_range(j34, j24, jb, j45)
            r5, _ = intertwiner_range(j45, j35, j25, jb)
            rm = ((0, 0), r2, r3, r4, r5)

            # compute vertex
            v = vertex_BF_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], rm;)

            # face dims
            dfj = (2j23 + 1) * (2j24 + 1) * (2j25 + 1) * (2j34 + 1) * (2j35 + 1) * (2j45 + 1)

            # contract
            dfj * dot(v.a, v.a)

        end

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

printstyled("\nStarting computation with jb $(JB) up to cutoff $(CUTOFF)...\n"; bold=true, color=:cyan)
@time ampls = self_energy_BF(CUTOFF, JB, SPINS_CONF_FOLDER);

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame([ampls], ["amp"])
CSV.write("$(STORE_AMPLS_FOLDER)/ampls_cutoff_$(CUTOFF)_ib_0.0.csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
