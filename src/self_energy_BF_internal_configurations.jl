using Distributed
using DelimitedFiles

number_of_workers = nworkers()

printstyled("\nSelf energy BF divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 4 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    STORE_FOLDER")

@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
@eval STORE_FOLDER = $(ARGS[4])

printstyled("precompiling packages and source codes...\n"; bold=true, color=:cyan)
@everywhere begin
    include("../inc/pkgs.jl")
    include("init.jl")
    include("utilities.jl")
    include("spins_configurations.jl")
end

CUTOFF_FLOAT = parse(Float64, ARGS[2])
CUTOFF = HalfInt(CUTOFF_FLOAT)

JB_FLOAT = parse(Float64, ARGS[3])
JB = HalfInt(JB_FLOAT)

# vector with weights on internal faces
# each internal face with spin j has dimension (2j+1)^(weight)
@everywhere FACE_WEIGHTS_VEC = [1.0]

printstyled("\ninitializing library...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, 0.123) # fictitious Immirzi 

SPINS_CONF_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/spins_configurations"
mkpath(SPINS_CONF_FOLDER)

# TODO: modify (this check can be misleading)
if (!isfile("$(SPINS_CONF_FOLDER)/spins_configurations_cutoff_$(CUTOFF_FLOAT).csv"))
    printstyled("computing spins configurations for jb=$(JB) up to K=$(CUTOFF)...\n\n"; bold=true, color=:cyan)
    @time self_energy_spins_conf(CUTOFF, JB, SPINS_CONF_FOLDER)
end

STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/exact/BF"
INTERNAL_CONTRIBUTIONS_FOLDER = "$(STORE_AMPLS_FOLDER)/internal_contributions"
mkpath(INTERNAL_CONTRIBUTIONS_FOLDER)

function self_energy_BF(cutoff, jb::HalfInt, spins_conf_folder::String, face_weights_vec, internal_contributions_folder::String, step=half(1))

    total_number_of_ampls = Int(2 * cutoff + 1)
    boundary_dim = Int(2 * jb + 1)
    number_of_weights = size(face_weights_vec)[1]

    # tensor with amplitudes
    ampls_tensor = zeros(total_number_of_ampls, number_of_weights, boundary_dim)

    for pcutoff = 0:step:cutoff

        index_pcutoff = Int(2 * pcutoff + 1)

        @load "$(spins_conf_folder)/configs_pcutoff_$(twice(pcutoff)/2).jld2" spins_configurations

        number_of_spins_configs = size(spins_configurations)[1]

        all_contributions_internal = zeros(number_of_spins_configs, 7)

        # case pcutoff = 0 AND amplitude = 0
        # TODO: generalize to take into account integer boundary spin case
        if isempty(spins_configurations)
            ampls_tensor[index_pcutoff, :, :] .= 0.0
            continue
        end

        bulk_ampls = SharedArray{Float64}(number_of_spins_configs, number_of_weights, boundary_dim)

        @time @sync @distributed for spins_index in eachindex(spins_configurations)

            j23, j24, j25, j34, j35, j45 = spins_configurations[spins_index]

            # compute vertex
            v = vertex_BF_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45])

            # face base dims
            dfj = (2j23 + 1) * (2j24 + 1) * (2j25 + 1) * (2j34 + 1) * (2j35 + 1) * (2j45 + 1)

            # contract
            for weight_index = 1:number_of_weights
                weight = face_weights_vec[weight_index]
                for ib_index = 1:boundary_dim
                    bulk_ampls[spins_index, weight_index, ib_index] = dfj^(weight) * dot(v.a[:, :, :, :, ib_index], v.a[:, :, :, :, ib_index])
                end
            end

            all_contributions_internal[spins_index, 1:6] .= spins_configurations[spins_index]
            all_contributions_internal[spins_index, 7] = bulk_ampls[spins_index, 1, 1] # modify last two indices depending on what you want

        end

        for weight_index = 1:number_of_weights
            for ib_index = 1:boundary_dim
                tampl = sum(bulk_ampls[:, weight_index, ib_index])
                ampls_tensor[index_pcutoff, weight_index, ib_index] = ampls_tensor[index_pcutoff-1, weight_index, ib_index] + tampl
            end
        end

        log("\nAt partial cutoff = $pcutoff the ampls matrix is:\n")
        display(ampls_tensor[index_pcutoff, :, :])
        println("\n")

        open("$(internal_contributions_folder)/internal_pcutoff_$(twice(pcutoff)/2).csv", "w") do io
            writedlm(io, all_contributions_internal, ',')
        end

    end # partial cutoffs loop

    ampls_tensor

end

total_number_of_ampls = Int(2 * CUTOFF_FLOAT + 1)
boundary_dim = Int(2 * JB + 1)
number_of_weights = size(FACE_WEIGHTS_VEC)[1]

printstyled("\nstarting computation with jb=$(JB) up to K=$(CUTOFF)...\n\n"; bold=true, color=:cyan)
@time ampls_tensor = self_energy_BF(CUTOFF, JB, SPINS_CONF_FOLDER, FACE_WEIGHTS_VEC, INTERNAL_CONTRIBUTIONS_FOLDER);

printstyled("\nsaving dataframe...\n"; bold=true, color=:cyan)

for weight_index = 1:number_of_weights

    weight = round(FACE_WEIGHTS_VEC[weight_index], digits=3)

    for ib_index = 1:boundary_dim

        STORE_AMPLS_FINAL_FOLDER = "$(STORE_AMPLS_FOLDER)/weight_$(weight)/ib_$(ib_index-1)"
        mkpath(STORE_AMPLS_FINAL_FOLDER)

        ampls = ampls_tensor[:, weight_index, ib_index]
        df = DataFrame([ampls], ["amp"])

        #CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/ampls_cutoff_$(CUTOFF).csv", df)

    end

end

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
