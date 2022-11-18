using Distributed

number_of_workers = nworkers()

printstyled("\nSelf energy EPRL divergence parallelized on $(number_of_workers) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 6 && error("use these arguments: DATA_SL2CFOAM_FOLDER    CUTOFF    JB    DL_MAX     IMMIRZI    STORE_FOLDER")

@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
DL_MAX = parse(Int, ARGS[4])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[5]))
@eval STORE_FOLDER = $(ARGS[6])

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

printstyled("\ninitializing library with immirzi $(IMMIRZI)...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, IMMIRZI)

SPINS_CONF_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/spins_configurations"

# TODO: modify (this check can be misleading)
if (!isfile("$(SPINS_CONF_FOLDER)/spins_configurations_cutoff_$(CUTOFF_FLOAT).csv"))
    printstyled("\ncomputing spins configurations for jb=$(JB) up to K=$(CUTOFF)...\n\n"; bold=true, color=:cyan)
    @time self_energy_spins_conf(CUTOFF, JB, SPINS_CONF_FOLDER)
    println("done\n")
end

STORE_AMPLS_FOLDER = "$(STORE_FOLDER)/data/self_energy/jb_$(JB_FLOAT)/exact/EPRL/immirzi_$(IMMIRZI)/NEW_EXTRAPOLATION"
mkpath(STORE_AMPLS_FOLDER)

function self_energy_EPRL(cutoff, jb::HalfInt, Dl::Int, spins_conf_folder::String, step=half(1))

    total_number_of_ampls = Int(2 * cutoff + 1)

    # amplitudes
    ampls_Dl = zeros(total_number_of_ampls)
    ampls_Dlm1 = zeros(total_number_of_ampls)
    ampls_Dlm2 = zeros(total_number_of_ampls)

    ampls_W1 = zeros(total_number_of_ampls)
    ampls_W2 = zeros(total_number_of_ampls)
    ampls_W3 = zeros(total_number_of_ampls)
    ampls_W4 = zeros(total_number_of_ampls)

    result_return = (ret=true, store=false, store_batches=false)

    for pcutoff = 0:step:cutoff

        index_pcutoff = Int(2 * pcutoff + 1)

        @load "$(spins_conf_folder)/configs_pcutoff_$(twice(pcutoff)/2).jld2" spins_configurations

        number_of_spins_configs = size(spins_configurations)[1]

        if isempty(spins_configurations)

            ampls_Dl[index_pcutoff] = 0.0
            ampls_Dlm1[index_pcutoff] = 0.0
            ampls_Dlm2[index_pcutoff] = 0.0

            ampls_W1[index_pcutoff] = 0.0
            ampls_W2[index_pcutoff] = 0.0
            ampls_W3[index_pcutoff] = 0.0
            ampls_W4[index_pcutoff] = 0.0

            continue

        end

        bulk_ampls_Dl = SharedArray{Float64}(number_of_spins_configs)
        bulk_ampls_Dlm1 = SharedArray{Float64}(number_of_spins_configs)
        bulk_ampls_Dlm2 = SharedArray{Float64}(number_of_spins_configs)

        W1_extrapolated = 0.0
        W2_extrapolated = 0.0
        W3_extrapolated = 0.0

        negative_partial_amplitude_found = false

        @time @sync @distributed for spins_index in eachindex(spins_configurations)

            j23, j24, j25, j34, j35, j45 = spins_configurations[spins_index]

            # compute vertices
            v_Dl = vertex_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], Dl; result=result_return)
            v_Dlm1 = vertex_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], Dl - 1; result=result_return)
            v_Dlm2 = vertex_compute([jb, jb, jb, jb, j23, j24, j25, j34, j35, j45], Dl - 2; result=result_return)

            # extrapolate amplitude W_1 for each intertwiner
            v_extrapolated = zeros(size(v_Dl.a))
            v_extrapolated .= ((v_Dl.a .* v_Dlm2.a .- v_Dlm1.a .^ 2) ./ (v_Dl.a .- 2 .* v_Dlm1.a .+ v_Dlm2.a))

            # face dims
            dfj = (2j23 + 1) * (2j24 + 1) * (2j25 + 1) * (2j34 + 1) * (2j35 + 1) * (2j45 + 1)

            # contract
            bulk_ampls_Dl[spins_index] = dot(v_Dl.a[:, :, :, :, 1], v_Dl.a[:, :, :, :, 1])
            bulk_ampls_Dlm1[spins_index] = dot(v_Dlm1.a[:, :, :, :, 1], v_Dlm1.a[:, :, :, :, 1])
            bulk_ampls_Dlm2[spins_index] = dot(v_Dlm2.a[:, :, :, :, 1], v_Dlm2.a[:, :, :, :, 1])

            # accumulate W_1
            W1_extrapolated += dfj * dot(v_extrapolated[:, :, :, :, 1], v_extrapolated[:, :, :, :, 1])

            # extrapolate and accumulate W_2
            W2_extrapolated += ((bulk_ampls_Dl[spins_index] * bulk_ampls_Dlm2[spins_index] - bulk_ampls_Dlm1[spins_index]^2) / (bulk_ampls_Dl[spins_index] - 2 * bulk_ampls_Dlm1[spins_index] + bulk_ampls_Dlm2[spins_index]))

            # face dims
            bulk_ampls_Dl[spins_index] *= dfj
            bulk_ampls_Dlm1[spins_index] *= dfj
            bulk_ampls_Dlm2[spins_index] *= dfj

            if (bulk_ampls_Dl[spins_index] < 0 || bulk_ampls_Dlm1[spins_index] < 0 || bulk_ampls_Dlm2[spins_index] < 0)
                negative_partial_amplitude_found = true
            end

        end

        tampl_Dl = sum(bulk_ampls_Dl[:])
        tampl_Dlm1 = sum(bulk_ampls_Dlm1[:])
        tampl_Dlm2 = sum(bulk_ampls_Dlm2[:])

        # extrapolate W_3
        W3_extrapolated = ((tampl_Dl * tampl_Dlm2 - tampl_Dlm1^2) / (tampl_Dl - 2 * tampl_Dlm1 + tampl_Dlm2))

        ampls_Dl[index_pcutoff] = ampls_Dl[index_pcutoff-1] + tampl_Dl
        ampls_Dlm1[index_pcutoff] = ampls_Dlm1[index_pcutoff-1] + tampl_Dlm1
        ampls_Dlm2[index_pcutoff] = ampls_Dlm2[index_pcutoff-1] + tampl_Dlm2

        ampls_W1[index_pcutoff] = ampls_W1[index_pcutoff-1] + W1_extrapolated
        ampls_W2[index_pcutoff] = ampls_W2[index_pcutoff-1] + W2_extrapolated
        ampls_W3[index_pcutoff] = ampls_W3[index_pcutoff-1] + W3_extrapolated
        ampls_W4[index_pcutoff] = ((ampls_Dl[index_pcutoff] * ampls_Dlm2[index_pcutoff] - ampls_Dlm1[index_pcutoff]^2) / (ampls_Dl[index_pcutoff] - 2 * ampls_Dlm1[index_pcutoff] + ampls_Dlm2[index_pcutoff]))

        log("\nAt partial cutoff = $pcutoff the amplitudes are:\n")

        println("ampls_Dl = $(ampls_Dl[index_pcutoff])\n")
        println("ampls_Dlm1 = $(ampls_Dlm1[index_pcutoff])\n")
        println("ampls_Dlm2 = $(ampls_Dlm2[index_pcutoff])\n")

        println("W_1 = $(ampls_W1[index_pcutoff])\n")
        println("W_2 = $(ampls_W2[index_pcutoff])\n")
        println("W_3 = $(ampls_W3[index_pcutoff])\n")
        println("W_4 = $(ampls_W4[index_pcutoff])\n")

        println("negative_partial_amplitude_found = $(negative_partial_amplitude_found)\n")

        println("\n")

    end # partial cutoffs loop

    ampls_Dl, ampls_Dlm1, ampls_Dlm2, ampls_W1, ampls_W2, ampls_W3, ampls_W4

end

total_number_of_ampls = Int(2 * CUTOFF_FLOAT + 1)
boundary_dim = Int(2 * JB + 1)

printstyled("\nstarting computation with jb=$(JB), Dl_max=$(DL_MAX), Immirzi=$(IMMIRZI) up to K=$(CUTOFF)...\n"; bold=true, color=:cyan)

STORE_AMPLS_FOLDER_DL = "$(STORE_AMPLS_FOLDER)/Dl_MAX_$(DL_MAX)"

@time ampls_Dl, ampls_Dlm1, ampls_Dlm2, ampls_W1, ampls_W2, ampls_W3, ampls_W4 = self_energy_EPRL(CUTOFF, JB, DL_MAX, SPINS_CONF_FOLDER)

printstyled("\nsaving dataframes...\n"; bold=true, color=:cyan)

STORE_AMPLS_FINAL_FOLDER = "$(STORE_AMPLS_FOLDER_DL)"
mkpath(STORE_AMPLS_FINAL_FOLDER)

ampls_Dl_df = DataFrame(Dl=ampls_Dl[:])
ampls_Dlm1_df = DataFrame(Dlm1=ampls_Dlm1[:])
ampls_Dlm2_df = DataFrame(Dlm2=ampls_Dlm2[:])

ampls_W1_df = DataFrame(W1=ampls_W1[:])
ampls_W2_df = DataFrame(W2=ampls_W2[:])
ampls_W3_df = DataFrame(W3=ampls_W3[:])
ampls_W4_df = DataFrame(W4=ampls_W4[:])

CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/Dl_cutoff_$(CUTOFF).csv", ampls_Dl_df)
CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/Dlm1_cutoff_$(CUTOFF).csv", ampls_Dlm1_df)
CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/Dlm2_cutoff_$(CUTOFF).csv", ampls_Dlm2_df)

CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/W1_cutoff_$(CUTOFF).csv", ampls_W1_df)
CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/W2_cutoff_$(CUTOFF).csv", ampls_W2_df)
CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/W3_cutoff_$(CUTOFF).csv", ampls_W3_df)
CSV.write("$(STORE_AMPLS_FINAL_FOLDER)/W4_cutoff_$(CUTOFF).csv", ampls_W4_df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
