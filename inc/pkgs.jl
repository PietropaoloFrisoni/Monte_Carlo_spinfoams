using SL2Cfoam
using JLD2
using Distributed
using DataFrames
using Dates
using CSV
using HalfIntegers
using LoopVectorization
using LinearAlgebra
using Random
using Distributions
using SharedArrays
using WignerSymbols

# useful command to install all required pkgs in a single loop

#=
vec = ["Distributions", "Random", "HalfIntegers", "WignerSymbols", "LoopVectorization", "SharedArrays", "JLD2", "LinearAlgebra", "Distributed", "ElasticArrays", "CSV", "DataFrames", "Dates"]
import Pkg
for p in vec
Pkg.add("$p")
end
=#
