module BayVel

# Run just at the first installation
# Pkg.add("LanguageServer")
# Pkg.add("Revise")
# Pkg.add("Distributions")
# Pkg.add("Random")
# Pkg.add("LinearAlgebra")
# Pkg.add("PDMats")
# Pkg.add("StatsBase")
# Pkg.add("ToggleableAsserts")

using Distributions, Random
using LinearAlgebra, PDMats, StatsBase
using ToggleableAsserts
using ProgressMeter
using LogExpFunctions

##### Include
include(joinpath("struct.jl")) 
include(joinpath("model.jl"))
include(joinpath("updateSS.jl"))
include(joinpath("updateT0.jl"))
include(joinpath("generalFunction.jl"))
include(joinpath("priors.jl"))
include(joinpath("updateTStar.jl"))
include(joinpath("updateEta.jl"))
include(joinpath("updateCatt.jl"))

##### Functions
export    
    MCMC,
    initParam!
end 

# module BayVel
