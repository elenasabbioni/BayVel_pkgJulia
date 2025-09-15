#################################
#      MODEL TYPE 
#################################
abstract type modelType end

# Model with group and subgroup structures, where beta is kept constant and equal to 1
struct groupSubgroup <: modelType 
    function groupSubgroup()
        new()
    end
end

# TO DO: model in which Beta is not fixed equal to 1
struct notFixedBeta
    function notFixedBeta()
        new()
    end
end


#################################
#      ADAPTIVE STRUCTURES for MCMC
#################################
abstract type adaptiveStep_1 end  


struct adaptiveAlg4 <: adaptiveStep_1
    alphaTarget::Float64
    varInit::Matrix{Float64} 
    meanVec::Vector{Float64}
    varMat::Symmetric{Float64, Matrix{Float64}}
    lambdaVec::Vector{Float64}
    stepVect::Vector{Float64} 
    start::Vector{Bool}  
    update::Vector{Bool} 
    dimP::Int64 

    # Structure for handling adaptive MCMC with a multivariate random-walk proposal (Algorithm 4 of Andrieu and Thomas [2008])
    # It stores estimates of the proposal mean, covariance, scaling, and
    # flags for when adaptation should start/stop.
    #  Arguments
    # - alphaTarget: target acceptance rate (typically around 0.234 for multivariate proposals).
    # - varInit: initial proposal variance used in the burn-in.
    # - meanVec: running mean vector of sampled parameters.
    # - varMat: current variance and covariance matrix of the proposal distribution.
    # - lambdaVec: scaling factor for the random walk.
    # - stepVect: adaptation step sizes.
    # - start: flags indicating whether to start using updated parameters (instead of initial ones).
    # - update: flags indicating whether to keep updating after a given iteration.
    # - dimP: dimension of the parameter vector (helps determine if some components remain fixed, e.g. beta).

    # Constructor
    #     adaptiveAlg4(alpha, var, param_MCMC, stepVect, st, upd, dim; gene)

    function adaptiveAlg4(alpha::Float64, var::Vector{Float64}, param_MCMC::Matrix{Float64}, stepVect::Vector{Float64}, st::Vector{Bool}, upd::Vector{Bool}, dim::Int64; gene::Int64 = 1)
        dimP = dim
        varInit = Symmetric(Diagonal(var[1:dimP]))                              # initial variance and covariance matrix
        meanVec = rand(Normal(0.0, 0.01), dimP) + param_MCMC[1:dimP,gene]       # initial mean vector, centered around the current MCMC state with small gaussian noise
        varMat = deepcopy(varInit)   
        lambdaVec = [0.01]                                                      #  Start with a small scaling factor.
        start = deepcopy(st)
        update = deepcopy(upd)

        new(alpha, varInit, meanVec,varMat,lambdaVec,stepVect, start, update, dimP)
    end  
end


struct adaptiveUniv_1 <: adaptiveStep_1
    alphaTarget::Float64
    varInit::Matrix{Float64}  
    varMat::Matrix{Float64}
    batchSize::Int64
    sumMHalpha::Vector{Float64}
    stepVect::Vector{Float64} 
    start::Vector{Bool} 
    update::Vector{Bool} 

    # Structure for handling adaptive MCMC with a univariate random-walk proposal as suggested in Robert and Casella [2009].
    # It stores variance estimates, acceptance tracking, and flags for when adaptation should start/stop.

    # Arguments
    # - alphaTarget: target acceptance rate
    # - varInit: initial variance for the random walk 
    # - varMat: current variance of the proposal distribution (updated adaptively).
    # - batchSize: number of iterations in one batch for adaptation updates.
    # - sumMHalpha: cumulative Metropolis–Hastings acceptance counts for adaptation
    # - stepVect: adaptation step-size parameters 
    # - start: flags indicating whether to start adaptive updates or continue with initial variance.
    # - update: flags indicating whether to keep updating after a certain iteration.

    # # Constructor
    #     adaptiveUniv_1(alpha, var, batch, stepVect, st, upd)

    function adaptiveUniv_1(alpha::Float64, var::Float64, batch::Int64, stepVect::Vector{Float64}, st::Vector{Bool}, upd::Vector{Bool})
        varInit::Matrix{Float64} = var .* ones(Float64, 1, 1)          
        varMat::Matrix{Float64} = deepcopy(varInit)
        sumMH::Float64 = 0.0
        start::Vector{Bool} = deepcopy(st)
        update::Vector{Bool} = deepcopy(upd)
        
        new(alpha, varInit,varMat,batch,[sumMH], stepVect, start, update)
    end
end


function updateAdapt!(
    adaptAlg::Vector{adaptiveAlg4}, 
    gene::Int64, 
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}}, 
    currIter::Int64, 
    MHalpha::Float64, 
    LogSS_MCMC::Matrix{Float64}
)

# Perform multivariate adaptive update for the proposal distribution of a given gene during MCMC sampling.
# It follows algorithm 4 of Andrieu and Thomas [2008]
#
# Arguments
# ----------
# - adaptAlg : Vector of adaptiveAlg4 structures storing adaptation parameters for each gene.
# - gene     : Index of the current gene being updated.
# - mcmc     : NamedTuple with fields (iter, thin, burnin) describing the MCMC settings.
# - currIter : Current MCMC iteration number.
# - MHalpha  : Acceptance probability.
# - LogSS_MCMC : Matrix of current log-scale steady-state parameter samples.
#
    
    # After 100 iterations we allow the adaptive variance update to start
    if currIter == 100
        adaptAlg[gene].start[:] .= true
    end

    # Stop adapting after 90% of the burn-in phase has been reached
    if currIter == round(0.9*mcmc.burnin)
        adaptAlg[gene].update[:] .= false
    end

    # Proceed with adaptation only if updates are still allowed
    if adaptAlg[gene].update[1]
        # Backup the current covariance matrix before trying to update it
        saveSigma = deepcopy(adaptAlg[gene].varMat)
        # Adaptation parameter A/(B+n)   
        step::Float64 = adaptAlg[gene].stepVect[1] / ( adaptAlg[gene].stepVect[2] + currIter)  

        # Update covariance matrix: Sigma <- Sigma + step * ((X - mean)(X - mean)^t - Sigma)
        adaptAlg[gene].varMat.data[:,:] .=  Symmetric(adaptAlg[gene].varMat .+ step .* (((LogSS_MCMC[1:adaptAlg[gene].dimP, gene] - adaptAlg[gene].meanVec) * transpose(LogSS_MCMC[1:adaptAlg[gene].dimP, gene] - adaptAlg[gene].meanVec)) .- adaptAlg[gene].varMat))
        
        # Check if the updated covariance matrix is positive definite
        if isposdef(adaptAlg[gene].varMat)
            #  update mean: mean <- mean + step * (X - mean)
            adaptAlg[gene].meanVec[:] .= adaptAlg[gene].meanVec .+ step .* (LogSS_MCMC[1:adaptAlg[gene].dimP, gene] .- adaptAlg[gene].meanVec)
            # update scaling factor: lambda <- lambda * exp(step * (alpha  - alpha_target))
            adaptAlg[gene].lambdaVec[:] .= exp.(log.(adaptAlg[gene].lambdaVec) .+  step *(MHalpha - adaptAlg[gene].alphaTarget))
        else
            # If update produces invalid covariance, restore previous one
            println("The matrix is not positive definite")
            adaptAlg[gene].varMat.data[:,:] = saveSigma.data[:,:]
        end
    end
end


function updateAdapt!(
    adaptAlg::Matrix{adaptiveUniv_1}, 
    gene::Int64, 
    cellTy::Vector{Int64}, 
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}}, 
    currIter::Int64, 
    MHalpha::Float64
)

# Perform univariate adaptive update for the proposal variance of a given gene during MCMC sampling, when the parater of interest depend on the group (cell type).
# It follows the algorithm suggested in Robert and Casella [2009].
#
# Arguments
# ----------
# - adaptAlg : Matrix of adaptiveUniv_1 structures, one per (cell type, gene) combination.
# - gene     : Index of the current gene being updated.
# - cellTy   : Vector of indices for the cell types being updated in this iteration.
# - mcmc     : NamedTuple with fields (iter, thin, burnin) describing the MCMC settings.
# - currIter : Current MCMC iteration number.
# - MHalpha  : Acceptance probability.
#
# Notes
# -----
# - Adaptation uses using batch averages of acceptance probabilities.


    # After 100 iterations, allow adaptive variance updates
    if currIter == 100
        for c = cellTy
            adaptAlg[c,gene].start[:] .= true
        end
    end

    # Stop adaptation after 90% of the burn-in phase
    if currIter == round(0.9*mcmc.burnin)
        for c = cellTy
            adaptAlg[c,gene].update[:] .= false
        end
    end

    # Accumulate acceptance probabilities for this batch
    for c = cellTy
        adaptAlg[c, gene].sumMHalpha[:] .= adaptAlg[c, gene].sumMHalpha[:] .+ MHalpha
    end
    
    # Perform variance update only if:
    #  1) adaptation is still allowed, AND
    #  2) current iteration is a multiple of batchSize
    if adaptAlg[cellTy[1],gene].update[1] & (mod(currIter, adaptAlg[cellTy[1], gene].batchSize)==0)
        step::Float64 = adaptAlg[cellTy[1],gene].stepVect[1] / ( adaptAlg[cellTy[1],gene].stepVect[2] + currIter)  # step size: step = A / (B + n)
        # Update variance: log(sigma^2) <- log(sigma^2) + step * (alpha_batch - alpha_target)
        adaptAlg[cellTy[1],gene].varMat[:,:] .= exp.(log.(adaptAlg[cellTy[1],gene].varMat[:,:]) + step .* (adaptAlg[cellTy[1], gene].sumMHalpha ./ adaptAlg[cellTy[1], gene].batchSize .-  adaptAlg[cellTy[1],gene].alphaTarget))

        # Synchronize variance across all cell types and reset counters
        for c = cellTy
            adaptAlg[c,gene].varMat[:,:] = adaptAlg[cellTy[1],gene].varMat[:,:]
            adaptAlg[c, gene].sumMHalpha[:] .= 0.0
        end
    end
end


function updateAdapt!(
    adaptAlg::Vector{adaptiveUniv_1}, 
    gc::Int64, 
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}}, 
    currIter::Int64, 
    MHalpha::Float64
)

# Perform univariate adaptive update for the proposal variance of a given gene/cell during MCMC sampling, when the parater of interest does not depend on the group (cell type).
# It follows the algorithm suggested in Robert and Casella [2009].
#
# Arguments
# ----------
# - adaptAlg : Matrix of adaptiveUniv_1 structures, one per (cell type, gene) combination.
# - gc       : Index of the current gene/cell being updated.
# - mcmc     : NamedTuple with fields (iter, thin, burnin) describing the MCMC settings.
# - currIter : Current MCMC iteration number.
# - MHalpha  : Acceptance probability.
#
# Notes
# -----
# - Adaptation uses using batch averages of acceptance probabilities.

    # After 100 iterations, allow adaptive variance updates
    if currIter == 100
        adaptAlg[gc].start[:] .= true
    end

    # Stop adaptation after 90% of the burn-in phase
    if currIter == round(0.9*mcmc.burnin)
        adaptAlg[gc].update[:] .= false
    end

    # Accumulate acceptance probabilities for this batch
    adaptAlg[gc].sumMHalpha[:] = adaptAlg[gc].sumMHalpha[:] .+ MHalpha
    
    # Perform variance update only if:
    #  1) adaptation is still allowed, AND
    #  2) current iteration is a multiple of batchSize
    if (adaptAlg[gc].update[1] & (mod(currIter, adaptAlg[gc].batchSize)==0))
        step::Float64 = adaptAlg[gc].stepVect[1] / ( adaptAlg[gc].stepVect[2] + currIter)  # step size: step = A / (B + n)
        # Update variance: log(sigma^2) <- log(sigma^2) + step * (alpha_batch - alpha_target)
        adaptAlg[gc].varMat[:,:] .= exp.(log.(adaptAlg[gc].varMat[:,:]) + step .* (adaptAlg[gc].sumMHalpha[:] ./ adaptAlg[gc].batchSize .-  adaptAlg[gc].alphaTarget))
        # Reset counters
        adaptAlg[gc].sumMHalpha[:] .= 0.0
    end
end



#################################
#      MCMC PROPOSALS
#################################
# multivariate proposal 
function proposal(
    lastAcc::Matrix{Float64}, 
    gene::Int64, 
    adapt::Vector{adaptiveAlg4}
)::Vector{Float64}

# Generate a new proposal vector from a multivariate random walk for the Metropolis–Hastings algorithm using either:
#   - the initial fixed covariance (before adaptation starts), or
#   - the adaptive covariance and scaling (after adaptation).
#
# Arguments
# ----------
# - lastAcc : Current accepted states. Only the first dimP rows for the given gene are used as the proposal mean.
# - gene    : Index of the gene being updated.
# - adapt   : Vector of adaptiveAlg4 adaptive MCMC structures, one per gene.
#
# Returns
# -------
# - prop : Proposed new state for the given gene.
#
# Notes
# -----
# - Before adaptation begins (start == false), the proposal is drawn from a multivariate normal distribution with the initial covariance matrix varInit.
# - Once adaptation starts, the proposal uses the adaptive covariance varMat, scaled by lambdaVec, plus a small jitter term for numerical stability.
###########################################################
    
    if(!adapt[gene].start[1]) # before adaptation starts --> use fixed initial variance
        prop = rand(MvNormal(lastAcc[1:adapt[gene].dimP, gene], adapt[gene].varInit[1:adapt[gene].dimP, 1:adapt[gene].dimP] + Diagonal([0.0000000000000000001 for i = 1:adapt[gene].dimP])))     
    else # after adaptation starts --> use adaptive variance
        prop = lastAcc[1:adapt[gene].dimP, gene] +  sqrt(adapt[gene].lambdaVec[1])*(cholesky(adapt[gene].varMat[1:adapt[gene].dimP,1:adapt[gene].dimP]).L * rand(Normal(0.0,1.0),adapt[gene].dimP)) +  rand(Normal(0.0,0.000000000000001^0.5),adapt[gene].dimP)
    end
    return prop
end

function proposal(
    lastAcc::Float64, 
    gene::Int64, 
    cellTy::Vector{Int64}, 
    adapt::Matrix{adaptiveUniv_1}
)
# Generate a new proposal value from a univariate random walk for the Metropolis–Hastings algorithm, for parameters that depend on the group (cell type). 
# It uses either:
#   - the initial fixed variance (before adaptation starts), or
#   - the adaptive variance (after adaptation).
#
# Arguments
# ----------
# - lastAcc : Current accepted state.
# - gene    : Index of the gene being updated.
# - cellTy  : Vector of indices for the cell types being updated in this iteration.
# - adapt   : Vector of adaptiveUniv_1 adaptive MCMC structures, one per gene.
#
# Returns
# -------
# - prop : Proposed new state for the given gene.
#
# Notes
# -----
# - Before adaptation begins (start == false), the proposal is drawn from a univariate normal distribution with variance varInit.
# - Once adaptation starts, the proposal uses the adaptive variance varMat.

    if !adapt[cellTy[1],gene].start[1] # before adaptation starts --> use fixed initial variance
        prop = rand(Normal(lastAcc, sqrt.(adapt[cellTy[1], gene].varInit[1,1])))     
    else # after adaptation starts --> use adaptive variance
        prop = rand(Normal(lastAcc, sqrt.(adapt[cellTy[1], gene].varMat[1,1])))    
    end
    return prop
end

# univariate proposal for parameters that do not depend on the type of cells
function proposal(
    lastAcc::Float64, 
    gene::Int64, 
    adapt::Vector{adaptiveUniv_1}
)
# Generate a new proposal value from a univariate random walk for the Metropolis–Hastings algorithm, for parameters that do not depend on the group. 
# It uses either:
#   - the initial fixed variance (before adaptation starts), or
#   - the adaptive variance (after adaptation).
#
# Arguments
# ----------
# - lastAcc : Current accepted state.
# - gene    : Index of the gene being updated.
# - adapt   : Vector of adaptiveUniv_1 adaptive MCMC structures, one per gene.
#
# Returns
# -------
# - prop : Proposed new state for the given gene.
#
# Notes
# -----
# - Before adaptation begins (start == false), the proposal is drawn from a univariate normal distribution with variance varInit.
# - Once adaptation starts, the proposal uses the adaptive variance varMat.
    if !adapt[gene].start[1] # before adaptation starts --> use fixed initial variance
        prop = rand(Normal(lastAcc, sqrt.(adapt[gene].varInit[1,1])))     
    else # after adaptation starts --> use adaptive variance
        prop = rand(Normal(lastAcc, sqrt.(adapt[gene].varMat[1,1])))    
    end
    return prop
end

#################################
#      DISTRIBUTIONS
#################################
struct Log_Beta <: Distribution{Univariate, Continuous}
    par1::Float64
    par2::Float64
    par3::Float64
    B::Beta{Float64}

    # Truncated Beta distribution but defined on the log scale.
    #
    # Arguments
    # ------
    # - par1 : First shape parameter of the Beta distribution.
    # - par2 : Second shape parameter of the Beta distribution.
    # - par3 : Upper limit of truncation interval
    # - B    : Standard Beta distribution with parameters (par1, par2).
    #
    # Constructor
    # -----------
    # - Log_Beta(par1, par2, par3)
    #   Creates a new Log_Beta distribution object and internally builds the corresponding Beta distribution.
  
    function Log_Beta(par1, par2, par3)
        new(par1, par2, par3, Beta(par1, par2))
    end
end

# uniform between 0 and 1 with probability 1 - p_zero, 0 with probability p
struct UniformWithMass <: Distribution{Univariate, Continuous}
    p_zero::Float64
    Unif::Uniform{Float64}

    # A univariate continuous distribution with mass in 0. Specifically, it is
    # - 0 with probability p_zero
    # -Unif(0,1) with probability 1 - p_zero

    # Arguments
    # ------
    # - p_zero : Probability mass at zero.
    # - Unif   : Standard Uniform(0,1) distribution.
    #
    # Constructor
    # -------
    # Creates a new UniformWithMass distribution with a probability mass at 0 of size p_zero.

    function UniformWithMass(p_zero)
        new(p_zero, Uniform(0.0, 1.0))
    end
end

struct Log_Exponential <: Distribution{Univariate, Continuous}
    phi::Float64
    E::Exponential{Float64}
    
    # Univariate Exponential distribution but defined on the log scale.
    
    # Arguments
    # ------
    # - phi : parameter of the Exponential distribution
    # - E   : Standard Exponential distribution
    
    # Constructor
    # -------
    # Creates a new Log_Exponential distribution with paramters phi.

    function Log_Exponential(phi)
        new(phi, Exponential(phi))
    end
end


# --------------- PROBABILITY FUNCTIONS associated with the defined distribution
# Notes : 
# PDF: probability density function
# CDF: cumulative distribution function

function logpdf(distr::Distribution, x::Float64)
    # PDF function for the continuous distributions that have not been redefined in BayVel

    # Arguments 
    # ------------
    # dist: type of distribution
    # x   : continuous value in which we want to evaluate the PDF
    Distributions.logpdf(distr, x)

end

function logpdf(distr::Distribution, x::Int64)
    # PDF function for the discrete distributions that have not been redefined in BayVel

    # Arguments 
    # ------------
    # dist: type of distribution
    # x   : discrete value in which we want to evaluate the PDF
    Distributions.logpdf(distr, x)
end

function quantile(distr::Distribution, x::Float64)
    # quantile function for the distributions that have not been redefined in BayVel

    # Arguments 
    # ------------
    # dist: type of distribution
    # x   : values of the quantile that we want to compute
    Distributions.quantile(distr, x)
end

function cdf(distr::Distribution, x::Float64)
    # CDF function for all the continuous distributions that have not been redefined in BayVel

    # Arguments 
    # ------------
    # dist: type of distribution
    # x   : continuous value in which we want to evaluate the CDF
    Distributions.cdf(distr, x)
end

function cdf(distr::Distribution, x::Int64)
    # CDF function for all the discrete distributions that have not been redefined in BayVel

    # Arguments 
    # ------------
    # dist: type of distribution
    # x   : continuous value in which we want to evaluate the CDF
    Distributions.cdf(distr, x)
end


###### per quelle che ridefiniamo noi 

function logpdf(
    dist::Log_Exponential,
    logX::Float64
)

    # logarithm of the PDF function for the Log_Exponential distribution defined in BayVel

    # Arguments 
    # ------------
    # dist: type of distribution (Log_Exponential)
    # x   : continuous value in which we want to evaluate the log-PDF
    res::Float64 = logX + logpdf(dist.E, exp(logX))

    return res
end