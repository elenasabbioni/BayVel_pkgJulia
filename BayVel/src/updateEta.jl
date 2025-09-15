function updateETA!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    SS_MCMC::Matrix{Float64},
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    TStar_MCMC::Matrix{Float64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    priorsEta::Distribution{Univariate, Continuous},
    adaptVar::Vector{<:adaptiveStep_1}, 
    prop::Vector{Float64},    
    prior_PROP::Vector{Float64},  
    loglik_PROP::Matrix{Float64}, 
    mu_u_MCMC::Matrix{Float64}, 
    mu_s_MCMC::Matrix{Float64}, 
    prior_MCMC::Vector{Float64},
    loglik_MCMC::Matrix{Float64},
    acceptRate::Vector{Int64}, 
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}},
    currIter::Int64,
    model::mod,
    par2::Matrix{Float64},
    par4::Matrix{Float64}, 
)where{mod<:modelType}


# MCMC update for the overdispersion for all the genes, 
# This function is a wrapper around the single-gene updateETA! function, iterating over all genes and launching their updates in parallel.
#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC: Current values of model parameters
# - LogEta_MCMC: current log-transformed values of the overdispersion 
# - invEta_MCMC: Current values of 1/overdispersion 
# - catt_MCMC: Current values for the capture efficiency in the natural scale
# - priorsEta: Prior distribution over overdispersion
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - prop: Proposed values for the overdispersions in the natural scale
# - prior_PROP: Prior densities of the proposed values
# - loglik_PROP: Log-likelihood values of the proposed values
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities.
# - loglik_MCMC: Current log-likelihood values.
# - acceptRate: Vector storing acceptance rate
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - par2, par4: Intermediate parameter matrices used in likelihood.

    # iterates in parallel over the different genes
    Threads.@threads for g = 1:size(unspliced, 2)
        updateETA!(unspliced, spliced, SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC, priorsEta, adaptVar, prop, prior_PROP, loglik_PROP, mu_u_MCMC, mu_s_MCMC, prior_MCMC, loglik_MCMC, acceptRate, mcmc, currIter, g, model, par2, par4)
    end
end

function updateETA!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    SS_MCMC::Matrix{Float64},
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    TStar_MCMC::Matrix{Float64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    priorsEta::Distribution{Univariate, Continuous},
    adaptVar::Vector{<:adaptiveStep_1}, 
    prop::Vector{Float64},     
    prior_PROP::Vector{Float64},  
    loglik_PROP::Matrix{Float64},
    mu_u_MCMC::Matrix{Float64}, 
    mu_s_MCMC::Matrix{Float64}, 
    prior_MCMC::Vector{Float64},
    loglik_MCMC::Matrix{Float64},
    acceptRate::Vector{Int64}, 
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}},
    currIter::Int64, 
    gene::Int64,
    model::mod, 
    par2::Matrix{Float64},
    par4::Matrix{Float64}, 
)where{mod<:modelType}
# MCMC update for the overdispersion for all the genes, 
# This function is a wrapper around the single-gene updateETA! function, iterating over all genes and launching their updates in parallel.
#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC: Current values of model parameters
# - LogEta_MCMC: current log-transformed values of the overdispersion 
# - invEta_MCMC: Current values of 1/overdispersion 
# - catt_MCMC: Current values for the capture efficiency in the natural scale
# - priorsEta: Prior distribution over overdispersion
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - prop: Proposed values for the overdispersions in the natural scale
# - prior_PROP: Prior densities of the proposed values
# - loglik_PROP: Log-likelihood values of the proposed values
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities.
# - loglik_MCMC: Current log-likelihood values.
# - acceptRate: Vector storing acceptance rate
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - gene: index of the gene we are considering
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - par2, par4: Intermediate parameter matrices used in likelihood.

    
    # ------------------
    # New proposed value
    # ------------------
    prop[gene] = proposal(LogEta_MCMC[gene], gene, adaptVar) # logarithmic scale   
    prop[gene] = exp(-prop[gene]) # natural scale
    loglik_g!(unspliced, spliced, mu_u_MCMC, mu_s_MCMC, prop, loglik_PROP, gene, par2, par4, model) # update the log-likelihood according to the new proposed value
    prop[gene] = -log(prop[gene]) # logarithmic scale

    prior_PROP[gene] = priorETA(prop, priorsEta, gene, model)     # update prior distribution according to the new proposed value

    numerator::Float64 = sum(loglik_PROP[:,gene]) + prior_PROP[gene]    # update numerator of MH rate

    # ------------------
    # Current value
    # ------------------
    denominator::Float64 = sum(loglik_MCMC[:,gene]) + prior_MCMC[gene]  # recompute denominator of MH rate

    # Compute MH rate
    MHalpha::Float64 = min(1,exp((numerator - denominator)))
  
    # Decide if accept or not the new proposed value. In case of acceptance, update all the modified quantities    
    if rand(Uniform(0.0,1.0))< MHalpha
        LogEta_MCMC[gene] = prop[gene]
        invEta_MCMC[gene] = exp(-LogEta_MCMC[gene])

        loglik_MCMC[:,gene] .= loglik_PROP[:,gene]
        prior_MCMC[gene] = prior_PROP[gene]

        acceptRate[gene] = acceptRate[gene] + 1    
    end

    # ------------------
    # Update variance of proposal distribution for adaptive MCMC
    # ------------------
    updateAdapt!(adaptVar, gene, mcmc, currIter, MHalpha)    
    
end



