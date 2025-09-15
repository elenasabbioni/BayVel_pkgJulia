function updateCATT!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    SS_MCMC::Matrix{Float64},
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    TStar_MCMC::Matrix{Float64},
    Tau_MCMC::Matrix{Float64},
    k_MCMC::Matrix{Int64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    LogitCatt_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    priorsCatt::Distribution{Univariate, Continuous},
    adaptVar::Vector{<:adaptiveStep_1}, 
    mu_u_PROP::Matrix{Float64},
    mu_s_PROP::Matrix{Float64},
    Logit_PROP::Vector{Float64},
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
    typeCell::Vector{Int64},
    typeCellT0_off::Vector{Int64},
    par2::Matrix{Float64},
    par4::Matrix{Float64}, 
)where{mod<:modelType}


# MCMC update for the capture efficiency for all cells.
# This function is a wrapper around the single-cell updateCATT! function, iterating over all cells and launching their updates in parallel.
#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, Tau_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC: Current values of model parameters
# - LogitCatt_MCMC: Current logit-transformed values for the capture efficiency
# - catt_MCMC: Current values for the capture efficiency in the natural scale
# - priorsCatt: Prior distribution over capture efficiency
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - mu_u_PROP, mu_s_PROP: Proposed means for unspliced/spliced counts distribution
# - Logit_PROP: Proposed logit-transformed values for the capture efficiency
# - prop: Proposed values for the capture efficiency in the natural scale
# - prior_PROP: Prior densities of the proposed values
# - loglik_PROP: Log-likelihood values of the proposed values#
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities.
# - loglik_MCMC: Current log-likelihood values.
# - acceptRate: Vector storing acceptance rate
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - typeCell:  Vector of cell type indices
# - typeCellT0_off:  Vector of switching clustering for all the cells
# - par2, par4: Intermediate parameter matrices used in likelihood.

    # iterates in parallel over the different cells 
    Threads.@threads for c = 1:size(unspliced, 1)
        updateCATT!(unspliced, spliced, SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, Tau_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC, LogitCatt_MCMC, catt_MCMC, priorsCatt, adaptVar, mu_u_PROP, mu_s_PROP, Logit_PROP, prop, prior_PROP, loglik_PROP, mu_u_MCMC, mu_s_MCMC, prior_MCMC, loglik_MCMC, acceptRate, mcmc, currIter, c, model, typeCell, typeCellT0_off, par2, par4)
    end
end


function updateCATT!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    SS_MCMC::Matrix{Float64},
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    TStar_MCMC::Matrix{Float64},
    Tau_MCMC::Matrix{Float64},
    k_MCMC::Matrix{Int64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    LogitCatt_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    priorsCatt::Distribution{Univariate, Continuous},
    adaptVar::Vector{<:adaptiveStep_1},
    mu_u_PROP::Matrix{Float64}, 
    mu_s_PROP::Matrix{Float64},
    Logit_PROP::Vector{Float64},
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
    cell::Int64,
    model::mod,
    typeCell::Vector{Int64},
    typeCellT0_off::Vector{Int64},
    par2::Matrix{Float64},
    par4::Matrix{Float64}
)where{mod<:modelType}

# This function is a wrapper around the single-cell updateCATT! function, iterating over all cells and launching their updates in parallel.
#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, Tau_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC: Current values of model parameters
# - LogitCatt_MCMC: Current logit-transformed values for the capture efficiency
# - catt_MCMC: Current values for the capture efficiency in the natural scale
# - priorsCatt: Prior distribution over capture efficiency
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - mu_u_PROP, mu_s_PROP: Proposed means for unspliced/spliced counts distribution
# - Logit_PROP: Proposed logit-transformed values for the capture efficiency
# - prop: Proposed values for the capture efficiency in the natural scale
# - prior_PROP: Prior densities of the proposed values
# - loglik_PROP: Log-likelihood values of the proposed values#
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities.
# - loglik_MCMC: Current log-likelihood values.
# - acceptRate: Vector storing acceptance rate
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - cell: index of the cell we are considering.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - typeCell:  Vector of cell type indices
# - typeCellT0_off:  Vector of switching clustering for all the cells
# - par2, par4: Intermediate parameter matrices used in likelihood.


    # ------------------
    # New proposed value
    # ------------------
    Logit_PROP[cell] = proposal(LogitCatt_MCMC[cell], cell, adaptVar) # logit scale
    prop[cell] = logistic(Logit_PROP[cell]) # natural scale
    
    mu_tot_c!(SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, Tau_MCMC, k_MCMC, mu_u_PROP, mu_s_PROP, cell, prop, false, typeCellT0_off, model)  # update the mean of the spliced and unspliced distribution according to the new proposed value
    
    loglik_c!(unspliced, spliced, mu_u_PROP, mu_s_PROP, invEta_MCMC, loglik_PROP, cell, par2, par4, model) # update the log-likelihood according to the new proposed value

    prior_PROP[cell] = priorCATT(Logit_PROP, priorsCatt, cell, model)    # update prior distribution according to the new proposed value

    numerator::Float64 = sum(loglik_PROP[cell,:]) + prior_PROP[cell]  # update numerator of MH rate

   
    # ------------------
    # Current value
    # ------------------
    denominator::Float64 = sum(loglik_MCMC[cell,:]) + prior_MCMC[cell] # recompute denominator of MH rate

    # Compute MH rate
    MHalpha::Float64 = min(1,exp((numerator - denominator))) 

    # Decide if accept or not the new proposed value. In case of acceptance, update all the modified quantities    
    if rand(Uniform(0.0,1.0))< MHalpha
        LogitCatt_MCMC[cell] = Logit_PROP[cell]
        catt_MCMC[cell] = prop[cell]
        mu_u_MCMC[cell, :] = mu_u_PROP[cell, :]
        mu_s_MCMC[cell, :] = mu_s_PROP[cell, :]
        loglik_MCMC[cell,:] .= loglik_PROP[cell,:]
        prior_MCMC[cell] = prior_PROP[cell]

        acceptRate[cell] = acceptRate[cell] + 1    
    end


    # ------------------
    # Update variance of proposal distribution for adaptive MCMC
    # ------------------
    updateAdapt!(adaptVar, cell, mcmc, currIter, MHalpha)    

end

