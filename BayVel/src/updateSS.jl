function updateSS!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    LogSS_MCMC::Matrix{Float64},
    SS_MCMC::Matrix{Float64},
    LogSS_Star_MCMC::Matrix{Float64},
    SS_Star_MCMC::Matrix{Float64},
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    TStar_MCMC::Matrix{Float64},
    TStar_withM_MCMC::Matrix{Float64},
    Tau_MCMC::Matrix{Float64},
    k_MCMC::Matrix{Int64},
    u_MCMC::Matrix{Float64},
    phi_MCMC::Matrix{Float64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    priorsLogU_SS_on::Distribution{Univariate, Continuous},
    priorsLogS_SS_on::Distribution{Univariate, Continuous},
    priorsLogU_SS_off::Distribution{Univariate, Continuous},
    priorsLogBeta::Distribution{Univariate, Continuous},
    priorsT::Distribution{Univariate, Continuous},
    adaptVar::Vector{<:adaptiveStep_1},
    mu_u_PROP::Matrix{Float64}, 
    mu_s_PROP::Matrix{Float64}, 
    propStarLog::Matrix{Float64},
    propStar::Matrix{Float64},  
    SS_PROP::Matrix{Float64},
    u0_off_PROP::Matrix{Float64},
    s0_off_PROP::Matrix{Float64},
    TStar_PROP::Matrix{Float64},
    Tau_PROP::Matrix{Float64},
    k_PROP::Matrix{Int64},
    u_PROP::Matrix{Float64},
    phi_PROP::Matrix{Float64},
    prior_PROP::Vector{Float64}, 
    priorTStar_PROP::Matrix{Float64},
    loglik_PROP::Matrix{Float64},

    mu_u_MCMC::Matrix{Float64}, 
    mu_s_MCMC::Matrix{Float64}, 
    prior_MCMC::Vector{Float64},
    priorTStar_MCMC::Matrix{Float64},
    loglik_MCMC::Matrix{Float64},
    acceptRate::Vector{Int64}, 
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}},
    currIter::Int64, 
    model::mod,
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64},
    typeCellT0_off::Vector{Int64},
    par2::Matrix{Float64},
    par4::Matrix{Float64}
)where{mod<:modelType}
  
# MCMC update for the steady-state parameters for all genes.
# This function is a wrapper around the single-gene updateSS! function, iterating over all genes and launching their updates in parallel.
#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - LogSS_MCMC: Current log-transformed values for u-coordinate lower steady state, s-coordinate lower steady state, difference between the u-coordinates of the steady states, beta
# - SS_MCMC: Current values for u-coordinate lower steady state, s-coordinate lower steady state, difference between the u-coordinates of the steady states, beta, in the natural scale
# - LogSS_Star_MCMC: Current log-transformed values for u-coordinate of upper steady state, s-coordinate of upper steady state, u-coordinate of lower steady state, beta
# - SS_Star_MCMC: Current values for u-coordinate of upper steady state, s-coordinate of upper steady state, u-coordinate of lower steady state, beta, in the natural scale
# - t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, k_MCMC, u_MCMC, phi_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC: Current values of model parameters
# - priorsLogU_SS_on: prior distribution for logarithm of the u-coordinate of the upper steady state
# - priorsLogS_SS_on: prior distribution for logarithm of the s-coordinate of the upper steady state
# - priorsLogU_SS_off: prior distribution for logarithm of the u-coordinate of the lower steady state
# - priorsLogBeta: prior distribution for logarithm of Beta
# - priorsT: prior distribution for time
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - mu_u_PROP, mu_s_PROP: Proposed means for unspliced/spliced counts distribution
# - propStarLog: Proposed values for logarithm of u-coordinate of upper steady state, s-coordinate of upper steady state, u-coordinate of lower steady state, beta
# - propStar: Proposed values of u-coordinate of upper steady state, s-coordinate of upper steady state, u-coordinate of lower steady state, beta, in the natural scale
# - SS_PROP: Proposed values of u-coordinate of lower steady state, s-coordinate of lower steady state, difference of the u-coordinate of steady states, beta, in the natural scale
# - u0_off_PROP, s0_off_PROP, TStar_PROP, Tau_PROP, k_PROP, u_PROP, phi_PROP: Other values of the model that are modified as a consequence of the new proposed values for the steady-states
# - prior_PROP: Prior density of the proposed steady-states
# - priorTStar_PROP: Prior density for the proposed times (modified as a consequence of the steady states)
# - loglik_PROP: Log-likelihood values of the proposed values
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities for steady-states
# - priorTStar_MCMC: Current prior densities for times
# - loglik_MCMC: Current log-likelihood values.
# - acceptRate: Vector storing acceptance rate
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - subtypeCell: Vector of cell subtype indices
# - typeCell:  Vector of cell type indices
# - typeCellT0_off:  Vector of switching clustering for all the cells
# - par2, par4: Intermediate parameter matrices used in likelihood.

    # iterates in parallel over the different genes 
    Threads.@threads for g = 1:size(unspliced, 2)
        updateSS!(unspliced, spliced, LogSS_MCMC, SS_MCMC, LogSS_Star_MCMC, SS_Star_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, k_MCMC, u_MCMC, phi_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC, priorsLogU_SS_on, priorsLogS_SS_on, priorsLogU_SS_off, priorsLogBeta, priorsT, adaptVar, mu_u_PROP, mu_s_PROP, propStarLog, propStar, SS_PROP, u0_off_PROP, s0_off_PROP, TStar_PROP, Tau_PROP, k_PROP, u_PROP, phi_PROP, prior_PROP, priorTStar_PROP, loglik_PROP, mu_u_MCMC, mu_s_MCMC, prior_MCMC, priorTStar_MCMC, loglik_MCMC, g, acceptRate, mcmc, currIter, model, subtypeCell, typeCell, typeCellT0_off, par2, par4)
    end

    return nothing
end

function updateSS!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    LogSS_MCMC::Matrix{Float64},
    SS_MCMC::Matrix{Float64},
    LogSS_Star_MCMC::Matrix{Float64},
    SS_Star_MCMC::Matrix{Float64},
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    TStar_MCMC::Matrix{Float64},
    TStar_withM_MCMC::Matrix{Float64},
    Tau_MCMC::Matrix{Float64},
    k_MCMC::Matrix{Int64},
    u_MCMC::Matrix{Float64},
    phi_MCMC::Matrix{Float64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    priorsLogU_SS_on::Distribution{Univariate, Continuous},
    priorsLogS_SS_on::Distribution{Univariate, Continuous},
    priorsLogU_SS_off::Distribution{Univariate, Continuous},
    priorsLogBeta::Distribution{Univariate, Continuous},
    priorsT::Distribution{Univariate, Continuous},
    adaptVar::Vector{<:adaptiveStep_1}, 
    mu_u_PROP::Matrix{Float64}, 
    mu_s_PROP::Matrix{Float64},
    propStarLog::Matrix{Float64},      
    propStar::Matrix{Float64},  
    SS_PROP::Matrix{Float64},
    u0_off_PROP::Matrix{Float64},
    s0_off_PROP::Matrix{Float64},
    TStar_PROP::Matrix{Float64},
    Tau_PROP::Matrix{Float64},
    k_PROP::Matrix{Int64},
    u_PROP::Matrix{Float64},
    phi_PROP::Matrix{Float64},
    prior_PROP::Vector{Float64},  
    priorTStar_PROP::Matrix{Float64},
    loglik_PROP::Matrix{Float64}, 
    mu_u_MCMC::Matrix{Float64}, 
    mu_s_MCMC::Matrix{Float64}, 
    prior_MCMC::Vector{Float64},
    priorTStar_MCMC::Matrix{Float64},
    loglik_MCMC::Matrix{Float64},
    gene::Int64,
    acceptRate::Vector{Int64},
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}},
    currIter::Int64, 
    model::mod,
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64},
    typeCellT0_off::Vector{Int64},
    par2::Matrix{Float64},
    par4::Matrix{Float64}
)where{mod<:modelType}

# MCMC update for the steady-state parameters for a selected gene. Note that phi will remain fixed for all the cells, so we will need to update both the time parameters and the switching time parameters.
#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - LogSS_MCMC: Current log-transformed values for u-coordinate lower steady state, s-coordinate lower steady state, difference between the u-coordinates of the steady states, beta
# - SS_MCMC: Current values for u-coordinate lower steady state, s-coordinate lower steady state, difference between the u-coordinates of the steady states, beta, in the natural scale
# - LogSS_Star_MCMC: Current log-transformed values for u-coordinate of upper steady state, s-coordinate of upper steady state, u-coordinate of lower steady state, beta
# - SS_Star_MCMC: Current values for u-coordinate of upper steady state, s-coordinate of upper steady state, u-coordinate of lower steady state, beta, in the natural scale
# - t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, k_MCMC, u_MCMC, phi_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC: Current values of model parameters
# - priorsLogU_SS_on: prior distribution for logarithm of the u-coordinate of the upper steady state
# - priorsLogS_SS_on: prior distribution for logarithm of the s-coordinate of the upper steady state
# - priorsLogU_SS_off: prior distribution for logarithm of the u-coordinate of the lower steady state
# - priorsLogBeta: prior distribution for logarithm of Beta
# - priorsT: prior distribution for time
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - mu_u_PROP, mu_s_PROP: Proposed means for unspliced/spliced counts distribution
# - propStarLog: Proposed values for logarithm of u-coordinate of upper steady state, s-coordinate of upper steady state, u-coordinate of lower steady state, beta
# - propStar: Proposed values of u-coordinate of upper steady state, s-coordinate of upper steady state, u-coordinate of lower steady state, beta, in the natural scale
# - SS_PROP: Proposed values of u-coordinate of lower steady state, s-coordinate of lower steady state, difference of the u-coordinate of steady states, beta, in the natural scale
# - u0_off_PROP, s0_off_PROP, TStar_PROP, Tau_PROP, k_PROP, u_PROP, phi_PROP: Other values of the model that are modified as a consequence of the new proposed values for the steady-states
# - prior_PROP: Prior density of the proposed steady-states
# - priorTStar_PROP: Prior density for the proposed times (modified as a consequence of the steady states)
# - loglik_PROP: Log-likelihood values of the proposed values
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities for steady-states
# - priorTStar_MCMC: Current prior densities for times
# - loglik_MCMC: Current log-likelihood values.
# - gene: index of the gene we are considering
# - acceptRate: Vector storing acceptance rate
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - subtypeCell: Vector of cell subtype indices
# - typeCell:  Vector of cell type indices
# - typeCellT0_off:  Vector of switching clustering for all the cells
# - par2, par4: Intermediate parameter matrices used in likelihood.

    # ------------------
    # New proposed values
    # ------------------
    # We propose: 
    # - log(U_on)= log(alpha_on/beta)
    # - log(S_on) = log(alpha_on/gamma)
    # - log(U_off) = log(alpha_off/beta)
    # But at the end we will return log(U_off), log(S_off), log(Diff_Off)
    # -------------

    propStarLog[1:adaptVar[gene].dimP,gene] = proposal(LogSS_Star_MCMC , gene, adaptVar)  # log-scale
    propStar[:,gene] .= exp.(propStarLog[:,gene]) # natural scale
    
    # check if the parameters are bigger than their bounds (the priors are truncated distributions)
    failedBounds::Bool = false
    if (propStar[1, gene] > priorsLogU_SS_on.par3) |  (propStar[2, gene] > priorsLogS_SS_on.par3) | (propStar[3, gene] >= propStar[1, gene])
        failedBounds = true
    end

    MHalpha::Float64 = 0.0 

    if !failedBounds    # respected bounds
        SS_PROP[1, gene], SS_PROP[2, gene], SS_PROP[3, gene] = ssStar_to_ss(propStar, gene, model) # transform the proposed values into log(U_off), log(S_off), log(Diff_Off)

        # update u0_off and s0_off and check if they are negative
        failedU0S0::Bool = mu_tot_g_withFailed!(SS_PROP, t0_off_MCMC, u0_off_PROP, s0_off_PROP, Tau_MCMC, k_MCMC, mu_u_PROP, mu_s_PROP, gene, catt_MCMC, true, typeCellT0_off, model) # 
        
        k_PROP[:, gene] .= 1
        # update time related parameters with this new values of the steady states, since phi remains fixed
        for sty = unique(subtypeCell)
            subcellTy::Vector{Int64} = findall(subtypeCell .== sty)
            TStar_PROP[subcellTy, gene] .= trasformPhi_toT(phi_MCMC, priorsT, SS_PROP, t0_off_MCMC, u0_off_PROP, gene, subcellTy, subtypeCell, typeCellT0_off, model)
            Tau_PROP[subcellTy, gene] .= TStar_PROP[subcellTy, gene]

            tyT0_off = typeCellT0_off[subcellTy[1]]
            if TStar_PROP[subcellTy[1], gene] > t0_off_MCMC[tyT0_off, gene]
                Tau_PROP[subcellTy, gene] .= TStar_PROP[subcellTy[1], gene] .- t0_off_MCMC[tyT0_off, gene]
                k_PROP[subcellTy, gene] .= 0
            end
        end
       
        # update the mean of the spliced and unspliced distribution according to the new proposed value and check if they are negative
        failed::Bool = mu_tot_g_withFailed!(SS_PROP, t0_off_MCMC, u0_off_PROP, s0_off_PROP, Tau_PROP, k_PROP, mu_u_PROP, mu_s_PROP, gene, catt_MCMC, false, typeCellT0_off, model) # aggiorniamo mu_u_PROP
        u_PROP[:, gene] .= mu_u_PROP[:, gene] ./ catt_MCMC[:]

        if (!failed & !failedU0S0)
            loglik_g!(unspliced, spliced, mu_u_PROP, mu_s_PROP, invEta_MCMC, loglik_PROP, gene, par2, par4, model) # update the log-likelihood according to the new proposed value
            
            prior_PROP[gene] = priorLogSS_Star(propStarLog, priorsLogU_SS_on, priorsLogS_SS_on, priorsLogU_SS_off, priorsLogBeta, gene, model)        # update prior distribution according to the new proposed value
                
            numerator::Float64 = sum(loglik_PROP[:,gene]) + prior_PROP[gene]    # update numerator of MH rate
        
            # ------------------
            # Current values
            # ------------------    
            denominator::Float64 = sum(loglik_MCMC[:,gene]) + prior_MCMC[gene] # recompute denominator of MH rate
        
            # Compute MH rate
            MHalpha = min(1.0,exp((numerator - denominator)))
        
        else
            MHalpha = 0.0 # discard the new proposed values if the switching points and the means of the counts distribution are negative
        end
    else
        MHalpha = 0.0 # discard the new proposed values if they do not respect the bounds of the prior distributions (priors density == 0) 
    end
   
    # Decide if accept or not the new proposed value. In case of acceptance, update all the modified quantities    
    if rand(Uniform(0.0,1.0)) < MHalpha
        LogSS_Star_MCMC[1:adaptVar[gene].dimP,gene] .= propStarLog[1:adaptVar[gene].dimP,gene]
        SS_Star_MCMC[1:adaptVar[gene].dimP,gene] .= propStar[1:adaptVar[gene].dimP,gene]
        
        SS_MCMC[1:adaptVar[gene].dimP,gene] .= SS_PROP[1:adaptVar[gene].dimP,gene]
        LogSS_MCMC[1:adaptVar[gene].dimP,gene] .= log.(SS_PROP[1:adaptVar[gene].dimP,gene])

        u0_off_MCMC[:, gene] .= u0_off_PROP[:, gene]
        s0_off_MCMC[:, gene] .= s0_off_PROP[:, gene]
        TStar_MCMC[:, gene] .= TStar_PROP[:, gene]
        TStar_withM_MCMC[:, gene] .= TStar_MCMC[:, gene]
        TStar_withM_MCMC[findall(TStar_withM_MCMC[:, gene] .< 0.0)] .= 0.0
        k_MCMC[:, gene] .= k_PROP[:, gene]
        u_MCMC[:, gene] .= u_PROP[:, gene]
        mu_u_MCMC[:,gene] .= mu_u_PROP[:,gene]
        mu_s_MCMC[:,gene] .= mu_s_PROP[:,gene]
        loglik_MCMC[:,gene] .= loglik_PROP[:,gene]

        prior_MCMC[gene] = deepcopy(prior_PROP[gene])
        acceptRate[gene] = acceptRate[gene] + 1    
    end

    # ------------------
    # Update variance of proposal distribution for adaptive MCMC
    # ------------------
    updateAdapt!(adaptVar, gene, mcmc, currIter, MHalpha, LogSS_Star_MCMC)

end
