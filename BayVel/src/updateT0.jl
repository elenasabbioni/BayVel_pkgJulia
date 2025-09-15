function updateT0!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    SS_MCMC::Matrix{Float64},
    LogT0_off_MCMC::Matrix{Float64},
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    TStar_MCMC::Matrix{Float64},
    TStar_withM_MCMC::Matrix{Float64},
    Tau_MCMC::Matrix{Float64},
    phi_MCMC::Matrix{Float64},
    u_MCMC::Matrix{Float64},
    k_MCMC::Matrix{Int64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    priorsLogT0_off::Distribution{Univariate, Continuous},
    priorsT::Distribution{Univariate, Continuous},
    adaptVar::Matrix{<:adaptiveStep_1}, 
    mu_u_PROP::Matrix{Float64}, 
    mu_s_PROP::Matrix{Float64}, 
    propLog::Matrix{Float64},
    prop::Matrix{Float64},      
    u0_off_PROP::Matrix{Float64},
    s0_off_PROP::Matrix{Float64},
    k_PROP::Matrix{Int64},
    TStar_PROP::Matrix{Float64},
    Tau_PROP::Matrix{Float64},
    phi_PROP::Matrix{Float64}, 
    u_PROP::Matrix{Float64},
    prior_PROP::Matrix{Float64},
    priorTStar_PROP::Matrix{Float64},
    loglik_PROP::Matrix{Float64},
    mu_u_MCMC::Matrix{Float64}, 
    mu_s_MCMC::Matrix{Float64}, 
    prior_MCMC::Matrix{Float64},
    priorTStar_MCMC::Matrix{Float64},
    loglik_MCMC::Matrix{Float64},
    acceptRate::Matrix{Int64}, 
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}},
    currIter::Int64, 
    model::groupSubgroup,
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64},
    typeCellT0_off::Vector{Int64},
    par2::Matrix{Float64},
    par4::Matrix{Float64}
)

# MCMC update for the switching time parameters for all the genes and all the switching clusters. 
# This function is a wrapper around the single-gene and cluster updateT0! function, iterating over all genes and clusters and launching their updates in parallel.

#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - SS_MCMC: Current values for u-coordinate lower steady state, s-coordinate lower steady state, difference between the u-coordinates of the steady states, beta, in the natural scale
# - LogT0_off_MCMC: Current log-transformed values for off-switching time
# - t0_off_MCMC: Current values for off-switching time, in the natural scale
# - u0_off_MCMC: Current values for u-coordinate of the off-switching point
# - s0_off_MCMC: Current values for s-coordinate of the off-switching point
# - TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, phi_MCMC, u_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC: Current values of model parameters
# - priorsLogT0_off: prior distribution for logarithm of the off-switching time
# - priorsT: prior distribution for time
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - mu_u_PROP, mu_s_PROP: Proposed means for unspliced/spliced counts distribution
# - propLog: Proposed values for logarithm of off-switching time
# - prop: Proposed values of the off-switching time, in the natural scale
# - SS_PROP: Proposed values of u-coordinate of lower steady state, s-coordinate of lower steady state, difference of the u-coordinate of steady states, beta, in the natural scale
# - u0_off_PROP, s0_off_PROP_ Proposed values for the off-switching point.
# - k_PROP, TStar_PROP, Tau_PROP, phi_PROP, u_PROP: Other values of the model that are modified as a consequence of the new proposed values for the off-switching time
# - prior_PROP: Prior density of the proposed switching time
# - priorTStar_PROP: Prior density for the proposed times (modified as a consequence of the new switching time)
# - loglik_PROP: Log-likelihood values of the proposed values
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities for switching time
# - priorTStar_MCMC: Current prior densities for times
# - loglik_MCMC: Current log-likelihood values.
# - acceptRate: Vector storing acceptance rate.
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - subtypeCell: Vector of cell subtype indices
# - typeCell:  Vector of cell type indices
# - typeCellT0_off:  Vector of switching clustering for all the cells
# - par2, par4: Intermediate parameter matrices used in likelihood.


    n_typeT0_off::Int64 = maximum(typeCellT0_off)
    
    # obtain the transcription and degradation rates
    alpha_off, alpha_on, gamma = ss_to_rates(SS_MCMC, model)
    
    Threads.@threads for g = 1:size(unspliced, 2) # iterates over genes        
        # initial points of the on branch
        u0_on_MCMC::Float64 = alpha_off[g]/SS_MCMC[4, g]
        s0_on_MCMC::Float64 = alpha_off[g]/gamma[g]

        for tyT0_off = 1:n_typeT0_off # iterates over switching clusters
            updateT0!(unspliced, spliced, alpha_off[g], alpha_on[g], gamma[g], SS_MCMC,  LogT0_off_MCMC, t0_off_MCMC,u0_off_MCMC, s0_off_MCMC, u0_on_MCMC, s0_on_MCMC, TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, phi_MCMC,u_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC, priorsLogT0_off, priorsT, adaptVar, mu_u_PROP, mu_s_PROP, propLog, prop, u0_off_PROP, s0_off_PROP, k_PROP, TStar_PROP, Tau_PROP, phi_PROP, u_PROP, prior_PROP, priorTStar_PROP, loglik_PROP, mu_u_MCMC, mu_s_MCMC, prior_MCMC, priorTStar_MCMC, loglik_MCMC, g, tyT0_off, acceptRate, mcmc, currIter, model, subtypeCell, typeCell, typeCellT0_off, par2, par4)             
        end
    end
end


function updateT0!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    alpha_off::Float64, 
    alpha_on::Float64, 
    gamma::Float64,
    SS_MCMC::Matrix{Float64},
    LogT0_off_MCMC::Matrix{Float64},
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    u0_on_MCMC::Float64,
    s0_on_MCMC::Float64,
    TStar_MCMC::Matrix{Float64},
    TStar_withM_MCMC::Matrix{Float64},
    Tau_MCMC::Matrix{Float64},
    phi_MCMC::Matrix{Float64},
    u_MCMC::Matrix{Float64},
    k_MCMC::Matrix{Int64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    priorsLogT0_off::Distribution{Univariate, Continuous},
    priorsT::Distribution{Univariate, Continuous},
    adaptVar::Matrix{<:adaptiveStep_1}, 
    mu_u_PROP::Matrix{Float64}, 
    mu_s_PROP::Matrix{Float64}, 
    propLog::Matrix{Float64},
    prop::Matrix{Float64},      
    u0_off_PROP::Matrix{Float64},
    s0_off_PROP::Matrix{Float64},
    k_PROP::Matrix{Int64},
    TStar_PROP::Matrix{Float64},
    Tau_PROP::Matrix{Float64},
    phi_PROP::Matrix{Float64}, 
    u_PROP::Matrix{Float64},
    prior_PROP::Matrix{Float64},
    priorTStar_PROP::Matrix{Float64},
    loglik_PROP::Matrix{Float64},
    mu_u_MCMC::Matrix{Float64}, 
    mu_s_MCMC::Matrix{Float64}, 
    prior_MCMC::Matrix{Float64},
    priorTStar_MCMC::Matrix{Float64},
    loglik_MCMC::Matrix{Float64},
    gene::Int64,
    tyT0_off::Int64,
    acceptRate::Matrix{Int64}, 
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}},
    currIter::Int64, 
    model::groupSubgroup,
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64},
    typeCellT0_off::Vector{Int64},
    par2::Matrix{Float64},
    par4::Matrix{Float64}
)
# MCMC update for the switching time parameters for s selected gene and a selected switching cluster. Note that phi will remain fixed for all the cells, so we will need to update the time parameters.
#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - alpha_off, alpha_on, gamma: current vales of the transcription rates and of the degradation rate for the specific gene
# - SS_MCMC: Current values for u-coordinate lower steady state, s-coordinate lower steady state, difference between the u-coordinates of the steady states, beta, in the natural scale
# - LogT0_off_MCMC: Current log-transformed values for off-switching time
# - t0_off_MCMC: Current values for off-switching time, in the natural scale
# - u0_off_MCMC: Current values for u-coordinate of the off-switching point
# - s0_off_MCMC: Current values for s-coordinate of the off-switching point
# - u0_on_MCMC, s0_on_MCMC: Current vales for initial points of on phase
# - TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, phi_MCMC, u_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC: Current values of model parameters
# - priorsLogT0_off: prior distribution for logarithm of the off-switching time
# - priorsT: prior distribution for time
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - mu_u_PROP, mu_s_PROP: Proposed means for unspliced/spliced counts distribution
# - propLog: Proposed values for logarithm of off-switching time
# - prop: Proposed values of the off-switching time, in the natural scale
# - u0_off_PROP, s0_off_PROP_ Proposed values for the off-switching point.
# - k_PROP, TStar_PROP, Tau_PROP, phi_PROP, u_PROP: other values of the model that are modified as a consequence of the new proposed values for the off-switching time
# - prior_PROP: Prior density of the proposed switching time
# - priorTStar_PROP: Prior density for the proposed times (modified as a consequence of the new switching time)
# - loglik_PROP: Log-likelihood values of the proposed values
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities for switching time
# - priorTStar_MCMC: Current prior densities for times
# - loglik_MCMC: Current log-likelihood values.
# - gene: index of the gene that we are considering
# - tyT0_off: index of the switching-cluster we are considering
# - acceptRate: Vector storing acceptance rate.
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - subtypeCell: Vector of cell subtype indices
# - typeCell:  Vector of cell type indices
# - typeCellT0_off:  Vector of switching clustering for all the cells
# - par2, par4: Intermediate parameter matrices used in likelihood.

    # obtain index of all the cells belonging to the switching cluster we are considering
    cellTyT0_off::Vector{Int64} = findall(typeCellT0_off.== tyT0_off)

    if size(cellTyT0_off, 1) > 0 # we have cells belonging to this switching-cluster 
        # ------------------
        # New proposed value
        # ------------------
        propLog[tyT0_off,gene] = proposal(LogT0_off_MCMC[tyT0_off, gene], gene, [tyT0_off], adaptVar) # logarithmic scale
        prop[tyT0_off, gene] = exp.(propLog[tyT0_off, gene]) # natural scale

        # check if the proposed switching time is exactly zero (if so, the value is rejected)
        reject::Bool = false
        if (prop[tyT0_off, gene] == 0.0)
            reject = true
        end

        MHalpha::Float64 = 0.0

        if !reject
            # update the coordinate of the switching point according to the new switching time
            u0_off_PROP[tyT0_off, gene], s0_off_PROP[tyT0_off, gene] = mu_tot_gc(prop[tyT0_off, gene], u0_off_PROP[tyT0_off, gene],  s0_off_PROP[tyT0_off, gene], u0_on_MCMC, s0_on_MCMC, Tau_PROP, k_PROP, gene, 1, catt_MCMC, alpha_off, alpha_on, gamma, true, model) 
            
            # update the times and states of the different cells according to the new switching time
            for sty = unique(subtypeCell[cellTyT0_off])
                subcellTy = findall(subtypeCell .== sty)
                TStar_PROP[subcellTy, gene] .= trasformPhi_toT(phi_MCMC, priorsT, SS_MCMC, prop, u0_off_PROP, gene, subcellTy, subtypeCell, typeCellT0_off, model)
            end
            off::Vector{Int64} = cellTyT0_off[findall(TStar_PROP[cellTyT0_off, gene] .> prop[tyT0_off, gene])]
            k_PROP[off, gene] .= 0
            on::Vector{Int64}  = cellTyT0_off[findall(TStar_PROP[cellTyT0_off, gene] .< prop[tyT0_off, gene])] 
            k_PROP[on, gene] .= 1
            Tau_PROP[off, gene] .= TStar_PROP[off, gene] .- prop[tyT0_off, gene]
            Tau_PROP[on, gene]  .= TStar_PROP[on, gene]

            # if the new switching point is to close to the lower steady state, reject the proposed value (the dynamic is not identifiable)
            if (abs(u0_off_PROP[tyT0_off, gene] - alpha_off/SS_MCMC[4, gene]) < 0.001) | (abs(s0_off_PROP[tyT0_off, gene] - alpha_off/gamma) < 0.001)
                MHalpha = 0.0
            else
                for c = cellTyT0_off
                    mu_u_PROP[c, gene], mu_s_PROP[c, gene] = mu_tot_gc(prop[tyT0_off, gene], u0_off_PROP[tyT0_off, gene], s0_off_PROP[tyT0_off, gene], u0_on_MCMC, s0_on_MCMC, Tau_PROP, k_PROP, gene, c, catt_MCMC, alpha_off, alpha_on, gamma, false, model) # update the mean of the spliced and unspliced distribution according to the new proposed value

                    u_PROP[c, gene] = mu_u_PROP[c, gene]/catt_MCMC[c]
        
                    loglik!(unspliced, spliced, mu_u_PROP, mu_s_PROP, invEta_MCMC, loglik_PROP, gene, c, par2, par4, model) # update the log-likelihood according to the new proposed value
                end    
        
                priorLogT0_off!(propLog, priorsLogT0_off, prior_PROP, gene, tyT0_off, model)     # update prior distribution according to the new proposed value

                numerator::Float64 = sum(loglik_PROP[cellTyT0_off,gene]) + prior_PROP[tyT0_off,gene]   # update numerator of MH rate

                
                # ------------------
                # Current value
                # ------------------
                denominator::Float64 = sum(loglik_MCMC[cellTyT0_off,gene]) + prior_MCMC[tyT0_off,gene]  # recompute denominator of MH rate

                # Compute MH rate
                MHalpha = min(1,exp(numerator - denominator)) 
            end
        end    

        # Decide if accept or not the new proposed value. In case of acceptance, update all the modified quantities    
        if (rand(Uniform(0.0,1.0)) < MHalpha)
            LogT0_off_MCMC[tyT0_off,gene] = propLog[tyT0_off,gene]
            t0_off_MCMC[tyT0_off, gene] = prop[tyT0_off, gene]
            u0_off_MCMC[tyT0_off, gene] = u0_off_PROP[tyT0_off, gene]
            s0_off_MCMC[tyT0_off, gene] = s0_off_PROP[tyT0_off, gene]

            TStar_MCMC[cellTyT0_off, gene] .= TStar_PROP[cellTyT0_off, gene]
            TStar_withM_MCMC[cellTyT0_off, gene] .= TStar_PROP[cellTyT0_off, gene]
            TStar_withM_MCMC[cellTyT0_off[findall(TStar_withM_MCMC[cellTyT0_off, gene] .< 0.0)], gene] .= 0.0
            Tau_MCMC[cellTyT0_off, gene] .= Tau_PROP[cellTyT0_off, gene]
            k_MCMC[cellTyT0_off, gene] .= k_PROP[cellTyT0_off, gene]
            u_MCMC[cellTyT0_off, gene] .= u_PROP[cellTyT0_off, gene]

            mu_u_MCMC[cellTyT0_off,gene] .= mu_u_PROP[cellTyT0_off,gene]
            mu_s_MCMC[cellTyT0_off,gene] .= mu_s_PROP[cellTyT0_off,gene]
            loglik_MCMC[cellTyT0_off,gene] .= loglik_PROP[cellTyT0_off,gene]

            prior_MCMC[tyT0_off,gene] = prior_PROP[tyT0_off,gene]
            
            acceptRate[tyT0_off,gene] = acceptRate[tyT0_off,gene] .+ 1    
        end        

        # ------------------
        # Update variance of proposal distribution for adaptive MCMC
        # ------------------
        updateAdapt!(adaptVar, gene, [tyT0_off], mcmc, currIter, MHalpha)    
    end
end