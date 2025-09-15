# propose phi, keep fixed t0 --> need to change u0_on, s0_on, u0_off, s0_off, t, u, mu_u, mu_s
function updateTStar!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    SS_MCMC::Matrix{Float64},
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    phi_MCMC::Matrix{Float64},
    u_MCMC::Matrix{Float64},
    TStar_MCMC::Matrix{Float64},
    TStar_withM_MCMC::Matrix{Float64},
    Tau_MCMC::Matrix{Float64},
    k_MCMC::Matrix{Int64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    LogitCatt_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    adaptVar::Matrix{<:adaptiveStep_1}, 
    priorsT::Distribution{Univariate, Continuous},
    mu_u_PROP::Matrix{Float64}, 
    mu_s_PROP::Matrix{Float64},
    prop::Matrix{Float64},     
    TStar_PROP::Matrix{Float64},
    Tau_PROP::Matrix{Float64},
    k_PROP::Matrix{Int64},
    prior_PROP::Matrix{Float64}, 
    loglik_PROP::Matrix{Float64}, 
    mu_u_MCMC::Matrix{Float64}, 
    mu_s_MCMC::Matrix{Float64}, 
    prior_MCMC::Matrix{Float64},
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

# MCMC update for the switching time parameters for all the genes and all the subgroups.
# This function is a wrapper around the single-gene and subgroup updateTStar! function, iterating over all genes and subgroups and launching their updates in parallel. 

#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - SS_MCMC: Current values for u-coordinate lower steady state, s-coordinate lower steady state, difference between the u-coordinates of the steady states, beta, in the natural scale
# - t0_off_MCMC: Current values for off-switching time, in the natural scale
# - u0_off_MCMC: Current values for u-coordinate of the off-switching point
# - s0_off_MCMC: Current values for s-coordinate of the off-switching point
# - phi_MCMC: Current angular coordinate associated with time for each gene and subgroup
# - u_MCMC: Current u-coordinate associated with time for each gene and subgroup
# - TStar_MCMC: Current time in the dynamic  for each gene and subgroup
# - TStar_withM_MCMC: Current time in the dynamic, with mass in the lower steady state, for each gene and subgroup
# - Tau_MCMC: Current time since the last switching point fo each gene and subgroup
# - k_MCMC: Current phase for each gene and subgroup (on-off)
# - LogEta_MCMC, invEta_MCMC,  LogitCatt_MCMC, catt_MCMC: Current values of model parameters
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - priorsT: prior distribution for time
# - mu_u_PROP, mu_s_PROP: Proposed means for unspliced/spliced counts distribution
# - prop: Proposed values of the time with mass, in the natural scale
# - TStar_PROP, Tau_PROP, k_PROP: Other time-related parameters of the model that are modified as a consequence of the new proposed values for the time with mass
# - prior_PROP: Prior density of the proposed time
# - loglik_PROP: Log-likelihood values of the proposed values
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities for time
# - loglik_MCMC: Current log-likelihood values.
# - acceptRate: Vector storing acceptance rate.
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - subtypeCell: Vector of cell subtype indices
# - typeCell:  Vector of cell type indices
# - typeCellT0_off:  Vector of switching clustering for all the cells
# - par2, par4: Intermediate parameter matrices used in likelihood.


    n_subtypeC::Int64 = maximum(subtypeCell)
    # obtain the transcription and degradation rates
    alpha_off::Vector{Float64}, alpha_on::Vector{Float64}, gamma::Vector{Float64} = ss_to_rates(SS_MCMC, model)

    Threads.@threads for g = 1:size(unspliced, 2) # iterates over genes        
        # initial points of the on branch
        u0_on_MCMC::Float64 = alpha_off[g]/1.0
        s0_on_MCMC::Float64 = alpha_off[g]/gamma[g]

        for sty = 1:n_subtypeC  # iterates over subgroups
            updateTStar!(unspliced, spliced, SS_MCMC, alpha_off[g], alpha_on[g], gamma[g], t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, u0_on_MCMC, s0_on_MCMC, phi_MCMC, u_MCMC, TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC, LogitCatt_MCMC, catt_MCMC, adaptVar, priorsT, mu_u_PROP, mu_s_PROP, prop, TStar_PROP, Tau_PROP, k_PROP, prior_PROP, loglik_PROP, g, sty, mu_u_MCMC, mu_s_MCMC, prior_MCMC, loglik_MCMC, acceptRate, mcmc, currIter, model, subtypeCell, typeCell, typeCellT0_off, par2, par4)
        end
    end
end


function updateTStar!(
    unspliced::Matrix{Int64}, 
    spliced::Matrix{Int64},
    SS_MCMC::Matrix{Float64},
    alpha_off::Float64, 
    alpha_on::Float64, 
    gamma::Float64,
    t0_off_MCMC::Matrix{Float64},
    u0_off_MCMC::Matrix{Float64},
    s0_off_MCMC::Matrix{Float64},
    u0_on_MCMC::Float64, 
    s0_on_MCMC::Float64, 
    phi_MCMC::Matrix{Float64},
    u_MCMC::Matrix{Float64},
    TStar_MCMC::Matrix{Float64},
    TStar_withM_MCMC::Matrix{Float64},
    Tau_MCMC::Matrix{Float64},
    k_MCMC::Matrix{Int64},
    LogEta_MCMC::Vector{Float64},
    invEta_MCMC::Vector{Float64},
    LogitCatt_MCMC::Vector{Float64},
    catt_MCMC::Vector{Float64},
    # varianza adattiva
    adaptVar::Matrix{<:adaptiveStep_1}, # così prende qualsiasi sottotipo di adaptiveStep_1
    priorsT::Distribution{Univariate, Continuous},
    mu_u_PROP::Matrix{Float64}, # mu_u nuova (da riempire)
    mu_s_PROP::Matrix{Float64}, # mu_s nuova (da riempire)
    prop::Matrix{Float64},      # matrice dove inserire la proposta
    TStar_PROP::Matrix{Float64},
    Tau_PROP::Matrix{Float64},
    k_PROP::Matrix{Int64},
    prior_PROP::Matrix{Float64},  # matrice dove inserire la prior della proposta
    loglik_PROP::Matrix{Float64}, # matrice dove inserire i valori della logLik della proposta
    gene::Int64,
    sty::Int64,
    mu_u_MCMC::Matrix{Float64}, # mu_u vecchia
    mu_s_MCMC::Matrix{Float64}, # mu_s vecchia
    prior_MCMC::Matrix{Float64},
    loglik_MCMC::Matrix{Float64}, 
    acceptRate::Matrix{Int64}, # valori correnti della loglik
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}},
    currIter::Int64, # iterazione corrente;
    model::groupSubgroup, # questa è già solo per il tempo condiviso
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64},    
    typeCellT0_off::Vector{Int64},   
    par2::Matrix{Float64},
    par4::Matrix{Float64}
)


# MCMC update for the switching time parameters for a specific gene and subtype. 
#
# Arguments
# ------
# - unspliced: unspliced counts matrix
# - spliced: spliced counts matrix
# - SS_MCMC: Current values for u-coordinate lower steady state, s-coordinate lower steady state, difference between the u-coordinates of the steady states, beta, in the natural scale
# - alpha_off, alpha_on, gamma: current vales of the transcription rates and of the degradation rate for the specific gene
# - t0_off_MCMC: Current values for off-switching time, in the natural scale
# - u0_off_MCMC: Current values for u-coordinate of the off-switching point
# - s0_off_MCMC: Current values for s-coordinate of the off-switching point
# - u0_on_MCMC, s0_on_MCMC: Current vales for initial points of on phase


# - phi_MCMC: Current angular coordinate associated with time for each gene and subgroup
# - u_MCMC: Current u-coordinate associated with time for each gene and subgroup
# - TStar_MCMC: Current time in the dynamic  for each gene and subgroup
# - TStar_withM_MCMC: Current time in the dynamic, with mass in the lower steady state, for each gene and subgroup
# - Tau_MCMC: Current time since the last switching point fo each gene and subgroup
# - k_MCMC: Current phase for each gene and subgroup (on-off)
# - LogEta_MCMC, invEta_MCMC,  LogitCatt_MCMC, catt_MCMC: Current values of model parameters
# - adaptVar: Adaptive variance structure for the proposal mechanism
# - priorsT: prior distribution for time
# - mu_u_PROP, mu_s_PROP: Proposed means for unspliced/spliced counts distribution
# - prop: Proposed values of the time with mass, in the natural scale
# - TStar_PROP, Tau_PROP, k_PROP: Other time-related parameters of the model that are modified as a consequence of the new proposed values for the time with mass
# - prior_PROP: Prior density of the proposed time
# - loglik_PROP: Log-likelihood values of the proposed values
# - gene: index of the gene we are considering
# - sty: index of the subgroups we are considering
# - mu_u_MCMC, mu_s_MCMC: Current means for unspliced/spliced counts distribution.
# - prior_MCMC: Current prior densities for time
# - loglik_MCMC: Current log-likelihood values.
# - acceptRate: Vector storing acceptance rate.
# - mcmc: named tuple with MCMC settings.
# - currIter: Current MCMC iteration.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - subtypeCell: Vector of cell subtype indices
# - typeCell:  Vector of cell type indices
# - typeCellT0_off:  Vector of switching clustering for all the cells
# - par2, par4: Intermediate parameter matrices used in likelihood.


    # obtain index of all the cells belonging to the subgroup we are considering
    subcellTy::Vector{Int64} = findall(subtypeCell.== sty)

    if size(subcellTy, 1) > 0 #  we have cells belonging to this subgroup
        # ------------------
        # New proposed value
        # ------------------
        prop[subcellTy,gene] .= proposal(phi_MCMC[subcellTy[1], gene], gene, subcellTy, adaptVar)  # value of phi
        prop[subcellTy,gene] .= mod(prop[subcellTy[1], gene], 1.0) # normale arrotolata

        TStar_PROP[subcellTy, gene] .= trasformPhi_toT(prop, priorsT, SS_MCMC, t0_off_MCMC, u0_off_MCMC, gene, subcellTy, subtypeCell, typeCellT0_off, model) # transform phi back to the time

        for c = subcellTy
            trasform_TStar!(TStar_PROP, TStar_PROP, gene, c, model) # put mass in the lower steady state if necessary
        end
        

        # update the time associated parameters
        Tau_PROP[subcellTy, gene] .= TStar_PROP[subcellTy, gene]
        k_PROP[subcellTy, gene] .= 1
        tyT0_off::Int64 = typeCellT0_off[subcellTy[1]]
        if (TStar_PROP[subcellTy[1], gene] > t0_off_MCMC[tyT0_off, gene])
            k_PROP[subcellTy, gene] .= 0
            Tau_PROP[subcellTy, gene] .= TStar_PROP[subcellTy[1], gene] .- t0_off_MCMC[tyT0_off, gene]
        end
        
        for c = subcellTy
            mu_u_PROP[c, gene], mu_s_PROP[c, gene] = mu_tot_gc(t0_off_MCMC[tyT0_off, gene], u0_off_MCMC[tyT0_off, gene], s0_off_MCMC[tyT0_off, gene], u0_on_MCMC, s0_on_MCMC, Tau_PROP, k_PROP, gene, c, catt_MCMC, alpha_off, alpha_on, gamma, false, model)             # update the mean of the spliced and unspliced distribution according to the new proposed value
            
            loglik!(unspliced, spliced, mu_u_PROP, mu_s_PROP, invEta_MCMC, loglik_PROP, gene, c, par2, par4, model)  # update the log-likelihood according to the new proposed value
                
        end    
    
        priorTStar!(prop, priorsT, prior_PROP, gene, subcellTy, model)  # update prior distribution according to the new proposed value
    
        numerator::Float64 = sum(loglik_PROP[subcellTy,gene]) + prior_PROP[subcellTy[1],gene] # update numerator of MH rate 


        # ------------------
        # Current value
        # ------------------
        denominator::Float64 = sum(loglik_MCMC[subcellTy,gene]) + prior_MCMC[subcellTy[1],gene] # recompute denominator of MH rate
                
        # Compute MH rate
        MHalpha = min(1,exp((numerator - denominator)))

        # Decide if accept or not the new proposed value. In case of acceptance, update all the modified quantities    
        if (rand(Uniform(0.0,1.0))< MHalpha)
            phi_MCMC[subcellTy, gene] .= prop[subcellTy[1], gene]
            TStar_MCMC[subcellTy, gene] .= TStar_PROP[subcellTy[1], gene]
            TStar_withM_MCMC[subcellTy, gene] .= TStar_MCMC[subcellTy[1], gene]
        
            k_MCMC[subcellTy, gene] .= k_PROP[subcellTy, gene]
            Tau_MCMC[subcellTy, gene] .= Tau_PROP[subcellTy, gene]

            mu_u_MCMC[subcellTy,gene] .= mu_u_PROP[subcellTy,gene]
            u_MCMC[subcellTy, gene] .= mu_u_MCMC[subcellTy, gene]./catt_MCMC[subcellTy]
            mu_s_MCMC[subcellTy,gene] .= mu_s_PROP[subcellTy,gene]
            loglik_MCMC[subcellTy,gene] .= loglik_PROP[subcellTy,gene]
            prior_MCMC[subcellTy,gene] .= prior_PROP[subcellTy,gene]

            acceptRate[subcellTy,gene] = acceptRate[subcellTy,gene] .+ 1    

        end
        
        # ------------------
        # Update variance of proposal distribution for adaptive MCMC
        # ------------------
        updateAdapt!(adaptVar, gene, subcellTy, mcmc, currIter, MHalpha)    
    end
end





