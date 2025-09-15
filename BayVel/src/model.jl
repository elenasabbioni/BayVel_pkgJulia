function MCMC(
    model::groupSubgroup,
    unspliced::Matrix{Int64},
    spliced::Matrix{Int64},
    typeCell::Vector{Int64};
    mcmc::NamedTuple{(:iter, :thin, :burnin), Tuple{Int64, Int64, Int64}} = (iter = 10000, thin = 2, burnin = 2),
    priorsLogU_SS_on::Distribution{Univariate, Continuous}, 
    priorsLogS_SS_on::Distribution{Univariate, Continuous} ,
    priorsLogU_SS_off::Distribution{Univariate, Continuous},
    priorsLogBeta::Distribution{Univariate, Continuous} = BayVel.Log_Uniform(0.0, 1.0) ,
    priorsLogT0_off::Distribution{Univariate, Continuous},
    priorsT::Distribution{Univariate, Continuous},
    priorsEta::Distribution{Univariate, Continuous} = TruncatedNormal(0.0, 10000.0, 0.0, Inf),
    priorsCatt::Distribution{Univariate, Continuous} = Uniform(0.0::Float64,1.0::Float64),
    initUoff::Vector{Float64},
    initSoff::Vector{Float64},
    initDiffU::Vector{Float64},
    initBeta::Vector{Float64},
    initT0_off::Matrix{Float64},
    initTStar::Matrix{Float64},
    initTau::Matrix{Float64},
    initEta::Vector{Float64},
    initCatt::Vector{Float64},
    simSS::Bool,
    simT0::Bool,
    simTau::Bool,
    simEta::Bool,
    simCatt::Bool,                                                                                                        
    alphaTarget::Float64 = 0.25,
    stepVect::Vector{Float64} = [30.0, 400.0],
    typeCellT0_off::Vector{Int64},
    subtypeCell::Vector{Int64},
)
# Perform MCMC sampling for the statistical model of BayVel
#
# Arguments
# - model:                 Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# - unspliced:             Matrix of unspliced counts (cells × genes)
# - spliced:               Matrix of spliced counts (cells × genes)
# - typeCell:              Vector of cell type indices
# - mcmc:                  NamedTuple with fields iter, thin, burnin (default: (10000, 2, 2))
# - priorsLogU_SS_on:      Prior distribution for logarithm of the u-coordinate of the upper steady-state
# - priorsLogS_SS_on:      Prior distribution for logarithm of the s-coordinate of the upper steady-state
# - priorsLogU_SS_off:     Prior distribution for logarithm of the u-coordinate of the lower steady-state
# - priorsLogBeta:         Prior distribution for logarithm of the splicing rate (not used for the moment, added for generality for future modifications)
# - priorsLogT0_off:       Prior distribution for logarithm of the off-switching point
# - priorsT:               Prior distribution for time / phi
# - priorsEta:             Prior for overdispersion (eta) (default: TruncatedNormal(0,10000,0,Inf))
# - priorsCatt:            Prior for cell-specific capture efficiency (default: Uniform(0,1))
# - initUoff, initSoff, initDiffU, initBeta: Initial values for u-coordinate of the lower steady state, s-coordinate of the lower steady state, difference of the u-coordinates of the steady states, splicing rate beta 
# - initT0_off, initTStar, initTau: Initial matrices of switching times, time on the dynamic and time elaplsed since the last switching
# - initEta, initCatt:     Initial vectors for overdispersion and capture efficiency
# - simSS, simT0, simTau, simEta, simCatt: Bools controlling which parameters are updated
# - alphaTarget:           Target acceptance rate for adaptive MCMC (default 0.25)
# - stepVect:              Adaptation parameters (default [30.0, 400.0])
# - typeCellT0_off:        Vector of switching clustering for all the cells
# - subtypeCell:           Vector of cell subtype indices
#
# Returns
# Tuple containing posterior chain of MCMC:
# - LogSS_out, SS_Star_out,                                 # steady-state chains
# - LogT0_off_out, T0_off_out                               # switching time chains
# - phi_out, TStar_out, TStar_withM_out, Tau_out, k_out     # time-related chains
# - LogEta_out, LogitCatt_out                               # overdispersion and capture efficiency
# - Acceptance rates for SS, T0_off, TStar, Eta, Catt       # acceptance rates
#
# Notes
# - This function implements an adaptive Metropolis-within-Gibbs scheme.
# - Updates for each block (SS, T0_off, TStar, Eta, Catt) can be enabled/disabled using the boolean flags.
# - Tracks acceptance rates for diagnostic purposes.
# - Supports multiple cell types and subtypes.
#

    # -----------------------
    # Dimensions and setup
    # -----------------------
    n_genes::Int64 = size(unspliced,2)                    # number of genes
    n_cells::Int64 = size(unspliced,1)                    # number of cells
    n_typeC::Int64 = maximum(typeCell)                    # number of groups
    n_subtypeC::Int64 = maximum(subtypeCell)              # number of subgroups
    n_typeT0_off::Int64 = maximum(typeCellT0_off)         # number of switching clusters



    #--------- MCMC Parameters
    SampleToSave::Int64 = Int64(trunc((mcmc.iter-mcmc.burnin)/mcmc.thin)) # Compute number of posterior samples after burn-in and thinning
    thinburnin = mcmc.burnins
    iter = Int64(1)

    p2 = Progress(mcmc.burnin +(SampleToSave-1)*mcmc.thin, desc="iterations ", offset=0,showspeed=true) # Progress bar
                
    println("Iterations: ",mcmc.iter)
    println("burnin: ",mcmc.burnin)
    println("thin: ",mcmc.thin)
    println("number of posterior samples: ",SampleToSave)
    println("Number of threads: ",Threads.nthreads())

    # -----------------------
    # Initialize matrices for current MCMC step
    # -----------------------
    # --- steady-states
    LogSS_MCMC::Matrix{Float64} = Matrix{Float64}(undef, 4, n_genes)                # log(u_ss_off, s_ss_off, diffU, beta)
    SS_MCMC::Matrix{Float64} = Matrix{Float64}(undef, 4, n_genes)                   # u_ss_off, s_ss_off, diffU, beta
    LogSS_Star_MCMC::Matrix{Float64} = Matrix{Float64}(undef, 4, n_genes)           # log(u_ss_on, s_ss_on, u_ss_off, beta)
    SS_Star_MCMC::Matrix{Float64} = Matrix{Float64}(undef, 4, n_genes)              # u_ss_on, s_ss_on, u_ss_off, beta

    # --- switching parameters
    LogT0_off_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)
    t0_off_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)
    u0_off_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)
    s0_off_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)

    # --- elapsed time parameters
    u_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes)              # mu_u_MCMC/catt_MCMC
    phi_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes)            # transformation parameter phi
    TStar_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes)
    TStar_withM_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes)
    Tau_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes)
    k_MCMC::Matrix{Int64} = Matrix{Int64}(undef, n_cells, n_genes)

    # --- overdispersion
    LogEta_MCMC::Vector{Float64} = Vector{Float64}(undef, n_genes)
    invEta_MCMC::Vector{Float64} = Vector{Float64}(undef, n_genes)
    # --- capture efficiency
    LogitCatt_MCMC::Vector{Float64} = Vector{Float64}(undef, n_cells)
    catt_MCMC::Vector{Float64} = Vector{Float64}(undef, n_cells)

    # --- mean for unspliced and spliced counts
    mu_u_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes) # u_MCMC * catt_MCMC
    mu_s_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes) # s_MCMC * catt_MCMC
    # --- log-likelihood
    loglik_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes) 
    par2::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes) # parameters of the Negative Binomial distribution
    par4::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes) # parameters of the Negative Binomial distribution

    # -----------------------
    # Initialization atrices for priors
    # -----------------------
    priorLogSS_Star_MCMC::Vector{Float64} = Vector{Float64}(undef, n_genes)
    priorLogT0_off_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)
    priorTStar_MCMC::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes)
    priorETA_MCMC::Vector{Float64} = Vector{Float64}(undef, n_genes)
    priorCATT_MCMC::Vector{Float64} = Vector{Float64}(undef, n_cells)

    # -----------------------
    # Filling matrices with current MCMC values
    # -----------------------
    # --- steady-state values
    LogSS_MCMC[1,:] .= log.(initUoff[:])
    LogSS_MCMC[2,:] .= log.(initSoff[:])
    LogSS_MCMC[3,:] .= log.(initDiffU[:])
    LogSS_MCMC[4,:] .= log.(initBeta[:])
    SS_MCMC[:,:] .= exp.(LogSS_MCMC) 
    SS_Star_MCMC[1, :], SS_Star_MCMC[2, :], SS_Star_MCMC[3, :]  = ss_to_ssStar(SS_MCMC, model)
    SS_Star_MCMC[4, :] .= SS_MCMC[4,:]
    LogSS_Star_MCMC[:, :] .= log.(SS_Star_MCMC[:, :])

    # --- switching time
    LogT0_off_MCMC[:,:] .= log.(initT0_off[:,:])
    t0_off_MCMC[:, :] .= initT0_off[:, :]

    # --- time's parameters
    TStar_MCMC[:, :] .= initTStar[:,:]
    TStar_withM_MCMC[:, :] .= TStar_MCMC[:,:] 
    for c = 1:n_cells
        for g = 1:n_genes
            if TStar_MCMC[c, g] < 0.0
                TStar_withM_MCMC[c, g] = 0.0
            end
        end
    end
    Tau_MCMC[:,:] = TStar_withM_MCMC[:,:]
    for tyT0_off = 1:n_typeT0_off
        cellTyT0_off::Vector{Int64} = findall(typeCellT0_off .== tyT0_off)
        for gene = 1:n_genes
            off::Vector{Int64} = cellTyT0_off[findall(TStar_withM_MCMC[cellTyT0_off, gene] .> t0_off_MCMC[tyT0_off, gene])]
            k_MCMC[off, gene] .= 0
            Tau_MCMC[off, gene] .= TStar_withM_MCMC[off, gene] .- t0_off_MCMC[tyT0_off, gene]
            on::Vector{Int64} = cellTyT0_off[findall(TStar_withM_MCMC[cellTyT0_off, gene] .< t0_off_MCMC[tyT0_off, gene])] 
            k_MCMC[on, gene] .= 1
        end
    end

    # --- overdispersion
    LogEta_MCMC[:] .= log.(initEta[:])
    invEta_MCMC[:] .= exp.(.- LogEta_MCMC[:])

    # --- capture efficiency
    LogitCatt_MCMC[:] .= logit.(initCatt[:])
    catt_MCMC[:] .= initCatt[:]

    for g = 1:n_genes
        # --- coordinates of the off-switching point s0_off e u0_off  
        mu_tot_g!(SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, Tau_MCMC, k_MCMC, mu_u_MCMC, mu_s_MCMC, g, catt_MCMC, true, typeCellT0_off, model)
        
        # --- mean of the distribution of unspliced and spliced counts for all cells (u(t,...)*catt and s(t, ...)*catt)
        mu_tot_g!(SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, Tau_MCMC, k_MCMC, mu_u_MCMC, mu_s_MCMC, g, catt_MCMC, false, typeCellT0_off, model)

        # --- log-likelihood
        loglik_g!(unspliced, spliced, mu_u_MCMC, mu_s_MCMC, invEta_MCMC, loglik_MCMC, g, par2, par4, model)

        # --- prior for logarithm of steady-states
        priorLogSS_Star_MCMC[g] = priorLogSS_Star(LogSS_Star_MCMC, priorsLogU_SS_on, priorsLogS_SS_on, priorsLogU_SS_off, priorsLogBeta, g, model)
        
        # --- prior for logarithm of the switching time
        priorLogT0_off!(LogT0_off_MCMC, priorsLogT0_off, priorLogT0_off_MCMC, g, typeCellT0_off, model)  
        
        # --- prior of the time and phi
        for cell = 1:n_cells
            u_MCMC[cell, g] = mu_u_MCMC[cell, g]/catt_MCMC[cell]
        end
        for sty = 1:n_subtypeC
            subcellTy::Vector{Int64} = findall(subtypeCell .== sty)
            phi_MCMC[subcellTy, g] .= trasformU_toPhi(u_MCMC, TStar_MCMC, priorsT, SS_MCMC, t0_off_MCMC, u0_off_MCMC, g, subcellTy, subtypeCell, typeCellT0_off)
        end
        priorTStar!(phi_MCMC, priorsT,  priorTStar_MCMC, g, subtypeCell, typeCell, model)

        # --- prior overdispersion
        priorETA_MCMC[g] = priorETA(LogEta_MCMC, priorsEta, g, model)

    end
  
    # --- prior capture efficiency
    for c = 1:n_cells
        priorCATT_MCMC[c] = priorCATT(LogitCatt_MCMC, priorsCatt, c, model)
    end

    # Assertions to catch NaN issues
    @toggled_assert all(!isnan, LogSS_MCMC) "NaN in LogSS_MCMC"
    @toggled_assert all(!isnan, TStar_MCMC) "NaN in TStar_MCMC"
    @toggled_assert all(!isnan, k_MCMC) "NaN in k_MCMC"
    @toggled_assert all(!isnan, LogEta_MCMC) "NaN in LogEta_MCMC"
    @toggled_assert all(!isnan, LogitCatt_MCMC) "NaN in LogitCatt_MCMC"
    @toggled_assert all(!isnan, mu_u_MCMC) "NaN in mu_u_MCMC"
    @toggled_assert all(!isnan, mu_s_MCMC) "NaN in mu_s_MCMC"

    # -----------------------
    # Initialize matrices for values proposed in the MCMC step
    # -----------------------
    mu_u_PROP::Matrix{Float64} = deepcopy(mu_u_MCMC)
    mu_s_PROP::Matrix{Float64} = deepcopy(mu_s_MCMC)
    loglik_PROP::Matrix{Float64} = deepcopy(loglik_MCMC)

    # --- steady-states
    LogSS_PROP::Matrix{Float64} = zeros(4, n_genes)
    SS_PROP::Matrix{Float64} = zeros(4, n_genes)
    LogSS_Star_PROP::Matrix{Float64} = zeros(4, n_genes)
    SS_Star_PROP::Matrix{Float64} = zeros(4, n_genes)
    LogSS_PROP[4,:] .= log.(initBeta[:])
    LogSS_Star_PROP[4,:] .= log.(initBeta[:])
    SS_Star_PROP[4,:] .= initBeta[:]
    SS_PROP[4,:] .= initBeta[:]
    priorLogSS_Star_PROP::Vector{Float64} = Vector{Float64}(undef, n_genes)

    # --- switching times
    LogT0_off_PROP::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)
    t0_off_PROP::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)
    priorLogT0_off_PROP::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)
    u0_off_PROP::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)
    s0_off_PROP::Matrix{Float64} = Matrix{Float64}(undef, n_typeT0_off, n_genes)

    # --- times
    u_PROP::Matrix{Float64} = deepcopy(u_MCMC)
    phi_PROP::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes)
    TStar_PROP::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes)
    Tau_PROP::Matrix{Float64} = deepcopy(Tau_MCMC)
    k_PROP::Matrix{Int64} = deepcopy(k_MCMC)
    priorTStar_PROP::Matrix{Float64} = Matrix{Float64}(undef, n_cells, n_genes)

    # --- overdispersion
    priorETA_PROP::Vector{Float64} = Vector{Float64}(undef, n_genes)
    eta_PROP::Vector{Float64} = Vector{Float64}(undef, n_genes)

    # --- capture efficiency
    priorCATT_PROP::Vector{Float64} = Vector{Float64}(undef, n_cells)
    catt_PROP::Vector{Float64} = Vector{Float64}(undef, n_cells)
    LogitCatt_PROP::Vector{Float64} = Vector{Float64}(undef, n_cells)


    # -------------------------------
    # Acceptance rate 
    # ------------------------------- 
    acceptRateSS::Vector{Int64} = [0 for g = 1:n_genes]
    acceptRateT0_off::Matrix{Int64} = zeros(n_typeT0_off, n_genes)
    acceptRateTStar::Matrix{Int64} = zeros(n_cells, n_genes)
    acceptRateEta::Vector{Int64} = [0 for g = 1:n_genes]
    acceptRateCatt::Vector{Int64} = [0 for c in 1:n_cells]

    # -------------------------------
    # Output arrays for posterior samples
    # -------------------------------
    SS_Star_out::Array{Float64, 3} = Array{Float64, 3}(undef, 4, n_genes, SampleToSave)
    LogSS_out::Array{Float64, 3} = Array{Float64, 3}(undef, 4, n_genes, SampleToSave)
    LogT0_off_out::Array{Float64, 3} = Array{Float64, 3}(undef, n_typeT0_off, n_genes, SampleToSave)
    T0_off_out::Array{Float64, 3} = Array{Float64, 3}(undef, n_typeT0_off, n_genes, SampleToSave)
    phi_out::Array{Float64, 3} = Array{Float64, 3}(undef, n_subtypeC, n_genes, SampleToSave)
    TStar_out::Array{Float64, 3} = Array{Float64, 3}(undef, n_subtypeC, n_genes, SampleToSave)
    TStar_withM_out::Array{Float64, 3} = Array{Float64, 3}(undef, n_subtypeC, n_genes, SampleToSave)
    Tau_out::Array{Float64, 3} = Array{Float64, 3}(undef, n_subtypeC, n_genes, SampleToSave)
    k_out::Array{Int64, 3} = Array{Int64, 3}(undef, n_subtypeC, n_genes, SampleToSave)
    LogEta_out::Array{Float64, 2} = Array{Float64, 2}(undef, n_genes, SampleToSave)
    LogitCatt_out::Array{Float64, 2} = Array{Float64, 2}(undef, n_cells, SampleToSave)

    # -------------------------------
    # Adaptive variance
    # -------------------------------
    adaptiveSS = [adaptiveAlg4(alphaTarget, [0.1, 0.1, 0.1, 0.1], LogSS_Star_MCMC, stepVect, [false], [true], 3; gene = g) for g = 1:n_genes]
    adaptiveT0_off::Matrix{adaptiveUniv_1} = Matrix{adaptiveUniv_1}(undef, n_typeT0_off, n_genes)
    adaptiveTStar::Matrix{adaptiveUniv_1} = Matrix{adaptiveUniv_1}(undef, n_cells, n_genes)
    for g = 1:n_genes
        for c = 1:n_cells
            adaptiveTStar[c, g] = adaptiveUniv_1(alphaTarget, 0.1, 100, stepVect, [false], [true])
        end
        for ty = 1:n_typeT0_off
            adaptiveT0_off[ty, g] = adaptiveUniv_1(alphaTarget, 0.1, 100, stepVect, [false], [true])
        end
    end
    adaptiveETA = [adaptiveUniv_1(alphaTarget, 0.1, 100, stepVect, [false], [true]) for g = 1:n_genes]
    adaptiveCATT = [adaptiveUniv_1(alphaTarget, 0.1, 100, stepVect, [false], [true]) for c = 1:n_cells]


    # -------------------------------
    # START OF MCMC - main MCMC loop
    # -------------------------------
    println("inizio MCMC")
    for iMCMC = 1:SampleToSave       
        for _ = 1:thinburnin
            iter += 1
            ProgressMeter.next!(p2; showvalues = [(:iterationstot,mcmc.iter), (:iterations,iter)])
            
            # --- update steady-states' coordinates and associated quantities
            if simSS
                updateSS!(unspliced, spliced, LogSS_MCMC, SS_MCMC, LogSS_Star_MCMC, SS_Star_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, k_MCMC, u_MCMC, phi_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC, priorsLogU_SS_on, priorsLogS_SS_on, priorsLogU_SS_off, priorsLogBeta, priorsT, adaptiveSS, mu_u_PROP, mu_s_PROP, LogSS_Star_PROP, SS_Star_PROP, SS_PROP, u0_off_PROP, s0_off_PROP, TStar_PROP, Tau_PROP, k_PROP, u_PROP, phi_PROP, priorLogSS_Star_PROP, priorTStar_PROP, loglik_PROP, mu_u_MCMC, mu_s_MCMC, priorLogSS_Star_MCMC, priorTStar_MCMC, loglik_MCMC, acceptRateSS, mcmc, iter, model, subtypeCell, typeCell, typeCellT0_off, par2, par4)
            end
        
            # --- update switching times and associated quantities
            if simT0
                updateT0!(unspliced, spliced, SS_MCMC, LogT0_off_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, phi_MCMC, u_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC, priorsLogT0_off, priorsT, adaptiveT0_off, mu_u_PROP, mu_s_PROP, LogT0_off_PROP, t0_off_PROP, u0_off_PROP, s0_off_PROP, k_PROP, TStar_PROP, Tau_PROP, phi_PROP, u_PROP, priorLogT0_off_PROP, priorTStar_PROP, loglik_PROP, mu_u_MCMC, mu_s_MCMC, priorLogT0_off_MCMC, priorTStar_MCMC, loglik_MCMC, acceptRateT0_off, mcmc, iter, model, subtypeCell, typeCell, typeCellT0_off, par2, par4)
            end
            
            # --- update time and associated quantities
            if simTau
                updateTStar!(unspliced, spliced, SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, phi_MCMC, u_MCMC, TStar_MCMC, TStar_withM_MCMC, Tau_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC, LogitCatt_MCMC, catt_MCMC, adaptiveTStar, priorsT, mu_u_PROP, mu_s_PROP, phi_PROP, TStar_PROP, Tau_PROP, k_PROP, priorTStar_PROP, loglik_PROP, mu_u_MCMC, mu_s_MCMC, priorTStar_MCMC, loglik_MCMC, acceptRateTStar, mcmc, iter, model, subtypeCell, typeCell, typeCellT0_off, par2, par4)
            end
            
            # --- update overdispersion and associated quantities
            if simEta
                updateETA!(unspliced, spliced, SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, LogEta_MCMC, invEta_MCMC, catt_MCMC, priorsEta, adaptiveETA, eta_PROP, priorETA_PROP, loglik_PROP, mu_u_MCMC, mu_s_MCMC, priorETA_MCMC, loglik_MCMC, acceptRateEta, mcmc, iter, model, par2, par4)
            end

            # update capture efficiency and associated quantities
            if simCatt
                updateCATT!(unspliced, spliced, SS_MCMC, t0_off_MCMC, u0_off_MCMC, s0_off_MCMC, TStar_MCMC, Tau_MCMC, k_MCMC, LogEta_MCMC, invEta_MCMC, LogitCatt_MCMC, catt_MCMC, priorsCatt, adaptiveCATT, mu_u_PROP, mu_s_PROP, LogitCatt_PROP, catt_PROP, priorCATT_PROP, loglik_PROP, mu_u_MCMC, mu_s_MCMC, priorCATT_MCMC, loglik_MCMC, acceptRateCatt, mcmc, iter, model, typeCell, typeCellT0_off, par2, par4)
            end
        end

        thinburnin = mcmc.thin
        
        # Save samples
        SS_Star_out[:, :, iMCMC] = SS_Star_MCMC[:, :]
        LogSS_out[:,:,iMCMC] = LogSS_MCMC[:,:]
        for sty = 1:n_subtypeC
            subcellTy = findall(subtypeCell .== sty)
            phi_out[sty, :, iMCMC] = phi_MCMC[subcellTy[1], :]
            TStar_out[sty, :, iMCMC] = TStar_MCMC[subcellTy[1], :]
            TStar_withM_out[sty,:, iMCMC] = TStar_withM_MCMC[subcellTy[1], :]
            Tau_out[sty, :, iMCMC] = Tau_MCMC[subcellTy[1], :]
            k_out[sty,:,iMCMC] = k_MCMC[subcellTy[1],:]        
        end
        for tyT0_off = 1:n_typeT0_off
            LogT0_off_out[tyT0_off,:, iMCMC] = LogT0_off_MCMC[tyT0_off,:]
            T0_off_out[tyT0_off, :, iMCMC] = t0_off_MCMC[tyT0_off,:]
        end
        LogEta_out[:,iMCMC] = LogEta_MCMC[:]
        LogitCatt_out[:,iMCMC] = LogitCatt_MCMC[:]
    end

    return LogSS_out, SS_Star_out, LogT0_off_out, T0_off_out, phi_out, TStar_out, TStar_withM_out, Tau_out, k_out, LogEta_out, LogitCatt_out, acceptRateSS./mcmc.iter, acceptRateT0_off./mcmc.iter, acceptRateTStar./mcmc.iter, acceptRateEta./mcmc.iter, acceptRateCatt/mcmc.iter;
end