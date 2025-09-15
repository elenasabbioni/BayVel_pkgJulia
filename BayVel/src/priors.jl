########################
#   SS
########################
# prior of SSSTar, where the priors are Beta
function priorLogSS_Star(
    LogSS_Star::Matrix{Float64},
    priorsLogU_SS_on::Log_Beta,
    priorsLogS_SS_on::Log_Beta,
    priorsLogU_SS_off::Log_Beta,
    priorsLogBeta::Log_Beta,
    gene::Int64,
    model::mod,
)::Float64 where{mod<:modelType}

# Compute the log-prior probability of steady-state parameters in log scale
# for a given gene under Beta-distributed priors.  

# # Arguments
# - LogSS_Star: log steady-state parameters (size 4 × n_genes) with 
#   - logarithm of u-coordinate of upper steady state steady-state 
#   - logarithm of s-coordinate of upper steady state steady-state 
#   - logarithm of u-coordinate of lower steady state steady-state 
#   - logarithm of splicing rate beta
# - priorsLogU_SS_on: prior distribution for logarithm of u-coordinate of upper steady-state 
# - priorsLogS_SS_on: prior distribution for logarithm of s-coordinate of upper steady-state 
# - priorsLogU_SS_off: prior distribution for logarithm of u-coordinate of lower steady-state 
# - priorsLogBeta: prior distribution for switching rate 
# - gene: gene index for which the prior is evaluated.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

# # Returns
# - log-prior probability for the selected gene.

    # Compute log-priors:
    # 1. U_on (normalized by a)
    # 2. S_on (normalized by a)
    # 3. U_off relative to U_on
    # Each prior is a Beta distribution applied to a scaled exponential of the log parameter

    prior::Float64 = 
        logpdf(Beta(priorsLogU_SS_on.par1, priorsLogU_SS_on.par2), 
            exp(LogSS_Star[1, gene])/priorsLogU_SS_on.par3
        ) + 
        logpdf(Beta(priorsLogS_SS_on.par1, priorsLogS_SS_on.par2), 
            exp(LogSS_Star[2, gene])/priorsLogS_SS_on.par3
        ) + 
        logpdf(Beta(priorsLogU_SS_off.par1, priorsLogU_SS_off.par2), 
        exp(LogSS_Star[3, gene])/exp(LogSS_Star[1, gene])
        ) - 
        # Jacobian corrections for log-transform
        LogSS_Star[1, gene] + LogSS_Star[1,gene] + LogSS_Star[2,gene] + LogSS_Star[3, gene]
        
        # Note: terms like -log(priorsLogU_SS_on.par3) cancel in Metropolis-Hastings
        # ratios, so they are omitted here.
    
    return prior
end

########################
#   T0
########################
function priorLogT0_off!(
    LogT0_off::Matrix{Float64},
    priorsLogT0_off::Distribution{Univariate, Continuous},
    prior::Matrix{Float64},
    gene::Int64,
    typeCellT0_off::Vector{Int64},
    model::mod,
)where{mod<:modelType}
# Compute and update the log-prior contribution of switching times t0_off  
# for all cell types associated with a given gene.

# # Arguments
# - LogT0_off: matrix of logarithm switching times (n_types × n_genes).
# - priorsLogT0_off: prior distribution for t0_offoff-switching time.
# - prior: matrix storing for log-prior values (n_cells × n_genes).
# - gene: index of the gene for which priors are computed.
# - typeCellT0_off: vector assigning each cell to a switching cluster (length = number of cells).
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)


    n_typeT0_off::Int64 = maximum(typeCellT0_off)
    for tyT0_off = 1:n_typeT0_off
        priorLogT0_off!(LogT0_off, priorsLogT0_off, prior, gene, tyT0_off, model)
    end   
end

function priorLogT0_off!(
    LogT0_off::Matrix{Float64},
    priorsLogT0_off::Distribution{Univariate, Continuous},
    prior::Matrix{Float64},
    gene::Int64,    
    tyT0_off::Int64, 
    model::mod,
)where{mod<:modelType}
# Compute and update the log-prior contribution of switching times t0_off  
# for a specific switching cluster with a given gene.

# # Arguments
# - LogT0_off: matrix of logarithm switching times (n_types × n_genes).
# - priorsLogT0_off: prior distribution for t0_offoff-switching time.
# - prior: matrix storing for log-prior values (n_cells × n_genes).
# - gene: index of the gene for which priors are computed.
# - tyT0_off: specific switching-cluster for which we have to compute the prior
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

    prior[tyT0_off, gene] = logpdf(priorsLogT0_off, LogT0_off[tyT0_off, gene])
end


########################
#   TStar/Phi
########################
function priorTStar!(
    phi_MCMC::Matrix{Float64},
    priorsT::Distribution{Univariate, Continuous},
    prior::Matrix{Float64},
    gene::Int64,
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64},
    model::groupSubgroup
)
# Compute and update the log-prior contribution for the elapsed time for all the cell types and for a specific gene

# # Arguments
# - phi_MCMC: matrix of angular transformation of time parameters (n_cells × n_genes).
# - priorsT: prior distribution for time
# - prior: matrix storing accumulated log-prior values (n_cells × n_genes).
# - gene: index of the gene for which priors are computed.
# - subtypeCell: vector mapping each cell to a subtype.
# - typeCell: vector mapping each cell to a main type.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

    n_typeC::Int64 = maximum(typeCell)
    # lopp over the cell types and update their log-prior contribution
    for ty = 1:n_typeC
        priorTStar!(phi_MCMC, priorsT, prior, gene, subtypeCell, typeCell, ty, model)
    end
    
end


function priorTStar!(
    phi_MCMC::Matrix{Float64},
    priorsT::Distribution{Univariate, Continuous},
    prior::Matrix{Float64},
    gene::Int64,
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64},
    ty::Int64,
    model::groupSubgroup    
)
# Compute and update the log-prior contribution for the elapsed time for a specific type and for a specific gene

# # Arguments
# - phi_MCMC: matrix of angular transformation of time parameters (n_cells × n_genes).
# - priorsT: prior distribution for time
# - prior: matrix storing accumulated log-prior values (n_cells × n_genes).
# - gene: index of the gene for which priors are computed.
# - subtypeCell: vector mapping each cell to a subtype.
# - typeCell: vector mapping each cell to a main type.
# - ty:: cell type for which we are computing the prior
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

    # find which cells are associated with that cell type
    cellTy::Vector{Int64} = findall(typeCell.== ty)
    if size(cellTy, 1) > 0 # in case we found cells of this type
        for sty = unique(subtypeCell[cellTy]) # iterate over all the subtypes associated with this type
            subcellTy::Vector{Int64} = findall(subtypeCell.== sty)  # find which cells are associated with that cell subtype
            if size(subcellTy, 1) > 0 # in case we found cells of this subtype
                priorTStar!(phi_MCMC, priorsT, prior, gene, subcellTy, model)  # update 
            end
        end
    end
end

function priorTStar!(
    phi_MCMC::Matrix{Float64},
    priorsT::Distribution{Univariate, Continuous},
    prior::Matrix{Float64},
    gene::Int64,
    subcellTy::Vector{Int64},
    model::groupSubgroup
)

# Compute and update the log-prior contribution for the elapsed time for a specific subtype and for a specific gene

# # Arguments
# - phi_MCMC: matrix of angular transformation of time parameters (n_cells × n_genes).
# - priorsT: prior distribution for time
# - prior: matrix storing accumulated log-prior values (n_cells × n_genes).
# - gene: index of the gene for which priors are computed.
# - subcellTy: vector of the cells belongings to the selected subtypes
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

    # compute the log-prior for this specific cell subtype and gene
    prior[subcellTy, gene] .= logpdf(priorsT.Unif, mod(phi_MCMC[subcellTy[1], gene], 1.0))
end

########################
#   ETA
########################
function priorETA(
    LogEta::Vector{Float64},
    priorsEta::Distribution{Univariate, Continuous},
    gene::Int64, 
    model::mod
)where{mod<:modelType}
# Compute the log-prior contribution of the overdispersion parameter eta # for a specific gene.

# # Arguments
# - LogEta: vector of logairthm of dispersion parameters (per gene).
# - priorsEta: prior distribution for eta
# - gene: index of the gene for which the prior is evaluated.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

# # Returns
# - prior: the log-prior contribution of η for the given gene.

    # Evaluate the log-prior density    
    prior = logpdf(priorsEta, exp(LogEta[gene])) + 
            # Add LogEta term for Jacobian adjustment 
            LogEta[gene]

    return prior
end


########################
#   CATT
########################
function priorCATT(
    LogitCatt::Vector{Float64},
    priorsCatt::Distribution{Univariate, Continuous},
    cell::Int64, 
    model::mod
)where{mod<:modelType}
# Compute the log-prior contribution of the capture efficiency (given in logit scale) for a specific cell
# given in logit scale.

# # Arguments
# - LogitCatt: vector of cell-specific capture efficiencies on the logit scale.
# - priorsCatt: prior distribution for capture efficicency
# - cell: index of the cell for which the prior is evaluated.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

# # Returns
# - prior: the log-prior contribution of capture efficiency for the given cell.

    prior = logpdf(priorsCatt, logistic(LogitCatt[cell])) + 
            # Add Jacobian correction for the logit transform
            LogitCatt[cell] - 2*log1pexp(LogitCatt[cell]) 

    return prior

end
