# =======================================
#       PARAMETERS' INITIALIZATIONS
# =======================================

function initParam!(
    initUoff::Vector{Float64},
    initSoff::Vector{Float64},
    initDiffU::Vector{Float64},
    initBeta::Vector{Float64},
    initT0_off::Matrix{Float64},
    initU0_off::Matrix{Float64},
    initS0_off::Matrix{Float64},
    initU0_on::Matrix{Float64},
    initS0_on::Matrix{Float64},
    initTStar_withM::Matrix{Float64},
    initTStar::Matrix{Float64},
    initTau::Matrix{Float64},
    initK::Matrix{Int64},
    initEta::Vector{Float64},
    initCatt::Vector{Float64},
   
    alpha_real::Matrix{Float64}, 
    beta_real::Vector{Float64},
    gamma_real::Vector{Float64},
    t0_off_real::Matrix{Float64},
    u0_off_real::Matrix{Float64},
    s0_off_real::Matrix{Float64},
    t_real::Matrix{Float64},
    tau_real::Matrix{Float64}, 
    k_real::Matrix{Int64}, 
    eta_real::Vector{Float64}, 
    catt_real::Vector{Float64}, 
    simSS::Bool, 
    simT0::Bool,
    simTau::Bool,
    simEta::Bool, 
    simCatt::Bool,
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64}, 
    typeCellT0_off::Vector{Int64},

    model::groupSubgroup
)

    # Initialize all parameters needed to start MCMC inference of the RNA velocity Bayesian model. 
    # The function handles real and simulated settings for each parameter depending on the sim* flags.

    # # Arguments

    # - initUoff, initSoff, initDiffU, initBeta: Vectors for steady-state (SS) initialization.
    # - initT0_off, initU0_off, initS0_off: Matrices for off switching time and associated coordinates.
    # - initU0_on, initS0_on: Matrices for on-branch initial values (usually constant).
    # - initTStar_withM, initTStar, initTau: Time matrices.
    # - initK: Subgroup state matrix (binary: 0 = off, 1 = on).
    # - initEta, initCatt: Gene-specific overdispersion and cell-specific capture efficiency.
    # - *_real: Ground-truth values (used when sim* = false).
    # - simSS, simT0, simTau, simEta, simCatt: Flags indicating whether to simulate the corresponding parameter.
    # - subtypeCell, typeCell, typeCellT0_off: Cell type/subtype groupings for initialization logic.
    # - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

    # Initialize from real values if simulation flags are off
    if(!simSS)
        realSS!(initUoff, initSoff, initDiffU, initBeta, initU0_on, initS0_on, alpha_real, beta_real, gamma_real, model) # Use real SS coordinates
    end
    if(!simT0)
        realT0_off!(initT0_off, initU0_off, initS0_off, t0_off_real, u0_off_real, s0_off_real, model) # Use real switching time t0_off
    end
    if(!simTau)
        realTStar!(initTStar, t_real, model) # Use real latent time t*
        initTStar_withM[:, :] .= initTStar[:, :]
        initTStar_withM[initTStar_withM .< 0.0] .= 0.0 # Ensure valid time
    end

    ##############
    # Simulate steady state parameters if requested
    ##############
    if(simSS) 
        initSS!(initUoff, initSoff, initDiffU, initBeta,  initU0_on, initS0_on, beta_real, model)       
    end

    ##############
    # Simulate T0, TStar, TStar_withM, K parameters if requested
    ##############
    if simT0*simTau
        initT0_offTauK!(initT0_off, initTStar_withM, initTStar, initTau, initK, subtypeCell,typeCell,typeCellT0_off, model)
    elseif (!simTau & !simT0)  # Just initialize k and tau from real values
        initTau[:, :] .= initTStar_withM[:, :]
        for g = 1:size(initTStar, 2)
            for sty = 1:maximum(subtypeCell)
                subcellTy::Vector{Int64} = findall(subtypeCell .== sty)
                if initTStar[subcellTy[1], g]  > initT0_off[typeCellT0_off[subcellTy[1]], g]
                    initK[subcellTy, g] .= 0
                    initTau[subcellTy, g] .= initTStar_withM[subcellTy, g] .- initT0_off[typeCellT0_off[subcellTy[1]], g]
                else
                    initK[subcellTy, g] .= 1
                end
            end
        end
    elseif (!simT0 & simTau)  # initialize tStar, tStar_withM, k and tau
        initTau_k!(initTStar, initTStar_withM, initTau, initK, t0_off_real, subtypeCell, typeCell, typeCellT0_off, model)
    elseif (simT0 & !simTau)  # initialize t0, k and tau
        initT0_off!(initT0_off, initTau, initK, t_real, typeCellT0_off, model)
    end

    # Recalculate initial U0_off and S0_off based on T0, SS, etc.
    for g = 1:size(initT0_off, 2)
        SS::Matrix{Float64} = Matrix(transpose([initUoff initSoff initDiffU initBeta]))
        mu_tot_g!(SS, initT0_off, initU0_off, initS0_off, initTau, initK, initU0_off, initS0_off,g, initCatt, true, typeCellT0_off, model)
    end  

    ##############
    # Simulate overdispersion parameters if requested
    ##############
    if(simEta) 
        initEta!(initEta, model)
    else
        realEta!(initEta, eta_real, model)
    end

    ##############
    # Simulate capture efficiency parameters if requested
    ##############
    if(simCatt) 
        initCatt!(initCatt, model)
    else
        realCatt!(initCatt, catt_real, model)
    end
end

function initSS!(
    initUoff::Vector{Float64},
    initSoff::Vector{Float64},
    initDiffU::Vector{Float64},
    initBeta::Vector{Float64},
    initU0_on::Matrix{Float64},
    initS0_on::Matrix{Float64},
    beta_real::Vector{Float64},
    model::mod
)where{mod<:modelType}

# Randomly initializes steady-state parameters.
# initUoff, initSoff, and initDiffU are sampled from a log-normal distribution: 0.5 + exp(Normal(0, 0.5)).
# beta is kept equal to the real beta (usually set equal to 1).

# # Arguments
# - initUoff: Vector to store initialized u coordinate for the off steady-state.
# - initSoff: Vector to store initialized s coordinate for the off steady-state.
# - initDiffU: Vector to store initialized difference in the u coordinate of the steady states.
# - initBeta: Vector to store initialized degradation rates .
# - initU0_on: Matrix to store u coordinate at t0_on.
# - initS0_on: Matrix to store s coordinate at t0_on.
# - beta_real: Ground-truth β parameter (used only if model is of type modelType).
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

# # Notes
# This function is used when simulating steady-state parameters (i.e., when simSS == true).

    initUoff[:] .= 0.5 .+  exp.(rand(Normal(0.0,0.5), size(initUoff,1)))
    initSoff[:] .= 0.5 .+  exp.(rand(Normal(0.0,0.5), size(initUoff,1)))
    initDiffU[:] .= 0.5 .+  exp.(rand(Normal(0.0,0.5), size(initUoff,1)))

    initU0_on[1,:] .= initUoff[:] # alpha_off/beta
    initS0_on[1,:] .= initSoff[:] # alpha_off/gamma

    if (typeof(model) <: modelType)
        initBeta[:] .= beta_real[:] # ones(Float64, size(initUoff,1))
    else
        initBeta[:] .= exp.(rand(Normal(0.0,1.0), size(initUoff,1))) # TO DO not implemented yet
    end
end

function initT0_offTauK!(
    initT0_off::Matrix{Float64},
    initTStar_withM::Matrix{Float64},
    initTStar::Matrix{Float64},
    initTau::Matrix{Float64},
    initK::Matrix{Int64},
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64},
    typeCellT0_off::Vector{Int64}, 
    model::groupSubgroup
)

# Initialize the off switching time points, subgroup times (`t_star`),
# the elapsed time tau, and switching status indicators (`k`) for each gene and cell.

# # Arguments
# - initT0_off: Matrix of initial OFF-switching times per group and gene.
# - initTStar_withM: Matrix of initial times with mass in the lower steady state.
# - initTStar: Matrix of initial times,
# - initTau: Matrix of initial elapsed time on the current branch.
# - initK: Matrix indicating ON/OFF switching status (1 = ON, 0 = OFF) for each gene in each subgroup.
# - subtypeCell: Subgroup index assigned to each cell.
# - typeCell: Group index assigned to each cell.
# - typeCellT0_off: Switching groups label
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)


    # initialize the off switching time with positive random values
    for tyT0_off =1:size(initT0_off, 1)
        initT0_off[tyT0_off,:] .= exp.(rand(Normal(0.0,1.0), size(initT0_off, 2)))
    end

    # Ensure no t0_off values are exactly zero (avoid mass at 0 at the initialization)
    for tyT0_off = 1:size(initT0_off, 1)
        if any(initT0_off[tyT0_off,:] .== 0.0)
            ind::Vector{Int64} = findall(initT0_off[tyT0_off, :] .== 0.0)
            initT0_off[tyT0_off,ind] .= initT0_off[tyT0_off, ind] + rand(Uniform(0.1, 1.0), size(ind, 1))
        end
    end
    
    # Initialize t_star values per subtype and gene
    for ty = 1:maximum(typeCell)
        cellTy::Vector{Int64} = findall(typeCell .== ty)
        
        for sty = unique(subtypeCell[cellTy])
            subCellTy::Vector{Int64} = findall(subtypeCell .== sty)    
            for g = 1:size(initTStar, 2)
                initTStar[subCellTy, g] .= exp.(rand(Normal(0.0, 1.0), 1))
            end
        end
    end
    # I some t_star <= 0.0, set them to 0 (mass at 0)
    for c = 1:size(initTStar, 1)
        trasform_TStar!(initTStar, initTStar_withM, c, model)
    end

    # Initialize branch status matrix
    initK[:,:] .= 1
    for tyT0_off = 1:size(initT0_off, 1)
        # Find cells belonging to the given type for t0_off
        cellTyT0_off::Vector{Int64} = findall(typeCellT0_off .== tyT0_off)
        for g = 1:size(initK, 2)
            # Cells where t_star >= t0_off: mark as OFF (k = 0)
            off::Vector{Int64} = cellTyT0_off[findall(initTStar_withM[cellTyT0_off, g] .>= initT0_off[tyT0_off,g])]
            initK[off, g] .= 0
        end
    end

    # set tau as a copy of the time (assuming t0_on = 0)
    initTau[:, :] .= initTStar_withM[:, :]
    # For OFF cells (k = 0), subtract t0_off to get τ = t_star - t0_off
    for tyT0_off = 1:size(initT0_off, 1)
        cellTyT0_off::Vector{Int64} = findall(typeCellT0_off .== tyT0_off) 
        for g = 1:size(initK, 2)
            off::Vector{Int64} = findall(initK[cellTyT0_off, g] .== 0)
            initTau[cellTyT0_off[off], g] .= initTau[cellTyT0_off[off], g] .- initT0_off[tyT0_off, g] 
        end
    end    
end

function initTau_k!(
    initTStar::Matrix{Float64},
    initTStar_withM::Matrix{Float64},
    initTau::Matrix{Float64},
    initK::Matrix{Int64},
    t0_off_real::Matrix{Float64},
    subtypeCell::Vector{Int64},
    typeCell::Vector{Int64},
    typeCellT0_off::Vector{Int64},
    model::groupSubgroup
)

# Initialize time (`t_star`), its version with mass in 0 (`t_star_withM`), 
# elapsed time on since th elast switching (`tau`), and switching indicators (`k`) using known OFF-switching 
# time `t0_off_real`.

# # Arguments
# - initTStar: Matrix to store time.
# - initTStar_withM: Matrix to store time with mass in 0.
# - initTau: Matrix of elapsed time tau = t_star − t0_on/t0_off.
# - initK: Binary matrix (1 = ON, 0 = OFF) for branch.
# - t0_off_real: Matrix of true OFF-switching points per type and gene.
# - subtypeCell: Subtype assignment vector for each cell.
# - typeCell: Type assignment vector for each cell.
# - typeCellT0_off: Type used to look up t0_off values per cell.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

    # Initialize t_star values randomly per subtype and gene
    for ty = 1:maximum(typeCell)
        cellTy::Vector{Int64} = findall(typeCell .== ty)
        
        for sty = unique(subtypeCell[cellTy])
            subCellTy::Vector{Int64} = findall(subtypeCell .== sty)    
            for g = 1:size(initTStar, 2)
                 # Assign same value for all cells in the subtype
                initTStar[subCellTy, g] .= exp.(rand(Normal(0.0, 1.0), 1))
            end
        end
    end
    
    #  Transform t_star to t_star_withM putting mass in 0
    for c = 1:size(initTStar, 1)
        trasform_TStar!(initTStar, initTStar_withM, c, model)
    end

    # initialize k and tau (tau = t - t0_off when gene is OFF (t_star ≥ t0_off_real),  tau = t when gene is ON (t_star < t0_off_real))
    initK[:,:] .= 1
    initTau[:, :] .= initTStar_withM[:, :]
    for tyT0_off = 1:maximum(typeCellT0_off)    
        cellTyT0_off::Vector{Int64} = findall(typeCellT0_off .== tyT0_off)
        for g = 1:size(initK, 2)
            off::Vector{Int64} = cellTyT0_off[findall(initTStar_withM[cellTyT0_off, g] .>= t0_off_real[tyT0_off,g])]
            initK[off, g] .= 0
             # Adjust tau by subtracting t0_off_real
            initTau[off, g] .= initTau[off, g] .- t0_off_real[tyT0_off, g] 
        end
    end
end   

function initT0_off!(
    initT0_off::Matrix{Float64},
    initTau::Matrix{Float64},
    initK::Matrix{Int64},
    t_real::Matrix{Float64},
    typeCellT0_off::Vector{Int64},
    model::groupSubgroup
)

# Initialize the off switching time points

# # Arguments
# - initT0_off: Matrix of initial OFF-switching times per group and gene.
# - initTau: Matrix of initial elapsed time on the current branch.
# - initK: Matrix indicating ON/OFF switching status (1 = ON, 0 = OFF) for each gene in each subgroup.
# - t_real: Ground-truth t values.
# - typeCellT0_off: Switching groups label
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

    # Initialize tau to the real pseudotime values
    initTau[:,:] .= t_real[:, :]

    # Loop over types used to define t0_off
    for tyT0_off = 1:maximum(typeCellT0_off)
        # Find cells assigned to this t0_off type
        cellTyT0_off::Vector{Int64} = findall(typeCellT0_off.== tyT0_off)
        for g = 1:size(initT0_off, 2)
            # Randomly initialize t0_off for this type and gene
            initT0_off[tyT0_off, g] = rand(Uniform(0.01, 100.0), 1)[1]
            
            # Find cells where gene is OFF (t > t0_off) and modify accordingly initK and initTau
            off::Vector{Int64} = cellTyT0_off[findall(t_real[cellTyT0_off, g] .> initT0_off[tyT0_off, g])]
            initK[off, g] .= 0
            initTau[off, g] .= t_real[off, g] .- initT0_off[tyT0_off, g]

            # Find cells where gene is ON (t > t0_off) and modify accordingly initK (initTau remains equal to t_real, since t0_on =)
            on::Vector{Int64} = cellTyT0_off[findall(t_real[cellTyT0_off, g] .< initT0_off[tyT0_off, g])]
            initK[on, g] .= 1
        end
    end 
end

function initEta!(
    initEta::Vector{Float64},
    model::mod
) where{mod<:modelType}
# Initializes the overdispersion parameter with random positive values, sampled from a log-normal distribution.
# # Arguments
# - initEta: Vector to be initialized.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
    initEta[:] .= exp.(rand(Normal(0.0,1.0), size(initEta,1)))
end

function initCatt!(
    initCatt::Vector{Float64},
    model::mod
)where{mod<:modelType}
# Initializes the capture efficiencies with random values uniformly distributed in [0, 1].

# Arguments
# - initCatt: Vector to be initialized.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
    initCatt[:] .= rand(Uniform(0.0, 1.0), size(initCatt,1))
end


function realSS!(
    initUoff::Vector{Float64},
    initSoff::Vector{Float64},
    initDiffU::Vector{Float64},
    initBeta::Vector{Float64},
    initU0_on::Matrix{Float64},
    initS0_on::Matrix{Float64},

    alpha_real::Matrix{Float64}, 
    beta_real::Vector{Float64},
    gamma_real::Vector{Float64}, 
    model::mod    
) where{mod<:modelType}

# Initialize steady-state parameters using real data.

# # Arguments
# - initUoff: Initial u coordinate for OFF steady-state.
# - initSoff: Initial s coordinate for OFF steady-state.
# - initDiffU: Initial difference between u coordinates of OFF and ON steady-state.
# - initBeta: Splicing rate.
# - initU0_on: Initial u coordinate at t0_on.
# - initS0_on: Initial s coordinate at t0_on.
# - alpha_real: Real transcription rates; 2 columns: alpha_off and alpha_on
# - beta_real: Real Splicing rates.
# - gamma_real: Real degradation rates.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

# # Details
# Uses analytical steady-state formulas:
# - For OFF steady state: (alpha_off/beta, alpha_off/gamma)
# - For ON steady state: (alpha_on/beta, alpha_on/gamma)

    initUoff[:] .= alpha_real[:, 1] ./ beta_real
    initSoff[:] .= alpha_real[:, 1] ./ gamma_real
    initDiffU[:] .= (alpha_real[:, 2] .- alpha_real[:, 1]) ./ beta_real
    initBeta[:] .= beta_real[:]
    
    initU0_on[1,:] .= initUoff[:]
    initS0_on[1,:] .= initSoff[:]

end

function realT0_off!(
    initT0_off::Matrix{Float64},
    initU0_off::Matrix{Float64},
    initS0_off::Matrix{Float64},
    t0_off_real::Matrix{Float64},
    u0_off_real::Matrix{Float64},
    s0_off_real::Matrix{Float64},
    model::mod
)where{mod<:modelType}

# Assigns real (true) values to the switching point at time t = t0_off.
# # Arguments
# - initT0_off: Matrix to initialize off switching time.
# - initU0_off: Matrix to initialize the u coordinate corresponding to the off switching time.
# - initS0_off: Matrix to initialize the s coordinate corresponding to the off switching time.
# - t0_off_real: Ground-truth t0_off values.
# - u0_off_real: Ground-truth u coordinate corresponding to t0_off.
# - s0_off_real: Ground-truth s coordinate corresponding to t0_off.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

# Notes
# This function is used when simT0 == false, i.e., when you do not want to simulate the t0_off-related parameters but rather initialize them from known ground-truth values.

    initT0_off[:,:] .= t0_off_real[:,:]
    initU0_off[:,:] .= u0_off_real[:,:]
    initS0_off[:,:] .= s0_off_real[:,:]
end

function realTStar!(
    initTStar::Matrix{Float64},
    t_real::Matrix{Float64},
    model::groupSubgroup
)

# Assigns real (true) values to the time.
# # Arguments
# - initTStar: Matrix to initialize time.
# - t_real: Ground-truth t values.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

# Notes
# This function is used when simTau == false, i.e., when you do not want to simulate the t-related parameters but rather initialize them from known ground-truth values.

    initTStar[:,:] .= t_real[:, :]
end

function realEta!(
    initEta::Vector{Float64},
    eta_real::Vector{Float64}, 
    model::mod
)where{mod<:modelType}
# Assigns real (true) values to overdispersion parameter.
# # Arguments
# - initEta: Vector to initialize overdispersion parameter.
# - eta_real: Ground-truth overdispersion values.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

# Notes
# This function is used when simEta == false, i.e., when you do not want to simulate the eta-related parameters but rather initialize them from known ground-truth values.

    initEta[:] .= eta_real[:]
end

function realCatt!(
    initCatt::Vector{Float64},
    catt_real::Vector{Float64}, 
    model::mod
)where{mod<:modelType}
# Assigns real (true) values to capture efficiency.
# # Arguments
# - initCatt: Vector to initialize capture efficiency.
# - catt_real: Ground-truth capture efficiency au values.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)

# Notes
# This function is used when simCatt == false, i.e., when you do not want to simulate the capture-related parameters but rather initialize them from known ground-truth values.

    initCatt[:] .= catt_real[:]
    
end

# =======================================
#       PARAMETERS' TRANSFORMATIONS
# =======================================
function trasform_TStar!(
    TStar::Matrix{Float64},
    TStar_withM::Matrix{Float64},
    cell::Int64,
    model::groupSubgroup
)
# Set to zero all components of `TStar` that are negative for a given cell,  
# and store the transformed version into `TStar_withM`. 
# This wrapper function loops over all genes `g` for a given `cell` and calls the lower-level method `trasform_TStar!(TStar, TStar_withM, g, cell, model)` that performs the actual transformation.

# # Arguments
# - `TStar`: Matrix of times (cells x genes)
# - `TStar_withM`: Matrix where the transformed version of `TStar` is stored 
#   (negative values set to 0).
# - `cell`: Index of the cell whose values are being transformed.
# - `model`:  An instance of the model type (currently unused but included for interface consistency).

    # Iterate over all genes for the given cell
    for g = 1:size(TStar,2)
        # Apply the transformation for each gene g of cell `cell`
        trasform_TStar!(TStar, TStar_withM, g, cell, model)
    end
end

function trasform_TStar!(
    TStar::Matrix{Float64},
    TStar_withM::Matrix{Float64},
    gene::Int64,
    cell::Int64,
    model::groupSubgroup
)
# Transform a single entry of `TStar` for a given `(cell, gene)` pair,  
# ensuring that negative values are set to zero. The result is stored in `TStar_withM`.

# # Arguments
# - `TStar`: Matrix of times (cells × genes)
# - `TStar_withM`: Matrix where the transformed version of `TStar` is stored.
# - `gene`: Index of the gene to transform.
# - `cell`: Index of the cell to transform.
# - `model`: An instance of the model type (currently unused but included for interface consistency).

# # Notes
# - If `TStar[cell, gene] < 0.0`, the value in `TStar_withM` is set to `0.0`.  
# - Otherwise, the original value is copied unchanged.  

    # If latent time is negative, reset to 0
    if TStar[cell, gene] < 0.0
        TStar_withM[cell, gene] = 0.0
    else
        # Otherwise, keep the original value
        TStar_withM[cell, gene] = TStar[cell, gene]
    end
end

function trasformU_toPhi(
    u::Matrix{Float64},
    tStar::Matrix{Float64},
    priorsT::Distribution{Univariate, Continuous}, 
    SS::Matrix{Float64}, 
    t0_off::Matrix{Float64}, 
    u0_off::Matrix{Float64},
    gene::Int64, 
    subcellTy::Vector{Int64},
    subtypeCell::Vector{Int64}, 
    typeCellT0_off ::Vector{Int64}
)
# Map unspliced RNA coordinate (u) to an angular position \phi ∈ [0, 1]
#
# Arguments
# - u: Matrix of u-coordinates (cells × genes).
# - tStar: Times 
# - priorsT: Prior distribution on times, used to put mass on the lower steady state.
# - SS: Matrix of steady-state values
# - t0_off: Matrix of off-switching times for each subtype and gene.
# - u0_off: Matrix of unspliced coordinates at the off-switching time.
# - gene: Index of the gene under consideration.
# - subcellTy: Indices of cells belonging to the current subtype.
# - subtypeCell: Vector mapping cells to subtypes.
# - typeCellT0_off: Vector mapping subtypes to switching clustering.
#
# Returns
# - \phi in [0, 1]: normalized angular coordinate corresponding to u.
    
    u0_on::Float64 = SS[1, gene] # u-corrdinate at t0_on
    lim::Float64 = (1.0 - priorsT.p_zero)/2.0 # proportion of the [0,1] interval associated with the on-phase

    tyT0_off::Int64 = typeCellT0_off[subcellTy[1]]  # index of the switching clustering
    diff_u::Float64 = u0_off[tyT0_off, gene] - u0_on # difference between OFF and ON steady states for u coordinate
 
    # we are in the on-phase: tStar before t0_off and positive
    if ((tStar[subcellTy[1], gene]) <= (t0_off[tyT0_off, gene])) & (tStar[subcellTy[1], gene] > 0.0) #

        phi = (u[subcellTy[1], gene] - u0_on)/diff_u*lim
        return phi

    elseif tStar[subcellTy[1], gene] > t0_off[tyT0_off, gene] # we are in the off-phase: tStar after that t0_off

        phi = (u[subcellTy[1], gene] - u0_off[tyT0_off, gene])/(-diff_u) *lim + lim

        return phi
    else # we are in the OFF steady state
        return 2.0*lim
    end
end

function trasformPhi_toT(
    phi::Matrix{Float64},
    priorsT::Distribution{Univariate, Continuous}, 
    SS::Matrix{Float64}, 
    t0_off::Matrix{Float64}, 
    u0_off::Matrix{Float64},
    gene::Int64, 
    subcellTy::Vector{Int64},
    subtypeCell::Vector{Int64}, 
    typeCellT0_off ::Vector{Int64}, 
    model::groupSubgroup
)
# Map angular coordinate \phi in [0, 1] to the corresponding time tStar
#
# Arguments
# - phi: Matrix of angular coordinates (cells × genes).
# - priorsT: Prior distribution on times, used to extract mass at zero (p_zero).
# - SS: Matrix of steady-state values
# - t0_off: Matrix of off-switching times for each subtype and gene.
# - u0_off: Matrix of unspliced coordinates at the off-switching time.
# - gene: Index of the gene under consideration.
# - subcellTy: Indices of cells belonging to the current subtype.
# - subtypeCell: Vector mapping cells to subtypes.
# - typeCellT0_off: Vector mapping subtypes to switching clustering.
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Returns
# - tStar: Time coordinate corresponding to phi.

    u0_on::Float64 = SS[1, gene]                    # u-corrdinate at t0_on 
    lim::Float64 = (1.0 - priorsT.p_zero)/2.0       # proportion of the [0,1] interval associated with the on-phase

    alpha_off, alpha_on, gamma = ss_to_rates(SS, gene, model)   # transcription and degradation rates 
    beta::Float64 = SS[4, gene]                                 # splicing rate

    tyT0_off::Int64 = typeCellT0_off[subcellTy[1]]    # index of the switching clustering
    diff_u::Float64 = u0_off[tyT0_off, gene] - u0_on  # difference between OFF and ON steady states for u coordinate 

    u_prop::Float64 = 0.0

    # transformation from phi to tStar
    if phi[subcellTy[1], gene] < lim # on-phase
        u_prop = u0_on + (phi[subcellTy[1], gene]/lim)*diff_u
        return 1/beta* log((alpha_off - alpha_on)/(beta*u_prop - alpha_on))
    elseif phi[subcellTy[1], gene] < (2.0*lim)  # off-phase
        u_prop = u0_off[tyT0_off, gene] -  ((phi[subcellTy[1], gene] - lim)/lim)*diff_u
        return 1/beta*log((beta*u0_off[tyT0_off, gene] - alpha_off)/(beta*u_prop - alpha_off))  + t0_off[tyT0_off, gene]
    else
        return 0.0   # lower steady state
    end
end

function ss_to_rates(
    SS::Matrix{Float64}, 
    model::mod
)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}} where{mod<:modelType}
# Compute transcription and splicing rates from steady-state information
#
# Arguments
# - SS: A 4 × n_genes matrix with rows:
#       1. u_ss_off  — steady-state unspliced coordinates in the OFF state
#       2. s_ss_off  — steady-state spliced coordinates in the OFF state
#       3. diffU     — difference between ON and OFF steady-state unspliced coordinates
#       4. beta      — splicing rate
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Returns
# - alpha_off: Vector of transcription rates in the OFF state
# - alpha_on:  Vector of transcription rates in the ON state
# - gamma:     Vector of degradation rates

    alpha_off::Vector{Float64} = Vector{Float64}(undef, size(SS,2))
    alpha_on::Vector{Float64} = Vector{Float64}(undef, size(SS,2))
    gamma::Vector{Float64} = Vector{Float64}(undef, size(SS,2))
    
    for g = 1:size(SS, 2)
        alpha_off[g], alpha_on[g], gamma[g] = ss_to_rates(SS, g, model)
    end
    
    return alpha_off, alpha_on, gamma
end

function ss_to_rates(
    SS::Matrix{Float64},
    gene::Int64,
    model::mod
)::Tuple{Float64, Float64, Float64} where{mod<:modelType}
# Compute transcription and splicing rates for a specific gene from steady-state information
# Arguments
# Arguments
# - SS: A 4 × n_genes matrix with rows:
#       1. u_ss_off  — steady-state unspliced coordinates in the OFF state
#       2. s_ss_off  — steady-state spliced coordinates in the OFF state
#       3. diffU     — difference between ON and OFF steady-state unspliced coordinates
#       4. beta      — splicing rate
# - gene: Index of the gene for which to compute the rates
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Returns
# - alpha_off: Transcription rate in the OFF state for the specified gene
# - alpha_on:  Transcription rate in the ON state for the specified gene
# - gamma:     Degradation rate for the specified gene

    # compute degradation rate
    uON::Float64 = SS[1,gene] .+ SS[3,gene]                 # compute the ON-state unspliced steady-state
    alpha_off::Float64 = SS[4,gene] .* SS[1,gene]           # compute transcription rates in OFF states
    alpha_on::Float64 = SS[4,gene] .* uON                   # compute transcription rates in ON states
    gamma::Float64 = SS[1,gene]./SS[2,gene] .*  SS[4,gene]  # compute the degradation rate

    return alpha_off, alpha_on, gamma
end

function ss_to_ssStar(
    SS::Matrix{Float64}, 
    model::mod
)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}} where{mod<:modelType}
# Compute steady state coordinates for all genes from the 4 × n_genes steady-state information
# Arguments
# - SS: A 4 × n_genes matrix with rows:
#       1. u_ss_off  — steady-state unspliced coordinates in the OFF state
#       2. s_ss_off  — steady-state spliced coordinates in the OFF state
#       3. diffU     — difference between ON and OFF steady-state unspliced coordinates
#       4. beta      — splicing rate
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Returns
# - u_SS_on:  Vector of ON steady-state unspliced coordinates for all genes
# - s_SS_on:  Vector of ON steady-state spliced coordinates for all genes
# - u_SS_off: Vector of steady-state unspliced coordinates for all genes
    
    u_SS_on::Vector{Float64} = Vector{Float64}(undef, size(SS,2))
    s_SS_on::Vector{Float64} = Vector{Float64}(undef, size(SS,2))
    u_SS_off::Vector{Float64} = Vector{Float64}(undef, size(SS,2))

    for g = 1:size(SS, 2)
        u_SS_on[g], s_SS_on[g], u_SS_off[g] = ss_to_ssStar(SS, g, model) # compute steady-state quantities for each gene individually
    end
    
    return u_SS_on, s_SS_on, u_SS_off
end

function ss_to_ssStar(
    SS::Matrix{Float64},
    gene::Int64,
    model::mod
)::Tuple{Float64, Float64, Float64} where{mod<:modelType}
# Compute steady state coordinates for a specific genes from the 4 × n_genes steady-state information
# Arguments
# - SS: A 4 × n_genes matrix with rows:
#       1. u_ss_off  — steady-state unspliced coordinates in the OFF state
#       2. s_ss_off  — steady-state spliced coordinates in the OFF state
#       3. diffU     — difference between ON and OFF steady-state unspliced coordinates
#       4. beta      — splicing rate
# - gene: Index of the gene to compute
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Returns
# - u_SS_on:  Vector of ON steady-state unspliced coordinates for the specific gene
# - s_SS_on:  Vector of ON steady-state spliced coordinates for the specific gene
# - u_SS_off: Vector of steady-state unspliced coordinates for the specific gene

    u_SS_on::Float64 = SS[1, gene] + SS[3, gene]            # u-coordinate of the ON steady state
    u_SS_off::Float64 =  SS[1, gene]                        # u-coordinate of the OFF steady state
    s_SS_on::Float64 = u_SS_on * SS[2, gene]/ u_SS_off      # s-coordinate of the ON steady state
    
    return u_SS_on, s_SS_on, u_SS_off
end

function ssStar_to_ss(
    ssStar::Matrix{Float64}, 
    model::mod
)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}} where{mod<:modelType}
# Transformation of steady-state coordinates for all genes 
#
# Arguments
# - ssStar: A 4 × n_genes matrix with rows:
#       1. u_SS_on   — upper steady-state unspliced coordinates
#       2. s_SS_on   — upper steady-state spliced coordinates
#       3. u_SS_off  — lower steady-state unspliced coordinates
#       4. beta      — splicing rate
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Returns
# - u_ss_off: Vector of lower steady-state unspliced coordinates for all genes
# - s_ss_off: Vector of lower steady-state spliced coordinates for all genes
# - diffU:    Vector of differences between ON and OFF steady-state unspliced coordinates for all genes

    u0_on::Vector{Float64} = Vector{Float64}(undef, size(rates,2))
    s0_on::Vector{Float64} = Vector{Float64}(undef, size(rates,2))
    diffU::Vector{Float64} = Vector{Float64}(undef, size(rates,2))

    for g = 1:size(rates, 2)
        u0_on[g], s0_on[g], diffU[g] = ssStar_to_ss(ssStar, g, model)
    end
    
    return u0_on, s0_on, diffU
end


function ssStar_to_ss(
    ssStar::Matrix{Float64},
    gene::Int64,
    model::mod
)::Tuple{Float64, Float64, Float64} where{mod<:modelType}
# Transformation of steady-state coordinates for a specific gene
#
# Arguments
# - ssStar: A 4 × n_genes matrix with rows:
#       1. u_SS_on   — upper steady-state unspliced coordinates
#       2. s_SS_on   — upper steady-state spliced coordinates
#       3. u_SS_off  — lower steady-state unspliced coordinates
#       4. beta      — splicing rate
# - gene: Index of the gene to compute
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Returns
# - u_ss_off: Vector of lower steady-state unspliced coordinates for a specific gene
# - s_ss_off: Vector of lower steady-state spliced coordinates for a specific gene
# - diffU:    Vector of differences between ON and OFF steady-state unspliced coordinates for a specific gene
    u0_on::Float64 = ssStar[3,gene]
    s0_on::Float64 = ssStar[3,gene]*ssStar[2, gene]/ssStar[1, gene]
    diffU::Float64 = ssStar[1,gene] - ssStar[3, gene]

    return u0_on, s0_on, diffU
end

# =======================================
#       SOLUTION OF THE ODE
# =======================================
function mu_tot_g!(
    SS::Matrix{Float64}, 
    t0_off::Matrix{Float64},
    u0_off::Matrix{Float64},
    s0_off::Matrix{Float64},
    Tau::Matrix{Float64},
    k::Matrix{Int64}, 
    mu_u::Matrix{Float64}, 
    mu_s::Matrix{Float64},
    gene::Int64,
    catt::Vector{Float64},
    initialPoint::Bool,
    typeCellT0_off::Vector{Int64},
    model::mod
)where{mod<:modelType}
# Compute the expected mean of the spliced and unspliced counts distribution (mu_u, mu_s) for a specific gene across cells
#
# Arguments
# - SS:           4 x n_genes matrix with steady-state parameters (u_ss_off, s_ss_off, diffU, beta) 
# - t0_off:       Matrix of off-switching times for each subtype × gene
# - u0_off:       Matrix of u-coordinate at t0_off for each subtype × gene
# - s0_off:       Matrix of s-coordinate at t0_off for each subtype × gene
# - Tau:          Matrix of times from the switching time (n_cells × n_genes)
# - k:            Matrix of on/off indicators (n_cells × n_genes)
# - mu_u:         Matrix to store mean of unspliced counts (n_cells × n_genes)
# - mu_s:         Matrix to store mean of spliced counts (n_cells × n_genes)
# - gene:         Index of the gene we are considering
# - catt:         Vector of cell-specific capture efficiencies
# - initialPoint: Bool, whether to update only the coordinates of the switching points (t0_off) or all cells
# - typeCellT0_off: Vector mapping cells to switching clustering
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Notes
# - If `initialPoint` is true, only the coordinates at t0_off are updated. Otherwise, all relevant cells are updated.
# - Internally calls `mu_tot_gc` for the computation of each gene.

    alpha_off, alpha_on, gamma = ss_to_rates(SS, gene, model) # Convert steady-state to kinetic rates for the gene
    u0_on::Float64 = alpha_off/SS[4, gene] # u-coordinate at upper steady state
    s0_on::Float64 = alpha_off/gamma       # s-coordinate at upper steady state
    cellTyT0_off::Vector{Int64} = [0]

    # Loop over all switching clusters
    for tyT0_off = 1:maximum(typeCellT0_off)
        # Identify cells belonging to this off-switching subtype and their switching coordinates
        cellTyT0_off = findall(typeCellT0_off.== tyT0_off) 
        t0_offTy = t0_off[tyT0_off, gene] 
        u0_offTy = u0_off[tyT0_off, gene]
        s0_offTy = s0_off[tyT0_off, gene]
        
        if initialPoint
            # Update only the coordinates of the switching point
            u0_off[tyT0_off, gene], s0_off[tyT0_off, gene] = mu_tot_gc(t0_offTy, u0_offTy, s0_offTy, u0_on, s0_on, Tau, k, gene, 1, catt, alpha_off, alpha_on, gamma, initialPoint, model)            
        else
            # Update all cells in this subtype
            for c = cellTyT0_off
                mu_u[c, gene], mu_s[c, gene] = mu_tot_gc(t0_offTy, u0_offTy, s0_offTy, u0_on, s0_on, Tau, k, gene, c, catt, alpha_off, alpha_on, gamma, initialPoint, model)
            end
        end
    end
end 

function mu_tot_g_withFailed!(
    SS::Matrix{Float64}, 
    t0_off::Matrix{Float64},
    u0_off::Matrix{Float64},
    s0_off::Matrix{Float64},
    Tau::Matrix{Float64},
    k::Matrix{Int64}, 
    mu_u::Matrix{Float64}, 
    mu_s::Matrix{Float64},
    gene::Int64,
    catt::Vector{Float64},
    initialPoint::Bool,
    typeCellT0_off::Vector{Int64},
    model::mod
)::Bool where{mod<:modelType}
# Compute the expected mean of the spliced and unspliced counts distribution (mu_u, mu_s) for a specific gene across cells and verify if the coordinates are negative or not. Returns a boolean.
#
# Arguments
# - SS:           4 x n_genes matrix with steady-state parameters (u_ss_off, s_ss_off, diffU, beta) 
# - t0_off:       Matrix of off-switching times for each subtype × gene
# - u0_off:       Matrix of u-coordinate at t0_off for each subtype × gene
# - s0_off:       Matrix of s-coordinate at t0_off for each subtype × gene
# - Tau:          Matrix of times from the switching time (n_cells × n_genes)
# - k:            Matrix of on/off indicators (n_cells × n_genes)
# - mu_u:         Matrix to store mean of unspliced counts (n_cells × n_genes)
# - mu_s:         Matrix to store mean of spliced counts (n_cells × n_genes)
# - gene:         Index of the gene we are considering
# - catt:         Vector of cell-specific capture efficiencies
# - initialPoint: Bool, whether to update only the coordinates of the switching points (t0_off) or all cells
# - typeCellT0_off: Vector mapping cells to switching clustering
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
# Returns
# - failed: Bool indicating whether any computed mu_u or mu_s value was negative
#
# Notes
# - Works like `mu_tot_g!` but additionally checks if computed values are negative.
# - If any negative values are found, `failed` is set to true. This can be used to detect invalid parameter combinations.
# - Internally calls `mu_tot_gc` for the computation of each gene.

    alpha_off, alpha_on, gamma = ss_to_rates(SS, gene, model) # Convert steady-state to kinetic rates for the gene
    u0_on::Float64 = alpha_off/SS[4, gene]                    # u coordinate for the upper steady state
    s0_on::Float64 = alpha_off/gamma                          # s coordinate for the lower steady state
    
    cellTy::Vector{Int64} = [0]
    failed::Bool = false

    # Loop over all switching clusters
    for tyT0_off = 1:maximum(typeCellT0_off)
        # Identify cells belonging to this off-switching subtype and their switching coordinates
        cellTyT0_off = findall(typeCellT0_off.== tyT0_off)
        t0_offTy = t0_off[tyT0_off, gene]
        u0_offTy = u0_off[tyT0_off, gene]
        s0_offTy = s0_off[tyT0_off, gene]
        
        if initialPoint
            # Update only the coordinates of the switching point
            u0_off[tyT0_off, gene], s0_off[tyT0_off, gene] = mu_tot_gc(t0_offTy, u0_offTy, s0_offTy, u0_on, s0_on, Tau, k, gene, 1, catt, alpha_off, alpha_on, gamma, initialPoint, model)  

            # check if the coordinates results negative
            if (u0_off[tyT0_off, gene] < 0.0) | (s0_off[tyT0_off, gene] < 0.0)
                failed = true
            end
        else
            # Update all cells in this subtype
            for c = cellTyT0_off
                mu_u[c, gene], mu_s[c, gene] = mu_tot_gc(t0_offTy, u0_offTy, s0_offTy, u0_on, s0_on, Tau, k, gene, c, catt, alpha_off, alpha_on, gamma, initialPoint, model)

                # check if the coordinates results negative
                if (mu_u[c, gene] < 0.0) | (mu_s[c, gene] < 0.0)
                    failed = true
                end
            end
        end
    end
    return failed
end 

function mu_tot_c!(
    SS::Matrix{Float64},
    t0_off::Matrix{Float64},
    u0_off::Matrix{Float64},
    s0_off::Matrix{Float64},
    Tau::Matrix{Float64}, 
    k::Matrix{Int64}, 
    mu_u::Matrix{Float64}, 
    mu_s::Matrix{Float64},
    cell::Int64,
    catt::Vector{Float64},
    initialPoint::Bool,
    typeCellT0_off::Vector{Int64}, 
    model::mod
)where{mod<:modelType}
# Compute the expected mean of the spliced and unspliced counts distribution (mu_u, mu_s) for a specific cell across genes
#
# Arguments
# - SS:             4 x n_genes matrix with steady-state parameters (u_ss_off, s_ss_off, diffU, beta) 
# - t0_off:         Matrix of off-switching times for each subtype × n_genes
# - u0_off:         Matrix of u-coordinates at t0_off for each subtype × n_genes
# - s0_off:         Matrix of s-coordinates at t0_off for each subtype × s_genes
# - Tau:            Matrix of times to compute expression (n_cells × n_genes)
# - k:              Matrix of on/off indicators (n_cells × n_genes)
# - mu_u:           Matrix to store expected unspliced counts (n_cells × n_genes)
# - mu_s:           Matrix to store expected spliced counts (n_cells × n_genes)
# - cell:           Index of the cell we are considering
# - catt:           Vector of cell-specific capture efficiencies
# - initialPoint:   Bool, whether to update only the coordinates of the switching points (t0_off) or cell
# - typeCellT0_off: Vector mapping cells to switching clustering
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Notes
# - If `initialPoint` is true, only updates the initial off-state coordinates (u0_off, s0_off) for the subtype of the cell.
# - Otherwise, updates the expected counts `mu_u` and `mu_s` for the given cell across all genes.
# - Internally calls `mu_tot_gc` for the computation of each gene.

    alpha_off, alpha_on, gamma = ss_to_rates(SS, model) # Convert steady-state to kinetic rates for the gene
    u0_on::Float64 = 0.0
    s0_on::Float64 = 0.0
    tyT0_off::Int64 = typeCellT0_off[cell] # switching cluster of the selected cell

    if initialPoint
        for g = 1:size(SS,2)
            # Update only the coordinates of the switching point for this cluster for all the genes
            u0_on = alpha_off[g]/SS[4,g]
            s0_on = alpha_off[g]/gamma[g]     
            u0_off[tyT0_off, g], s0_off[tyT0_off, g] = mu_tot_gc(t0_off[tyT0_off, g], u0_off[tyT0_off, g], s0_off[tyT0_off, g], u0_on, s0_on, Tau, k, g, 1, catt, alpha_off[g], alpha_on[g], gamma[g], initialPoint, model)
        end
    else
        for g = 1:size(SS,2)
            # update the mean of spliced and unspliced counts for the selected cell for all the genes
            u0_on = alpha_off[g]/SS[4,g]
            s0_on = alpha_off[g]/gamma[g]
            mu_u[cell, g], mu_s[cell, g] = mu_tot_gc(t0_off[tyT0_off, g], u0_off[tyT0_off, g], s0_off[tyT0_off, g], u0_on, s0_on, Tau, k, g, cell, catt, alpha_off[g], alpha_on[g], gamma[g], initialPoint, model)
        end
    end
end 

function mu_tot_gc(
    t0_off::Float64,
    u0_off::Float64,
    s0_off::Float64,
    u0_on::Float64,
    s0_on::Float64,
    Tau::Matrix{Float64}, 
    k::Matrix{Int64}, 
    gene::Int64,
    cell::Int64,
    catt::Vector{Float64},
    alpha_off::Float64, 
    alpha_on::Float64, 
    gamma::Float64,    
    initialPoint::Bool,
    model::mod
) where{mod<:modelType}
# Compute the expected mean of the spliced and unspliced counts distribution (mu_u, mu_s) for a specific cell and gene
# Arguments
# - t0_off:        Off-switching time for the current subtype and gene
# - u0_off:        Unspliced coordinate at t0_off for the subtype and gene
# - s0_off:        Spliced coordinate at t0_off for the subtype and gene
# - u0_on:         Unspliced coordinate at steady-state on for the selected gene
# - s0_on:         Spliced coordinate at steady-state on for the selected gene
# - Tau:           Matrix of times for cells × genes
# - k:             Matrix of on/off indicators for cells × genes
# - gene:          Index of the gene we are considering
# - cell:          Index of the cell we are considering
# - catt:          Vector of cell-specific capture efficiencies
# - alpha_off:     Transcription rate alpha_off for the gene
# - alpha_on:      Transcription rate alpha_on for the gene
# - gamma:         Degradation rate gamma for the gene
# - initialPoint:  Bool, whether to update only the coordinates of the switching points (t0_off) or of the selected cell and gene
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Returns
# - A tuple (u, s) with mean of unspliced and spliced counts
#
# Notes
# - If `initialPoint` is true, the function updates only u0_off and s0_off for the subtype.
# - Otherwise, it computes the mean of unspliced and spliced counts at `Tau[cell, gene]` considering whether the cell is in the on or off state according to `k[cell, gene]`.
# - Uses the rates alpha_off, alpha_on, and gamma to compute the expression dynamics.

    resU::Float64 = 0.0
    resS::Float64 = 0.0
    tau::Float64 = 0.0
    k_cg::Int64 = 1

    if initialPoint
        # the time in which we want to compute u(t,...) and s(t, ...) is t0_off
        tau = t0_off
        k_cg = 1
    else
        # the time in which we want to compute u(t,...) and s(t, ...) is the elapsed time for this cell and gene
        tau = Tau[cell, gene]
        k_cg = k[cell, gene]
    end
   
    expBeta::Float64  = exp.(0.0 .- (1.0 .* tau)) # e^{-beta*tau}, with beta = 1
    expGamma::Float64 = exp(- (gamma * tau))      # e^{-gamma*tau}

    if (k_cg == 0)
        # repressive phase: compute the solution of the ODE in tau, for the off branch
        resU = u0_off .* expBeta .+ alpha_off/1.0  .* (1.0 .- expBeta)        
        resS = s0_off .* expGamma .+ alpha_off/gamma .* (1.0 .- expGamma) .+ (alpha_off - 1.0*u0_off) ./ (gamma - 1.0) .* (expGamma .- expBeta)
    else 
        # inductive phase: compute the solution of the ODE in tau, for the on branch
        resU = u0_on .* expBeta .+ alpha_on/1.0 .* (1.0 .- expBeta)
        resS = s0_on .* expGamma .+ alpha_on/gamma .* (1.0 .- expGamma)  .+ (alpha_on - 1.0*u0_on) ./ (gamma - 1.0) .* (expGamma .- expBeta)
    end

    if !initialPoint    
        # scale the solution of the ODE by the capture efficiencies
        resU = catt[cell] .* resU
        resS = catt[cell] .* resS
    end
 
    return resU, resS
end

# =======================================
#       LIKELIHOOD
# =======================================

function loglik_g!(
    unspliced::Matrix{Int64},
    spliced::Matrix{Int64},
    mu_u::Matrix{Float64},
    mu_s::Matrix{Float64},
    invEta::Vector{Float64},
    loglik::Matrix{Float64},
    gene::Int64,
    par2::Matrix{Float64},
    par4::Matrix{Float64}, 
    model::mod
)where{mod<:modelType}
# Compute the log-likelihood for all the cells and a specific gene under a Negative Binomial or Poisson model
#
# Arguments
# - unspliced: Matrix of observed unspliced counts
# - spliced:   Matrix of observed spliced counts
# - mu_u:      Matrix of expected unspliced means
# - mu_s:      Matrix of expected spliced means
# - invEta:    Vector of inverse dispersion parameters (1/eta) for each gene
# - loglik:    Matrix to store the computed log-likelihood for each cell × gene
# - gene:      Index of the gene we are considering
# - par2:      Matrix to store the Negative Binomial probability parameter p for unspliced counts
# - par4:      Matrix to store the Negative Binomial probability parameter p for spliced counts
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Notes
# - The Negative Binomial parameterization is different between R and Julia. R follows convention with mean mu and dispersion eta, while Julia uses the size r and the probability parameter p. The transformation between the two is the following: 
# --> r = 1/eta,               
# --> p = r/(mu+r),
# - If invEta[gene] is infinite (eta is 0), the distribution is treated as Poisson.

    # iterate the computation of the likelihood over the different cells
    for c = 1:size(unspliced, 1)
        loglik!(unspliced, spliced, mu_u, mu_s, invEta, loglik, gene, c, par2, par4, model)
    end
end

function loglik_c!(
    unspliced::Matrix{Int64},
    spliced::Matrix{Int64},
    mu_u::Matrix{Float64},
    mu_s::Matrix{Float64},
    invEta::Vector{Float64},
    loglik::Matrix{Float64},
    cell::Int64,
    par2::Matrix{Float64},
    par4::Matrix{Float64}, 
    model::mod
)where{mod<:modelType}
# Compute the log-likelihood for all the genes and a specific cell under a Negative Binomial or Poisson model
#
# Arguments
# - unspliced: Matrix of observed unspliced counts
# - spliced:   Matrix of observed spliced counts
# - mu_u:      Matrix of expected unspliced means
# - mu_s:      Matrix of expected spliced means
# - invEta:    Vector of inverse dispersion parameters (1/eta) for each gene
# - loglik:    Matrix to store the computed log-likelihood for each cell × gene
# - cell:      Index of the cell we are considering
# - par2:      Matrix to store the Negative Binomial probability parameter p for unspliced counts
# - par4:      Matrix to store the Negative Binomial probability parameter p for spliced counts
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Notes
# - The Negative Binomial parameterization is different between R and Julia. R follows convention with mean mu and dispersion eta, while Julia uses the size r and the probability parameter p. The transformation between the two is the following: 
# --> r = 1/eta,               
# --> p = r/(mu+r),
# - If invEta[gene] is infinite (eta is 0), the distribution is treated as Poisson.

    # iterate the computation of the likelihood over the different genes
    for g = 1:size(unspliced, 2)
        loglik!(unspliced, spliced, mu_u, mu_s, invEta, loglik, g, cell, par2, par4, model)
    end
end

function loglik!(
    unspliced::Matrix{Int64},
    spliced::Matrix{Int64},
    mu_u::Matrix{Float64},
    mu_s::Matrix{Float64},
    invEta::Vector{Float64},
    loglik::Matrix{Float64},
    gene::Int64, 
    cell::Int64,
    par2::Matrix{Float64},
    par4::Matrix{Float64}, 
    model::mod
)where{mod<:modelType}
# Compute the log-likelihood for a specific gene and cell under a Negative Binomial or Poisson model
#
# Arguments
# - unspliced: Matrix of observed unspliced counts
# - spliced:   Matrix of observed spliced counts
# - mu_u:      Matrix of expected unspliced means
# - mu_s:      Matrix of expected spliced means
# - invEta:    Vector of inverse dispersion parameters (1/eta) for each gene
# - loglik:    Matrix to store the computed log-likelihood for each cell × gene
# - gene:      Index of the gene we are considering
# - cell:      Index of the cell we are considering
# - par2:      Matrix to store the Negative Binomial probability parameter p for unspliced counts
# - par4:      Matrix to store the Negative Binomial probability parameter p for spliced counts
# - model: Model type object, used to initialize initBeta (currently unused but included for interface consistency)
#
# Notes
# - The Negative Binomial parameterization is different between R and Julia. R follows convention with mean mu and dispersion eta, while Julia uses the size r and the probability parameter p. The transformation between the two is the following: 
# --> r = 1/eta,               
# --> p = r/(mu+r),
# - If invEta[gene] is infinite (eta is 0), the distribution is treated as Poisson.
     
    par1::Float64 = invEta[gene] # r for unspliced counts distribution
    par3::Float64 = invEta[gene] # r for spliced counts distribution

    par2[cell, gene]= (invEta[gene])/(mu_u[cell,gene] + invEta[gene]) # p for unspliced counts distribution
    par4[cell, gene]= (invEta[gene])/(mu_s[cell,gene] + invEta[gene]) # p for spliced counts distribution
    
    # Compute log-likelihood
    if !isinf(invEta[gene])
        # Negative Binomial case
        loglik[cell, gene] = logpdf(NegativeBinomial(par1, par2[cell, gene]), unspliced[cell,gene]) + logpdf(NegativeBinomial(par3, par4[cell, gene]), spliced[cell,gene]) 
    else
        # Poisson case (as a limit when eta -> 0)
        loglik[cell, gene] = logpdf(Poisson(mu_u[cell, gene]), unspliced[cell,gene]) + logpdf(Poisson(mu_s[cell, gene]), spliced[cell,gene]) 
    end
end
