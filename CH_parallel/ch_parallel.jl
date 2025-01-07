
"""
This module solves phase separation in binary mixtures using the Cahn-Hilliard model.

Functions include initialization, chemical potential computation, and concentration update.

Parallelization is achieved using ParallelStencil.
"""


using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Random
using DelimitedFiles

const USE_GPU = false
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

# Parameters
const Nx, Ny = 256, 256
const dx, dy = 1.0, 1.0
const M = 1.0
const a_val = 1.0
const kappa = 1.0
const avg_comp = 0.45
const fluctuation = 0.0001
const t_end = 1000.0
const out_interval = 10000

# Macro for periodic boundary indices
macro pbc(ix, iy, nx, ny)
    quote
        im1 = $(esc(ix)) == 1 ? $(esc(nx)) : $(esc(ix))-1
        ip1 = $(esc(ix)) == $(esc(nx)) ? 1 : $(esc(ix))+1
        jm1 = $(esc(iy)) == 1 ? $(esc(ny)) : $(esc(iy))-1
        jp1 = $(esc(iy)) == $(esc(ny)) ? 1 : $(esc(iy))+1
        (im1, ip1, jm1, jp1)
    end
end

# Compute chemical potential with periodic boundaries
@parallel_indices (ix,iy) function compute_mu!(mu, comp, dx, dy)
    im1, ip1, jm1, jp1 = @pbc(ix, iy, Nx, Ny)
    
    # Compute Laplacian with periodic boundaries
    lap_comp = (comp[ip1,iy] + comp[im1,iy] + comp[ix,jp1] + comp[ix,jm1] - 4*comp[ix,iy]) / (dx * dy)
    
    # Free energy derivative
    c = comp[ix,iy]
    df_dc = 2 * a_val * c * (1 - c) * (1 - 2 * c)
    
    # Chemical potential
    mu[ix,iy] = df_dc - kappa * lap_comp
    return
end

# Compute composition evolution with periodic boundaries
@parallel_indices (ix,iy) function compute_evolution!(comp_new,comp, mu, dx, dy, dt)
    im1, ip1, jm1, jp1 = @pbc(ix, iy, Nx, Ny)
    
    # Compute Laplacian with periodic boundaries
    lap_mu = (mu[ip1,iy] + mu[im1,iy] + mu[ix,jp1] + mu[ix,jm1] - 4*mu[ix,iy]) / (dx * dy)
    
    comp_new[ix,iy] = comp[ix,iy] + dt * M * lap_mu

    return
end


function initial_condition(nx, ny, fluctuation, avg_comp)
    lower_bound = avg_comp * (1 - fluctuation)
    upper_bound = avg_comp * (1 + fluctuation)
    
    # Initialize on CPU first
    comp = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            comp[i,j] = rand() * (upper_bound - lower_bound) + lower_bound
        end
    end
    
    # Convert to GPU array if needed
    return Data.Array(comp)
end

# Function to get and save data at specific time
function save_data_at_time(comp, filename)
    writedlm(filename, comp)
end

function solve_cahn_hilliard()
    # Initialize arrays
    comp     = @zeros(Nx, Ny)
    comp_new = @zeros(Nx, Ny)
    mu       = @zeros(Nx, Ny)
    
    # Set initial condition
    comp .= initial_condition(Nx, Ny, fluctuation, avg_comp)
    comp_new .= comp
    
    # Time stepping
    t = 0.0
    iter = 0
    dt = 0.01
    
    while t < t_end
        if (iter % out_interval == 0)
            save_time = round(t, digits=2)
            print("Saving for time: ", save_time , "\n")
            save_data_at_time(comp, "output/time_$save_time.dat")
        end 

        # Compute chemical potential with PBC
        @parallel (1:Nx,1:Ny) compute_mu!(mu, comp, dx, dy)
        
        # Compute composition evolution with PBC
        @parallel (1:Nx,1:Ny) compute_evolution!(comp_new, comp, mu, dx, dy, dt)
        
        # Swap arrays for next iteration
        comp, comp_new = comp_new, comp
        
        t += dt
        iter += 1
    end
end

solve_cahn_hilliard()