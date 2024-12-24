#=
    We first discretizes the right hand of equation to convert it into ODE
    and then solve it using the DifferentialEquations package 
=#

using DifferentialEquations
using Random

# Parameters
const Nx, Ny = 100, 100
const dx, dy = 1.0, 1.0
const M = 1.0
const epsilon = 1.0
const avg_comp = 0.3
const fluctuation = 0.0001
const a_val = 1.0
const kappa = 1.0
const t_end = 1000.0
const out_interval = 100.0

# Initial condition with proper array handling
function initial_condition(Nx, Ny, fluctuation, avg_comp)
    lower_bound = avg_comp * (1 - fluctuation)
    upper_bound = avg_comp * (1 + fluctuation)
    comp = zeros(Nx, Ny)
    for ind_x in 1:Nx
        for ind_y in 1:Ny
            comp[ind_x, ind_y] = rand() * (upper_bound - lower_bound) + lower_bound
        end
    end
    return comp
end

# Helper function for periodic boundary conditions
function apply_periodic_bc!(array)
    # Copy boundary values
    array[1, :] = array[Nx-1, :]
    array[Nx, :] = array[2, :]
    array[:, 1] = array[:, Ny-1]
    array[:, Ny] = array[:, 2]
end

# Cahn-Hilliard equation with proper boundary handling
function cahn_hilliard!(d_comp, comp, p, t)
    mu = zeros(Nx, Ny)
    
    # Apply periodic boundary conditions
    comp_ext = copy(comp)
    apply_periodic_bc!(comp_ext)
    
    # Calculate chemical potential
    for ind_x in 2:Nx-1
        for ind_y in 2:Ny-1
            # Compute Laplacian with proper spacing
            laplace_comp = (
                comp_ext[ind_x-1, ind_y] + 
                comp_ext[ind_x+1, ind_y] + 
                comp_ext[ind_x, ind_y-1] + 
                comp_ext[ind_x, ind_y+1] - 
                4 * comp_ext[ind_x, ind_y]
            ) / (dx * dy)
            
            # Free energy derivative
            c = comp_ext[ind_x, ind_y]
            df_dc = 2 * a_val * c * (1 - c) * (1 - 2 * c)
            
            # Chemical potential
            mu[ind_x, ind_y] = df_dc - kappa * laplace_comp
        end
    end
    
    # Apply periodic boundary conditions to chemical potential
    apply_periodic_bc!(mu)
    
    # Calculate composition evolution
    for ind_x in 2:Nx-1
        for ind_y in 2:Ny-1
            # Compute Laplacian of chemical potential
            laplace_mu = (
                mu[ind_x-1, ind_y] + 
                mu[ind_x+1, ind_y] + 
                mu[ind_x, ind_y-1] + 
                mu[ind_x, ind_y+1] - 
                4 * mu[ind_x, ind_y]
            ) / (dx * dy)
            
            d_comp[ind_x, ind_y] = M * laplace_mu
        end
    end
    
    # Apply periodic boundary conditions to rate of change
    apply_periodic_bc!(d_comp)
end

# Create initial condition
comp_init = initial_condition(Nx, Ny, fluctuation, avg_comp)

# Time span for simulation
tspan = (0.0, t_end)

# Problem definition
prob = ODEProblem(cahn_hilliard!, comp_init, tspan)

# Solve the problem with appropriate solver settings
sol = solve(prob, Tsit5(), 
    saveat=1.0,  # Save solution every 1.0 time units
    abstol=1e-6, 
    reltol=1e-6)


# Function to get and save composition at specific time
function save_composition_at_time(sol, time, filename)
    # Find the closest time point
    time_index = findmin(abs.(sol.t .- time))[2]
    actual_time = sol.t[time_index]
    
    # Get the composition at this time
    composition = sol.u[time_index]
    
    # Save to file using DelimitedFiles
    open(filename, "w") do io
        # Write the composition data
        for i in 1:size(composition,1)
            for j in 1:size(composition,2)
                print(io, composition[i,j])
                if j < size(composition,2)
                    print(io, ",")  # Tab separator
                end
            end
            println(io)  # New line after each row
        end
    end
    
    return composition, actual_time  # Return the data and actual time for reference
end

# Save compositions at different times
t = 0.0
while t <= t_end
    save_composition_at_time(sol, t, "output/composition_$t.dat")
    t += out_interval
end