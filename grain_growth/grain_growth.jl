# Import the module
include("PeriodicVoronoi.jl")
using .PeriodicVoronoi
using DelimitedFiles
using DifferentialEquations

# Define parameters
const nx, ny = 100, 100
const dx, dy = 1.0, 1.0
const t_end = 5000
alpha = 1.0
beta = 1.0
kappa = 1.0
L = 1.0
out_interval = 100  # Define the interval value for output
n_points = 50
dt = 0.01

# Generate points and create Voronoi grid
points = PeriodicVoronoi.generate_points(nx, ny, n_points)
grid, edges = PeriodicVoronoi.create_voronoi_grid(nx, ny, points)
PeriodicVoronoi.save_to_csv(points, grid, edges)

# Read the entire file as a matrix
data = readdlm("voronoi_grid.csv", ',')

# Determine the number of grains
num_grains = convert(Int16, maximum(data))

# Define the number of op_ranges(no. of order parameters)
num_op = 10

# Calculate the interval size
interval_size = div(num_grains, num_op)

# Create op_ranges
op_ranges = [(i * interval_size + 1, (i + 1) * interval_size) for i in 0:(num_op - 1)]
op_ranges[end] = (op_ranges[end][1], num_grains)  # Adjust the last range to include the maximum grain

# Initialize the etas array
etas = zeros(num_op, nx, ny)

# Fill each slice based on the corresponding range
for i in 1:num_op
    start, stop = op_ranges[i]
    etas[i, :, :] .= (data .>= start) .& (data .<= stop)
end

# Helper function for periodic boundary conditions
function apply_periodic_bc!(array)
    # Copy boundary values
    array[1, :] = array[end-1, :]
    array[end, :] = array[2, :]
    array[:, 1] = array[:, end-1]
    array[:, end] = array[:, 2]
end

# Function to get and save data at specific time
function save_data_at_time(etas, filename)
    out_data = zeros(nx, ny)
    for ind_op in 1:num_op
        for ind_x in 2:nx-1
            for ind_y in 2:ny-1
                out_data[ind_x, ind_y] += etas[ind_op, ind_x, ind_y] ^ 2
            end
        end
    end
    writedlm(filename, out_data)
end


for ind_time in 1:t_end
    if ind_time % out_interval == 0
        # Save the data
        save_data_at_time(etas, "output/time_$ind_time.dat")
    end
    for ind_op in 1:num_op
        apply_periodic_bc!(etas[ind_op, :, :])
        for ind_x in 2:nx-1
            for ind_y in 2:ny-1
                lap_etas = (
                    etas[ind_op, ind_x-1, ind_y] + 
                    etas[ind_op, ind_x+1, ind_y] + 
                    etas[ind_op, ind_x, ind_y-1] + 
                    etas[ind_op, ind_x, ind_y+1] - 
                    4 * etas[ind_op, ind_x, ind_y]
                ) / (dx * dy)

                sum = 0.0
                for ind_op2 in 1:num_op
                    if ind_op != ind_op2
                        sum += etas[ind_op2, ind_x, ind_y] ^ 2
                    end
                end
                dfdeta = alpha * (2 * beta * etas[ind_op, ind_x, ind_y] * sum + etas[ind_op, ind_x, ind_y] ^ 3 - etas[ind_op, ind_x, ind_y])
                etas[ind_op, ind_x, ind_y] = etas[ind_op, ind_x, ind_y] - dt * L * (dfdeta - kappa * lap_etas)
            end
        end
    end
end


