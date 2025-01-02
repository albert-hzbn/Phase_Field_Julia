"""
# PeriodicVoronoi

This module provides functionality to generate a periodic Voronoi diagram. It includes functions to generate random points within a specified grid, find the nearest point to a given location, create a Voronoi grid, and save the results to CSV files.

## Functions

- `generate_points(nx::Int, ny::Int, n_points::Int=30)`: Generates `n_points` random points within a grid of size `nx` by `ny`, ensuring that points are not too close to each other based on a target spacing derived from the cell area.

- `find_nearest_point(x::Float64, y::Float64, points, nx::Int, ny::Int)`: Finds the index of the nearest point to the given coordinates `(x, y)` within a periodic grid of size `nx` by `ny`.

- `create_voronoi_grid(nx::Int, ny::Int, points)`: Creates a Voronoi grid for the given points within a grid of size `nx` by `ny`. It returns the grid with cell assignments and a boolean grid indicating the edges of the Voronoi cells.

- `save_to_csv(points, grid, edges)`: Saves the generated points, Voronoi grid, and edge information to CSV files. The files are named `voronoi_points.csv`, `voronoi_grid.csv`, and `voronoi_edges.csv`.

## Usage

1. Generate random points:
    ```julia
    points = generate_points(100, 100, 30)
    ```

2. Create a Voronoi grid:
    ```julia
    grid, edges = create_voronoi_grid(100, 100, points)
    ```

3. Save the results to CSV files:
    ```julia
    save_to_csv(points, grid, edges)
    ```

This module is useful for creating and analyzing periodic Voronoi diagrams, which have applications in various fields such as materials science, biology, and geography.
"""


module PeriodicVoronoi

using Random
using Printf
using DelimitedFiles

function generate_points(nx::Int, ny::Int, n_points::Int=30)
    points = []

    cell_area = nx * ny / n_points
    target_spacing = sqrt(cell_area)

    function is_too_close(x, y, existing_points)
        for (x2, y2) in existing_points
            # Compute periodic distance
            dx = min(abs(x - x2), nx - abs(x - x2))
            dy = min(abs(y - y2), ny - abs(y - y2))
            dist = sqrt(dx^2 + dy^2)
            if dist < target_spacing
                return true
            end
        end
        return false
    end

    attempts = 0
    max_attempts = n_points * 100

    while length(points) < n_points && attempts < max_attempts
        x = rand() * nx
        y = rand() * ny

        if !is_too_close(x, y, points)
            push!(points, (x, y))
        end
        attempts += 1
    end

    return points
end

function find_nearest_point(x::Float64, y::Float64, points, nx::Int, ny::Int)
    min_dist = Inf
    nearest_idx = 1

    for (idx, (px, py)) in enumerate(points)
        # Compute periodic distance
        dx = min(abs(x - px), nx - abs(x - px))
        dy = min(abs(y - py), ny - abs(y - py))
        dist = sqrt(dx^2 + dy^2)

        if dist < min_dist
            min_dist = dist
            nearest_idx = idx
        end
    end

    return nearest_idx
end

function create_voronoi_grid(nx::Int, ny::Int, points)
    grid = zeros(Int, nx, ny)
    edges = zeros(Bool, nx, ny)

    for i in 1:nx
        for j in 1:ny
            grid[i, j] = find_nearest_point(Float64(i), Float64(j), points, nx, ny)
        end
    end

    for i in 1:nx
        for j in 1:ny
            # Check periodic neighbors
            if grid[i, j] != grid[mod1(i + 1, nx), j]
                edges[i, j] = true
                edges[mod1(i + 1, nx), j] = true
            end
            if grid[i, j] != grid[i, mod1(j + 1, ny)]
                edges[i, j] = true
                edges[i, mod1(j + 1, ny)] = true
            end
        end
    end

    return grid, edges
end

function save_to_csv(points, grid, edges)
    # Save points - First write headers, then data
    open("voronoi_points.csv", "w") do io
        println(io, "x,y")  # Write header
        writedlm(io, reduce(vcat, [[x, y] for (x, y) in points]'), ',')
    end

    # Save grid (cell assignments)
    writedlm("voronoi_grid.csv", grid, ',')

    # Save edges
    writedlm("voronoi_edges.csv", Int.(edges), ',')

    println("Data saved to:")
    println("1. voronoi_points.csv - Point coordinates")
    println("2. voronoi_grid.csv - Cell assignments")
    println("3. voronoi_edges.csv - Edge information (1 for edge, 0 for interior)")
end

end # module PeriodicVoronoi
