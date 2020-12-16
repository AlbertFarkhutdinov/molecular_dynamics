from numba import njit, prange
import numpy as np


@njit
def get_radius_vector(index_1, index_2, positions, cell_dimensions):
    radius_vector = positions[index_1] - positions[index_2]
    distance_squared = 0
    for k in prange(3):
        if radius_vector[k] < -cell_dimensions[k] / 2 or radius_vector[k] >= cell_dimensions[k] / 2:
            radius_vector[k] -= round(radius_vector[k] / cell_dimensions[k] + 1e-15) * cell_dimensions[k]
        distance_squared += radius_vector[k] * radius_vector[k]

    assert (
            -(cell_dimensions[0] / 2) <= radius_vector[0] < (cell_dimensions[0] / 2)
            or -(cell_dimensions[1] / 2) <= radius_vector[1] < (cell_dimensions[1] / 2)
            or -(cell_dimensions[2] / 2) <= radius_vector[2] < (cell_dimensions[2] / 2)
    )

    return radius_vector, distance_squared ** 0.5


@njit
def get_interparticle_distances(positions, distances, cell_dimensions):
    for i in prange(len(distances[0]) - 1):
        for j in prange(i + 1, len(distances[0])):
            # _, distances[i, j] = get_radius_vector(i, j, positions, cell_dimensions)
            distance = 0
            radius_vector = positions[i] - positions[j]
            for k in prange(3):
                if radius_vector[k] < -cell_dimensions[k] / 2 or radius_vector[k] >= cell_dimensions[k] / 2:
                    radius_vector[k] -= round(radius_vector[k] / cell_dimensions[k] + 1e-5) * cell_dimensions[k]
                distance += radius_vector[k] * radius_vector[k]

            # assert (
            #         -(cell_dimensions[0] / 2) <= radius_vector[0] < (cell_dimensions[0] / 2)
            #         or -(cell_dimensions[1] / 2) <= radius_vector[1] < (cell_dimensions[1] / 2)
            #         or -(cell_dimensions[2] / 2) <= radius_vector[2] < (cell_dimensions[2] / 2)
            # )
            distances[i, j] = distance ** 0.5
    return distances


@njit
def lf_cycle(
        particles_number,
        all_neighbours,
        first_neighbours,
        last_neighbours,
        r_cut,
        potential_table,
        potential_energies,
        positions,
        accelerations,
        cell_dimensions,
):
    virial = 0
    for i in prange(particles_number - 1):
        for k in prange(
                first_neighbours[i],
                last_neighbours[i] + 1,
        ):
            j = all_neighbours[k]
            radius_vector, distance = get_radius_vector(
                index_1=i,
                index_2=j,
                positions=positions,
                cell_dimensions=cell_dimensions
            )
            if distance < r_cut:
                # table_row = int((distance - 0.5) / 0.0001)
                table_row = round((distance - 0.5 + 1e-15) / 0.0001)
                potential_ij = potential_table[table_row - 1, 0]
                force_ij = potential_table[table_row - 1, 1]
                potential_energies[i] += potential_ij / 2.0
                potential_energies[j] += potential_ij / 2.0
                acceleration_ij = force_ij * radius_vector
                virial += force_ij * distance * distance / 2
                accelerations[i] += acceleration_ij
                accelerations[j] -= acceleration_ij
                assert table_row >= 1

    return virial


@njit
def update_list_cycle(
        rng: float,
        advances: np.ndarray,
        particles_number: int,
        positions: np.ndarray,
        cell_dimensions: np.ndarray,
        all_neighbours: np.ndarray,
        first_neighbours: np.ndarray,
        last_neighbours: np.ndarray,
):
    k = 1
    for i in prange(particles_number - 1):
        for j in prange(i + 1, particles_number):
            radius_vector, distance = get_radius_vector(
                index_1=i,
                index_2=j,
                positions=positions,
                cell_dimensions=cell_dimensions
            )
            if distance < rng:
                advances[j] = 1
            else:
                advances[j] = 0

        first_neighbours[i] = k
        for j in range(i + 1, particles_number):
            all_neighbours[k] = j
            k = k + advances[j]
        last_neighbours[i] = k - 1
