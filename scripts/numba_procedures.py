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


# TODO Check virial calculation (compare 2020-11-21 and the book, p.87)
# J.P.Hansen, I.R.McDonald. Theory Of Simple Liquids (2006), p.30
@njit
def lf_cycle(
        particles_number,
        verlet_list,
        marker_1,
        marker_2,
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
                marker_1[i],
                marker_2[i] + 1,
        ):
            j = verlet_list[k]
            radius_vector, distance = get_radius_vector(
                index_1=i,
                index_2=j,
                positions=positions,
                cell_dimensions=cell_dimensions
            )
            if distance < r_cut:
                table_row = int((distance - 0.5) / 0.0001)
                # table_row = round((distance - 0.5 + 1e-15) / 0.0001)
                potential_ij = potential_table[table_row - 1, 0]
                force_ij = potential_table[table_row - 1, 1]
                potential_energies[i] += potential_ij / 2.0
                potential_energies[j] += potential_ij / 2.0
                acceleration_ij = force_ij * radius_vector
                # virial += force_ij * distance * distance
                accelerations[i] += acceleration_ij
                accelerations[j] -= acceleration_ij
                assert table_row >= 1

    # TODO
    for i in prange(particles_number):
        for x in prange(3):
            virial += accelerations[i][x] * positions[i][x]

    return virial


@njit
def update_list_cycle(
        rng: float,
        advances: np.ndarray,
        particles_number: int,
        positions: np.ndarray,
        cell_dimensions: np.ndarray,
        marker_1: np.ndarray,
        marker_2: np.ndarray,
        verlet_list: np.ndarray,
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
            # distance_squared = (radius_vector ** 2).sum()
            if distance < rng:
                advances[j] = 1
            else:
                advances[j] = 0

        marker_1[i] = k
        for j in range(i + 1, particles_number):
            verlet_list[k] = j
            k = k + advances[j]
        marker_2[i] = k - 1
