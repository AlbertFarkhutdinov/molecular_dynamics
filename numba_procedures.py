import numba
import numpy as np


@numba.jit(nopython=True)
def get_radius_vector(index_1, index_2, positions, cell_dimensions):
    radius_vector = positions[index_1] - positions[index_2]
    if radius_vector[0] > cell_dimensions[0] / 2:
        radius_vector[0] -= cell_dimensions[0]
    elif radius_vector[0] < -cell_dimensions[0] / 2:
        radius_vector[0] += cell_dimensions[0]
    if radius_vector[1] > cell_dimensions[1] / 2:
        radius_vector[1] -= cell_dimensions[1]
    elif radius_vector[1] < -cell_dimensions[1] / 2:
        radius_vector[1] += cell_dimensions[1]
    if radius_vector[2] > cell_dimensions[2] / 2:
        radius_vector[2] -= cell_dimensions[2]
    elif radius_vector[2] < -cell_dimensions[2] / 2:
        radius_vector[2] += cell_dimensions[2]
    return radius_vector


@numba.jit(nopython=True)
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
    for i in range(particles_number - 1):
        for k in range(
                marker_1[i],
                marker_2[i] + 1,
        ):
            j = verlet_list[k]
            radius_vector = get_radius_vector(
                index_1=i,
                index_2=j,
                positions=positions,
                cell_dimensions=cell_dimensions
            )
            distance_squared = (radius_vector ** 2).sum()
            if distance_squared < r_cut * r_cut:
                table_row = int((distance_squared ** 0.5 - 0.5) / 0.0001)
                potential_ij = potential_table[table_row - 1, 0]
                force_ij = potential_table[table_row - 1, 1]
                potential_energies[i] += potential_ij / 2.0
                potential_energies[j] += potential_ij / 2.0
                virial += force_ij * distance_squared
                acceleration_ij = force_ij * radius_vector
                accelerations[i] += acceleration_ij
                accelerations[j] -= acceleration_ij
                assert table_row >= 1
    return virial


@numba.jit(nopython=True)
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
    for i in range(particles_number - 1):
        for j in range(i + 1, particles_number):
            radius_vector = get_radius_vector(
                index_1=i,
                index_2=j,
                positions=positions,
                cell_dimensions=cell_dimensions
            )
            distance_squared = (radius_vector ** 2).sum()
            if distance_squared < rng * rng:
                advances[j] = 1
            else:
                advances[j] = 0

        marker_1[i] = k
        for j in range(i + 1, particles_number):
            verlet_list[k] = j
            k = k + advances[j]
        marker_2[i] = k - 1


@numba.jit(nopython=True)
def get_interparticle_distances(positions, distances, cell_dimensions):
    for i in range(len(distances[0]) - 1):
        for j in range(i + 1, len(distances[0])):
            radius_vector = positions[i] - positions[j]
            if radius_vector[0] > cell_dimensions[0] / 2:
                radius_vector[0] -= cell_dimensions[0]
            elif radius_vector[0] < -cell_dimensions[0] / 2:
                radius_vector[0] += cell_dimensions[0]
            if radius_vector[1] > cell_dimensions[1] / 2:
                radius_vector[1] -= cell_dimensions[1]
            elif radius_vector[1] < -cell_dimensions[1] / 2:
                radius_vector[1] += cell_dimensions[1]
            if radius_vector[2] > cell_dimensions[2] / 2:
                radius_vector[2] -= cell_dimensions[2]
            elif radius_vector[2] < -cell_dimensions[2] / 2:
                radius_vector[2] += cell_dimensions[2]
            distances[i, j] = np.sqrt((radius_vector * radius_vector).sum())
    return distances


