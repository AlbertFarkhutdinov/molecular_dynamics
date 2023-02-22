import numba as nb
import numpy as np


@nb.njit
def math_round(value):
    rest = value - int(value)
    if rest >= 0.5 and value >= 0:
        return float(int(value) + 1)
    if rest <= -0.5 and value < 0:
        return float(int(value) - 1)
    return float(int(value))


@nb.njit
def get_radius_vector(index_1, index_2, positions, cell_dimensions):
    radius_vector = positions[index_1] - positions[index_2]
    if (
            radius_vector[0] < -cell_dimensions[0] / 2
            or radius_vector[0] >= cell_dimensions[0] / 2
    ):
        radius_vector[0] -= (
                math_round(radius_vector[0] / cell_dimensions[0])
                * cell_dimensions[0])

    if (
            radius_vector[0] < -cell_dimensions[0] / 2
            or radius_vector[0] >= cell_dimensions[0] / 2
    ):
        radius_vector[0] -= (
                math_round(radius_vector[0] / cell_dimensions[0])
                * cell_dimensions[0]
        )
    if (
            radius_vector[1] < -cell_dimensions[1] / 2
            or radius_vector[1] >= cell_dimensions[1] / 2
    ):
        radius_vector[1] -= (
                math_round(radius_vector[1] / cell_dimensions[1])
                * cell_dimensions[1]
        )
    if (
            radius_vector[2] < -cell_dimensions[2] / 2
            or radius_vector[2] >= cell_dimensions[2] / 2
    ):
        radius_vector[2] -= (
                math_round(radius_vector[2] / cell_dimensions[2])
                * cell_dimensions[2]
        )
    distance_squared = (
            radius_vector[0] * radius_vector[0]
            + radius_vector[1] * radius_vector[1]
            + radius_vector[2] * radius_vector[2]
    ) ** 0.5

    return radius_vector, distance_squared


@nb.njit
def get_interparticle_distances(positions, distances, cell_dimensions):
    for i in nb.prange(len(distances[0]) - 1):
        for j in nb.prange(i + 1, len(distances[0])):
            distance = 0
            for k in nb.prange(3):
                component = positions[i][k] - positions[j][k]
                if (
                        component < -cell_dimensions[k] / 2
                        or component >= cell_dimensions[k] / 2
                ):
                    component -= (
                            math_round(component / cell_dimensions[k])
                            * cell_dimensions[k]
                    )
                distance += component * component
            distances[i, j] = distance ** 0.5
    return distances


@nb.njit
def get_radius_vectors(positions, radius_vectors, cell_dimensions, distances):
    for i in nb.prange(len(radius_vectors[0]) - 1):
        for j in nb.prange(i + 1, len(radius_vectors[0])):
            distance = 0
            radius_vector = positions[i] - positions[j]
            for k in nb.prange(3):
                if (
                        radius_vector[k] < -cell_dimensions[k] / 2
                        or radius_vector[k] >= cell_dimensions[k] / 2
                ):
                    radius_vector[k] -= (
                            math_round(radius_vector[k] / cell_dimensions[k])
                            * cell_dimensions[k]
                    )
                distance += radius_vector[k] * radius_vector[k]
            distances[i, j] = distance ** 0.5
            radius_vectors[i, j] = radius_vector
    return radius_vectors, distances


@nb.njit
def get_time_displacements(positions_1, positions_2, distances):
    for i in nb.prange(len(distances[0]) - 1):
        for j in nb.prange(i + 1, len(distances[0])):
            distance = 0
            for k in nb.prange(3):
                component = positions_1[i][k] - positions_2[j][k]
                distance += component * component

            distances[i, j] = distance ** 0.5
    return distances


@nb.njit
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
    vir = 0
    for i in nb.prange(particles_number - 1):
        for k in nb.prange(
                first_neighbours[i],
                last_neighbours[i] + 1,
        ):
            j = all_neighbours[k]
            # log_debug_info(f'{i}, {j}')
            radius_vector, distance = get_radius_vector(
                index_1=i,
                index_2=j,
                positions=positions,
                cell_dimensions=cell_dimensions
            )
            if distance < r_cut:
                table_row = int(math_round((distance - 0.5) / 0.0001))
                potential_ij = potential_table[table_row - 1, 0]
                force_ij = potential_table[table_row - 1, 1]
                potential_energies[i] += potential_ij / 2.0
                potential_energies[j] += potential_ij / 2.0
                acceleration_ij = force_ij * radius_vector
                vir += force_ij * distance * distance / 2
                accelerations[i] += acceleration_ij
                accelerations[j] -= acceleration_ij
                # assert table_row >= 1

    return vir


@nb.njit
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
    for i in nb.prange(particles_number - 1):
        for j in nb.prange(i + 1, particles_number):
            _, distance = get_radius_vector(
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


@nb.njit
def get_static_structure_factors(
        wave_vectors,
        static_radius_vectors,
        particles_number,
):
    _static_structure_factors = []
    for i in nb.prange(wave_vectors.shape[0]):
        if (wave_vectors[i] == 0).all():
            item = static_radius_vectors.shape[0]
        else:
            item = 0
            for j in nb.prange(static_radius_vectors.shape[0]):
                angle = 0
                for k in nb.prange(3):
                    angle = (
                            angle
                            + wave_vectors[i][k]
                            * static_radius_vectors[j][k]
                    )

                item += np.cos(angle)
        _static_structure_factors.append(item / particles_number)
    return np.array(_static_structure_factors, dtype=np.float64)


@nb.njit
def get_unique_ssf(wave_numbers, static_structure_factors, layer_thickness):
    _wave_numbers = []
    _static_structure_factors = []
    _digits = int(np.log10(1 / layer_thickness))
    for i, number in enumerate(wave_numbers):
        _number = round(number, _digits)
        if _number not in _wave_numbers:
            _wave_numbers.append(_number)
            _static_structure_factors.append(static_structure_factors[i])
        else:
            _static_structure_factors[
                _wave_numbers.index(_number)
            ] += static_structure_factors[i]
    _wave_numbers = np.array(_wave_numbers)
    _static_structure_factors = np.array(_static_structure_factors)
    return _wave_numbers, _static_structure_factors


@nb.njit
def get_boundary_conditions(
        cell_dimensions: np.ndarray,
        particles_number: int,
        positions: np.ndarray,
):
    for i in range(particles_number):
        for j in range(3):
            if positions[i][j] >= cell_dimensions[j] / 2.0:
                positions[i][j] -= (
                        math_round(positions[i][j] / cell_dimensions[j])
                        * cell_dimensions[j]
                )
            if positions[i][j] < -cell_dimensions[j] / 2.0:
                positions[i][j] -= (
                        math_round(positions[i][j] / cell_dimensions[j])
                        * cell_dimensions[j]
                )
    return positions


def check_distances(particle_index, positions, cell_dimensions):
    for i in range(particle_index):
        radius_vector = (positions[i] - positions[particle_index])

        for k in range(3):
            radius_vector[k] -= cell_dimensions[k] * math_round(
                    radius_vector[k]
                    / cell_dimensions[k]
            )
        distance = (radius_vector ** 2).sum() ** 0.5
        if distance < 1.1:
            return True
    return False
