import numpy as np

from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.external_parameters import ExternalParameters
from scripts.helpers import get_empty_float_scalars, get_empty_int_scalars, get_empty_vectors
from scripts.log_config import log_debug_info, logger_wraps
from scripts.modeling_parameters import ModelingParameters
from scripts.integrators.npt_mttk import MTTK
from scripts.numba_procedures import lf_cycle, update_list_cycle
from scripts.potential_parameters import PotentialParameters
from scripts.static_parameters import SystemStaticParameters
from scripts.integrators.nvt_velocity_scaling import VelocityScaling


class Verlet:

    def __init__(
            self,
            static: SystemStaticParameters,
            dynamic: SystemDynamicParameters,
            external: ExternalParameters,
            model: ModelingParameters,
            potential: PotentialParameters,
    ):
        self.potential = potential
        self.static = static
        self.dynamic = dynamic
        self.external = external
        self.model = model
        self.neighbours_lists = {
            'all_neighbours': get_empty_int_scalars(self.static.particles_number),
            'first_neighbours': get_empty_int_scalars(self.static.particles_number),
            'last_neighbours': get_empty_int_scalars(100 * self.static.particles_number),
        }

    @logger_wraps()
    def system_dynamics(
            self,
            stage_id: int,
            environment_type: str,
    ):
        integrator = None
        if environment_type == 'velocity_scaling':
            integrator = VelocityScaling(
                dynamic=self.dynamic,
                external=self.external,
                model=self.model,
            )
        elif environment_type == 'mtk':
            integrator = MTTK(
                static=self.static,
                dynamic=self.dynamic,
                external=self.external,
                time_step=self.model.time_step,
            )
        if stage_id == 1:
            integrator.stage_1()
        if stage_id == 2:
            integrator.stage_2()

        system_kinetic_energy = self.dynamic.system_kinetic_energy
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        cell_volume = self.static.get_cell_volume()
        density = self.static.get_density(volume=cell_volume)
        pressure = self.dynamic.get_pressure(
            cell_volume=cell_volume,
            density=density,
        )
        total_energy = system_kinetic_energy + self.dynamic.potential_energy
        log_postfix = f'after {integrator.__class__.__name__}.stage_{stage_id}()'
        log_debug_info(f'Kinetic Energy {log_postfix}: {system_kinetic_energy};')
        log_debug_info(f'Temperature {log_postfix}: {temperature};')
        log_debug_info(f'Pressure {log_postfix}: {pressure};')
        log_debug_info(f'Total energy {log_postfix}: {total_energy};')
        log_debug_info(f'Cell volume {log_postfix}: {cell_volume};')
        log_debug_info(f'Density {log_postfix}: {density};')
        if stage_id == 1:
            return system_kinetic_energy, temperature
        if stage_id == 2:
            return cell_volume, density, pressure, total_energy

    def load_forces(
            self,
            potential_table: np.ndarray,
    ):
        log_debug_info(f"Entering 'load_forces(potential_table)'")
        potential_energies = get_empty_float_scalars(self.static.particles_number)
        self.dynamic.accelerations = get_empty_vectors(self.static.particles_number)
        if self.potential.update_test:
            log_debug_info(f'update_test = True')
            self.update_list()
            self.dynamic.displacements = get_empty_vectors(self.static.particles_number)
            self.potential.update_test = False

        self.dynamic.virial = lf_cycle(
            particles_number=self.static.particles_number,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
            r_cut=self.potential.r_cut,
            potential_table=potential_table,
            potential_energies=potential_energies,
            positions=self.dynamic.positions,
            accelerations=self.dynamic.accelerations,
            cell_dimensions=self.static.cell_dimensions,
        )
        acc_mag = (self.dynamic.accelerations ** 2).sum(axis=1) ** 0.5
        log_debug_info(f'Mean and max acceleration: {acc_mag.mean()}, {acc_mag.max()}')
        potential_energy = potential_energies.sum()
        self.dynamic.displacements += (
                self.dynamic.velocities * self.model.time_step
                + self.dynamic.accelerations * self.model.time_step * self.model.time_step / 2.0
        )
        self.load_move_test()
        log_debug_info(f'Potential energy: {potential_energy};')
        log_debug_info(f"Exiting 'load_forces(potential_table)'")
        self.dynamic.potential_energy = potential_energy

    @logger_wraps()
    def update_list(self):
        self.neighbours_lists = {
            'first_neighbours': get_empty_int_scalars(self.static.particles_number),
            'last_neighbours': get_empty_int_scalars(self.static.particles_number),
            'all_neighbours': get_empty_int_scalars(100 * self.static.particles_number),
        }
        advances = get_empty_int_scalars(self.static.particles_number)
        update_list_cycle(
            rng=self.potential.r_cut + self.potential.skin,
            advances=advances,
            particles_number=self.static.particles_number,
            positions=self.dynamic.positions,
            cell_dimensions=self.static.cell_dimensions,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
        )

    @logger_wraps()
    def load_move_test(self):
        ds_1, ds_2 = 0.0, 0.0
        for i in range(self.static.particles_number):
            ds = (self.dynamic.displacements[i] ** 2).sum() ** 0.5
            if ds >= ds_1:
                ds_2 = ds_1
                ds_1 = ds
            elif ds >= ds_2:
                ds_2 = ds
        self.potential.update_test = (ds_1 + ds_2) > self.potential.skin
