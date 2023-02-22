from typing import Optional, Union

import numpy as np

from ovito.data import DataCollection
from ovito.plugins.ParticlesPython import (
    Particles,
    ParticleType,
    CoordinationAnalysisModifier,
    VoronoiAnalysisModifier
)
from ovito.plugins.StdObjPython import SimulationCell
from ovito.pipeline import Pipeline, StaticSource


class OvitoProcessor:

    def __init__(
            self,
            positions: np.ndarray,
            cell_dimensions: np.ndarray,
    ) -> None:
        self.positions = positions
        self.cell_dimensions = cell_dimensions
        self.pipeline = self.get_pipeline()

    def get_particles(self) -> Particles:
        particles = Particles()
        particles.create_property('Position', data=self.positions)
        return particles

    def get_particles_types(self, particles: Optional[Particles] = None):
        _particles = particles or self.get_particles()
        type_prop = _particles.create_property('Particle Type')
        type_prop.types.append(
            ParticleType(id=1, name='Ar', color=(0.0, 1.0, 0.0))
        )
        for i in range(self.positions.shape[0]):
            type_prop[i] = 1
        return type_prop

    def get_cell(self) -> SimulationCell:
        cell = SimulationCell(pbc=(True, True, True))
        cell[...] = [
            [self.cell_dimensions[0], 0, 0, -self.cell_dimensions[0] / 2],
            [0, self.cell_dimensions[1], 0, -self.cell_dimensions[1] / 2],
            [0, 0, self.cell_dimensions[2], -self.cell_dimensions[2] / 2],
        ]
        return cell

    def get_pipeline(self) -> Pipeline:
        data_collection = DataCollection()
        particles = self.get_particles()
        self.get_particles_types(particles=particles)
        cell = self.get_cell()
        data_collection.objects.append(particles)
        data_collection.objects.append(cell)
        return Pipeline(source=StaticSource(data=data_collection))

    def get_voronoi_diagram(self) -> DataCollection:
        self.pipeline.modifiers.append(
            VoronoiAnalysisModifier(
                compute_indices=True,
                use_radii=True,
                edge_threshold=0.1,
            )
        )
        return self.pipeline.compute().particles

    def get_voronoi_indices(
            self,
            voronoi_diagram: Optional[DataCollection] = None,
    ) -> np.ndarray:
        _voronoi_diagram = voronoi_diagram or self.get_voronoi_diagram()
        return np.ascontiguousarray(_voronoi_diagram['Voronoi Index'])

    def get_voronoi_volumes(
            self,
            voronoi_diagram: Optional[DataCollection] = None,
    ) -> np.ndarray:
        _voronoi_diagram = voronoi_diagram or self.get_voronoi_diagram()
        return np.ascontiguousarray(_voronoi_diagram['Atomic Volume'])

    def get_voronoi_coordination(
            self,
            voronoi_diagram: Optional[DataCollection] = None,
    ) -> np.ndarray:
        _voronoi_diagram = voronoi_diagram or self.get_voronoi_diagram()
        return np.ascontiguousarray(_voronoi_diagram['Coordination'])

    def get_rdf(
            self,
            is_coordination_returned: bool = False,
            layer_thickness: float = 0.01,
    ) -> Union[
         tuple[np.ndarray, np.ndarray],
         tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        half_max = self.cell_dimensions[0] / 2
        self.pipeline.modifiers.append(
            CoordinationAnalysisModifier(
                number_of_bins=int(half_max / layer_thickness) - 1,
                cutoff=half_max,
            )
        )
        data_collection = self.pipeline.compute()
        radii, rdf = data_collection.tables['coordination-rdf'].xy().T
        if is_coordination_returned:
            coordination = np.ascontiguousarray(
                data_collection.particles['Coordination']
            )
            return radii, rdf, coordination
        return radii.round(2), rdf
