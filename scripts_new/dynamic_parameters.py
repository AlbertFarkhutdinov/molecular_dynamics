# # import os
# #
# # import numpy as np
# #
# # from scripts_new.constants import PATH_TO_DATA
# # from scripts_new.helpers import get_date, get_empty_vectors
# # from scripts_new.numba_procedures import get_radius_vectors
# #
# #
# # class SystemDynamicParameters:
# #
# #     def __init__(self):
# #         self.first_positions = get_empty_vectors(self.particles_number)
# #         self.first_velocities = get_empty_vectors(self.particles_number)
#         self.interparticle_vectors = np.zeros(
#             (self.particles_number, self.particles_number, 3),
#             dtype=np.float32,
#         )
#         self.interparticle_distances = np.zeros(
#             (self.particles_number, self.particles_number),
#             dtype=np.float32,
#         )
#
#     def calculate_interparticle_vectors(self):
#         self.interparticle_vectors, self.interparticle_distances = get_radius_vectors(
#             radius_vectors=self.interparticle_vectors,
#             positions=self.positions,
#             cell_dimensions=self.cell_dimensions,
#             distances=self.interparticle_distances,
#         )
# #
#     def save_xyz_file(self, filename: str, step: int):
#         _path = os.path.join(
#             os.path.join(
#                 PATH_TO_DATA,
#                 get_date(),
#             ),
#             filename,
#         )
#         _mode = 'a' if os.path.exists(_path) else 'w'
#         with open(_path, mode=_mode, encoding='utf8') as file:
#             file.write(
#                 f'{self.particles_number}\n')
#             file.write(f'step: {step} columns: name, pos cell:')
#             file.write(f"{','.join(self.cell_dimensions.astype(str))}\n")
#             for position in self.positions:
#                 file.write('A')
#                 for i in range(3):
#                     file.write(f'{position[i]:15.6f}')
#                 file.write('\n')
# #
# #
# # if __name__ == '__main__':
# #     stat_par = {
# #         "init_type": 1,
# #         "lattice_constant": 1.75,
# #         "particles_number": [7, 7, 7],
# #         "crystal_type": "гцк"
# #     }
# #     dynamic = SystemDynamicParameters()
# #     dynamic.save_xyz_file('test.xyz', step=1)
# #     dynamic.save_xyz_file('test.xyz', step=2)
