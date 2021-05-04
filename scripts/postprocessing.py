import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BasePP:

    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.data = pd.DataFrame()

    def rename_column(self, column_name, setup_number):
        self.data = self.data.rename(
            columns={column_name: f'setup_{setup_number}'}
        )


class BaseSpacePP(BasePP):

    def __init__(self, path_to_data):
        super().__init__(path_to_data)
        self.data = pd.DataFrame(columns=['radius'])


class BaseTimePP(BasePP):

    def __init__(self, path_to_data):
        super().__init__(path_to_data)
        self.data = pd.DataFrame(columns=['time'])
        self.filename_prefix = os.path.join(path_to_data, 'transport_')

    def append(self, filename_postfix, column_name):
        self.data = self.data.merge(
            right=pd.read_csv(
                f'{self.filename_prefix}{filename_postfix}.csv',
                sep=';',
            )[['time', column_name]],
            how='outer',
            on='time',
        )


class RadialDistributionFunctionPP(BaseSpacePP):

    def __init__(self, path_to_data):
        super().__init__(path_to_data)
        self.filename_prefix = os.path.join(path_to_data, 'rdf_')

    def append(self, filename_postfix):
        self.data = self.data.merge(
            right=pd.read_csv(
                f'{self.filename_prefix}{filename_postfix}.csv',
                sep=';',
            ),
            how='outer',
            on='radius',
        )


class MeanSquaredDisplacementPP(BaseTimePP):

    def append(self, filename_postfix, column_name='msd'):
        super().append(
            filename_postfix=filename_postfix,
            column_name=column_name,
        )


class VelocityAutoCorrelationFunctionPP(BaseTimePP):

    def append(self, filename_postfix, column_name='vaf'):
        super().append(
            filename_postfix=filename_postfix,
            column_name=column_name,
        )


class EinsteinDiffusionPP(BaseTimePP):

    def append(self, filename_postfix, column_name='einstein_diffusion'):
        super().append(
            filename_postfix=filename_postfix,
            column_name=column_name,
        )


class GKDiffusionPP(BaseTimePP):

    def append(self, filename_postfix, column_name='gk_diffusion'):
        super().append(
            filename_postfix=filename_postfix,
            column_name=column_name,
        )


class PostProcessor:

    def __init__(
            self,
            path_to_data: str,
            path_to_plots: str,
            plot_filename_postfix: str,
            setups: List[Dict[str, float]],
    ):
        self.path_to_data = path_to_data
        self.path_to_plots = path_to_plots
        self.plot_filename_postfix = plot_filename_postfix
        self.setups = setups
        try:
            os.mkdir(path_to_plots)
        except FileExistsError:
            pass
        self.rdf = RadialDistributionFunctionPP(path_to_data)
        self.msd = MeanSquaredDisplacementPP(path_to_data)
        self.vaf = VelocityAutoCorrelationFunctionPP(path_to_data)
        self.einstein_diffusion = EinsteinDiffusionPP(path_to_data)
        self.gk_diffusion = GKDiffusionPP(path_to_data)
        self.diffusion_coefficients = np.zeros(len(setups))
        self.post_init()

    def post_init(self):
        for i, setup in enumerate(self.setups):
            filename_postfix = self.get_filename_postfix(setup)
            self.rdf.append(filename_postfix=filename_postfix)
            self.msd.append(filename_postfix=filename_postfix)
            self.vaf.append(filename_postfix=filename_postfix)
            self.einstein_diffusion.append(
                filename_postfix=filename_postfix
            )
            self.gk_diffusion.append(filename_postfix=filename_postfix)
            self.diffusion_coefficients[i] = self.einstein_diffusion.data[
                'einstein_diffusion'
            ].values[-1]
            self.rdf.rename_column(column_name='rdf', setup_number=i)
            self.msd.rename_column(column_name='msd', setup_number=i)
            self.vaf.rename_column(
                column_name='vaf',
                setup_number=i,
            )
            self.einstein_diffusion.rename_column(
                column_name='einstein_diffusion',
                setup_number=i,
            )
            self.gk_diffusion.rename_column(
                column_name='gk_diffusion',
                setup_number=i,
            )

    def save_plot(self, filename):
        plt.savefig(
            os.path.join(self.path_to_plots, filename)
        )

    def get_parameters_filename(self):
        os.path.join(self.path_to_data, 'system_parameters.csv')

    @staticmethod
    def get_filename_postfix(setup):
        postfix = ''
        if setup['temperature'] is not None:
            postfix += f'T_{setup["temperature"]:.5f}_'
        if setup['pressure'] is not None:
            postfix += f'P_{setup["pressure"]:.5f}_'
        if setup['heating_velocity'] is not None:
            postfix += f'HV_{setup["heating_velocity"]:.5f}_'
        return postfix

    def plot_setups(
            self,
            x, y,
            x_label, y_label,
            plot_filename_prefix, **limits,
    ):
        for i, setup in enumerate(self.setups):
            plt.plot(
                x,
                y(i),
                label=fr'T = {setup["temperature"]:.5f} $\epsilon / k_B$'
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(
            left=limits.get('left'),
            right=limits.get('right'),
        )
        plt.ylim(
            bottom=limits.get('bottom'),
            top=limits.get('top'),
        )
        plt.legend()
        self.save_plot(
            filename=f'{plot_filename_prefix}_{self.plot_filename_postfix}.png'
        )

    def plot_rdf(self, **limits):
        self.plot_setups(
            x=self.rdf.data['radius'],
            y=lambda x: (
                    self.rdf.data[f'setup_{x}']
                    + 1 * (len(self.setups) - (x + 1))
            ),
            x_label=r'$r$, $\sigma $',
            y_label='$g(r)$',
            plot_filename_prefix='rdf',
            **limits,
        )

    def plot_msd(self, **limits):
        self.plot_setups(
            x=self.msd.data['time'],
            y=lambda x: self.msd.data[f'setup_{x}'],
            x_label=r'$t$, $\tau $',
            y_label=r'$\left<\Delta r^2(t)\right>$, $\sigma^2$',
            plot_filename_prefix='msd',
            **limits,
        )

    def plot_vaf(self, **limits):
        self.plot_setups(
            x=self.vaf.data['time'],
            y=lambda x: self.vaf.data[f'setup_{x}'],
            x_label=r'$t$, $\tau $',
            y_label=r'$\Psi(t)$, $\sigma^2$',
            plot_filename_prefix='vaf',
            **limits,
        )

    def plot_einstein_diffusion(self, **limits):
        self.plot_setups(
            x=self.einstein_diffusion.data['time'],
            y=lambda x: self.einstein_diffusion.data[f'setup_{x}'],
            x_label=r'$t$, $\tau $',
            y_label=r'$D_E(t)$, $\sigma^2 / \tau$',
            plot_filename_prefix='diffusion_einstein',
            **limits,
        )

    def plot_gk_diffusion(self, **limits):
        self.plot_setups(
            x=self.gk_diffusion.data['time'],
            y=lambda x: self.gk_diffusion.data[f'setup_{x}'],
            x_label=r'$t$, $\tau $',
            y_label=r'$D_{GK}(t)$, $\sigma^2 / \tau$',
            plot_filename_prefix='diffusion_gk',
            **limits,
        )

    def plot_diffusion(self, **limits):
        plt.scatter(
            np.array([setup['temperature'] for setup in self.setups]),
            self.diffusion_coefficients,
        )
        plt.xlabel(r'$T$, $\epsilon / k_B$')
        plt.ylabel(r'$D$, $\sigma^2 / \tau$')
        plt.xlim(
            left=limits.get('left'),
            right=limits.get('right'),
        )
        plt.ylim(
            bottom=limits.get('bottom'),
            top=limits.get('top'),
        )
        self.save_plot(
            filename=f'diffusion_{self.plot_filename_postfix}.png'
        )

    def get_system_parameters(self):
        system_parameters = pd.DataFrame()

        for parameters_filename in [
            item
            for item in os.listdir(self.path_to_data)
            if item.startswith('system_parameters')
        ]:
            system_parameters = pd.concat(
                [
                    system_parameters,
                    pd.read_csv(
                        os.path.join(
                            self.path_to_data,
                            parameters_filename,
                        ),
                        sep=';',
                    )
                ],
                ignore_index=True
            )

        for column in system_parameters.columns:
            system_parameters[column] = system_parameters[column].round(5)
        return system_parameters

    def plot_system_parameters(
            self,
            system_parameters,
            column_names,
            y_label,
            **limits,
    ):

        times = (system_parameters.index + 1) * 0.005
        for column_name in column_names:
            plt.scatter(
                times,
                system_parameters[column_name],
                s=1,
                label=column_name.replace('_', ' ').capitalize(),
            )
        plt.xlabel(r'$t$, $\tau$')
        plt.ylabel(y_label)
        plt.xlim(
            left=times.min(),
            right=times.max(),
        )
        plt.ylim(
            bottom=limits.get('bottom'),
            top=limits.get('top'),
        )
        if len(column_names) > 1:
            plt.legend(markerscale=10)
        prefix = column_names[0] if len(column_names) == 1 else 'all'
        self.save_plot(
            f'{prefix}_{self.plot_filename_postfix}.png'
        )

