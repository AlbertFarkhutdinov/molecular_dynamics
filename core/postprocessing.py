from copy import deepcopy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pretty_repr import RepresentableObject
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from drawing import Plotter, get_temperature_legend


def get_filename_postfix(setup):
    postfix = ''
    if setup['temperature'] is not None:
        postfix += f'T_{setup["temperature"]:.5f}_'
    if setup['pressure'] is not None:
        postfix += f'P_{setup["pressure"]:.5f}_'
    if setup['heating_velocity'] is not None:
        postfix += f'HV_{setup["heating_velocity"]:.5f}_'
    return postfix


class BasePP(RepresentableObject):

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

    def get_rdf_table(self):
        rdf_table = deepcopy(self.data)
        rdf_table.index = rdf_table['radius']
        rdf_table = rdf_table.drop(columns=['radius'])
        rdf_table = rdf_table.T[::-1]
        return rdf_table


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
            setups: list[dict[str, float]],
    ):
        self.path_to_data = path_to_data
        self.path_to_plots = path_to_plots
        self.plot_filename_postfix = plot_filename_postfix
        self.setups = setups
        self.rdf = RadialDistributionFunctionPP(path_to_data)
        self.msd = MeanSquaredDisplacementPP(path_to_data)
        self.vaf = VelocityAutoCorrelationFunctionPP(path_to_data)
        self.einstein_diffusion = EinsteinDiffusionPP(path_to_data)
        self.gk_diffusion = GKDiffusionPP(path_to_data)
        self.diffusion_coefficients = np.zeros(len(setups))
        self.post_init()
        self.system_parameters = self.get_system_parameters()

    def post_init(self):
        for i, setup in enumerate(self.setups):
            filename_postfix = get_filename_postfix(setup)
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

    # def save_plot(self, filename):
    #     _filename = filename
    #     if filename.endswith('.png'):
    #         _filename = _filename.removesuffix('.png')
    #     if filename.endswith('.eps'):
    #         _filename = _filename.removesuffix('.eps')
    #     plt.savefig(os.path.join(self.path_to_plots, f'{_filename}.png'))
    #     plt.savefig(os.path.join(self.path_to_plots, f'{_filename}.eps'))

    def get_parameters_filename(self):
        os.path.join(self.path_to_data, 'system_parameters.csv')

    def plot_setups(
            self,
            x, y,
            x_label, y_label,
            plot_filename_prefix,
            figsize,
            y_scale="linear",
            title=None,
            filename_postfix='',
            shown=None,
            accuracy=5,
            colors=None,
            **limits,
    ):
        plotter = Plotter(
            path_to_plots=self.path_to_plots,
            size=figsize,
            limits=dict(limits),
            labels=(x_label, y_label),
        )
        for i, setup in enumerate(self.setups):
            if shown is not None and i not in shown:
                continue
            plotter.ax.plot(
                x,
                y(i),
                label=get_temperature_legend(setup["temperature"], accuracy),
                color=colors[i] if colors else None,
            )
        plotter.ax.set_yscale(y_scale)
        plotter.ax.get_legend()
        plotter.ax.set_title(title)
        plotter.save_plot(
            filename=(
                f'{plot_filename_prefix}'
                f'_{self.plot_filename_postfix}'
                f'_{filename_postfix}.png'
            )
        )

    def plot_rdf(
            self,
            shift=0,
            y_scale='linear',
            **limits,
    ):
        self.plot_setups(
            x=self.rdf.data['radius'],
            y=lambda x: (
                    self.rdf.data[f'setup_{x}']
                    + shift * (len(self.setups) - (x + 1))
            ),
            x_label=r'$r$, $\sigma $',
            y_label='$g(r)$',
            y_scale=y_scale,
            plot_filename_prefix='rdf',
            **limits,
        )

    def plot_msd(
            self,
            y_scale="linear",
            **limits,
    ):
        self.plot_setups(
            x=self.msd.data['time'],
            y=lambda x: self.msd.data[f'setup_{x}'],
            x_label=r'$t$, $\tau $',
            y_label=r'$\left<\Delta r^2(t)\right>$, $\sigma^2$',
            y_scale=y_scale,
            plot_filename_prefix='msd',
            **limits,
        )

    def plot_vaf(
            self,
            y_scale="linear",
            **limits,
    ):
        self.plot_setups(
            x=self.vaf.data['time'],
            y=lambda x: self.vaf.data[f'setup_{x}'],
            x_label=r'$t$, $\tau $',
            y_label=r'$\Psi(t)$, $\sigma^2$',
            y_scale=y_scale,
            plot_filename_prefix='vaf',
            **limits,
        )

    def plot_einstein_diffusion(
            self,
            y_scale="linear",
            **limits,
    ):
        self.plot_setups(
            x=self.einstein_diffusion.data['time'],
            y=lambda x: self.einstein_diffusion.data[f'setup_{x}'],
            x_label=r'$t$, $\tau $',
            y_label=r'$D_E(t)$, $\sigma^2 / \tau$',
            y_scale=y_scale,
            plot_filename_prefix='diffusion_einstein',
            **limits,
        )

    def plot_gk_diffusion(
            self,
            y_scale="linear",
            **limits,
    ):
        self.plot_setups(
            x=self.gk_diffusion.data['time'],
            y=lambda x: self.gk_diffusion.data[f'setup_{x}'],
            x_label=r'$t$, $\tau $',
            y_label=r'$D_{GK}(t)$, $\sigma^2 / \tau$',
            y_scale=y_scale,
            plot_filename_prefix='diffusion_gk',
            **limits,
        )

    def plot_diffusion(
            self,
            y_scale="linear",
            **limits,
    ):
        plotter = Plotter(
            path_to_plots=self.path_to_plots,
            limits=dict(**limits),
            labels=('temperature', 'diffusion')
        )
        plotter.ax.scatter(
            np.array([setup['temperature'] for setup in self.setups]),
            self.diffusion_coefficients,
        )
        plt.yscale(y_scale)
        plotter.save_plot(
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
            column_names,
            y_label,
            y_scale="linear",
            file_name_prefix=None,
            **limits,
    ):
        plotter = Plotter(
            path_to_plots=self.path_to_plots,
            labels=('time', y_label),
            limits=dict(limits),
        )
        times = (self.system_parameters.index + 1) * 0.005
        for column_name in column_names:
            plt.scatter(
                times,
                self.system_parameters[column_name],
                s=1,
                label=column_name.replace('_', ' ').capitalize(),
            )
        plotter.ax.set_xlim(
            left=limits.get('left', times.min()),
            right=limits.get('right', times.max()),
        )
        plt.yscale(y_scale)
        prefix = column_names[0]
        if len(column_names) > 1:
            plt.legend(markerscale=10)
            if file_name_prefix is None:
                raise ValueError('`file_name_prefix` is absent.')
            prefix = file_name_prefix
        plotter.save_plot(
            filename=f'{prefix}_{self.plot_filename_postfix}.png'
        )

    def get_column_diff(self, column_name: str):
        _values = self.system_parameters[column_name].values
        _diff = np.zeros(_values.size)
        _diff[1:] = _values[1:] - _values[:-1]
        return _diff

    def get_integrated_column(self, column_name: str):
        _diff = self.__getattribute__(f'get_{column_name}_diff')()
        _integrated = np.zeros(self.system_parameters.shape[0])
        for i in range(1, _integrated.size):
            _integrated[i] = _integrated[i - 1] + _diff[i]
        self.system_parameters[column_name] = _integrated
        return _integrated

    def get_entropy_diff(self):
        return (
                       self.get_column_diff('total_energy')
                       + self.system_parameters['pressure'].values
                       * self.get_column_diff('volume')
               ) / self.system_parameters['temperature'].values

    def __get_pdv(self):
        return (
                self.system_parameters['pressure'].values
                * self.get_column_diff('volume')
        )

    def __get_vdp(self):
        return (
                self.system_parameters['volume'].values
                * self.get_column_diff('pressure')
        )

    def __get_tds(self):
        return (
                self.system_parameters['temperature'].values
                * self.get_entropy_diff()
        )

    def __get_sdt(self):
        return (
                self.system_parameters['entropy'].values
                * self.get_column_diff('temperature')
        )

    def get_enthalpy_diff(self):
        return self.__get_tds() + self.__get_vdp()

    def get_free_energy_diff(self):
        return -self.__get_pdv() - self.__get_sdt()

    def get_gibbs_energy_diff(self):
        return -self.__get_sdt() + self.__get_vdp()

    def get_enthalpy(self):
        return self.get_integrated_column('enthalpy')

    def get_entropy(self):
        # TODO entropy is negative after cooling to T = 0
        return self.get_integrated_column('entropy')

    def get_free_energy(self):
        if 'entropy' not in self.system_parameters.columns:
            self.get_entropy()
        return self.get_integrated_column('free_energy')

    def get_gibbs_energy(self):
        if 'enthalpy' not in self.system_parameters.columns:
            self.get_enthalpy()
        return self.get_integrated_column('gibbs_energy')


class RegressionRDF:

    def __init__(
            self,
            post_processor,
            setups,
            test_temperatures=None,
            test_heating_velocities=None,
    ):
        self.post_processor = post_processor
        self.rdf_table = self.post_processor.rdf.get_rdf_table()
        self.setups = setups
        self.temperatures = self.get_temperatures_from_indices()
        self.heating_velocities = self.get_hv_from_indices()
        self.test_temperatures = test_temperatures
        self.test_heating_velocities = test_heating_velocities

    def get_hv_from_indices(self):
        return np.array(
            [
                self.setups[value]['heating_velocity']
                for value
                in self.rdf_table.index.str[6:].values.astype(np.int32)
            ]
        )

    def get_temperatures_from_indices(self):
        return np.array(
            [
                self.setups[value]['temperature']
                for value
                in self.rdf_table.index.str[6:].values.astype(np.int32)
            ]
        )

    @staticmethod
    def get_linear_coefficients(x_train, y_train):
        regression = LinearRegression()
        regression.fit(x_train.reshape((x_train.size, 1)), y_train)
        return regression.coef_[0], regression.intercept_

    def fit_data(self, y_train):
        k, b = self.get_linear_coefficients(
            x_train=self.temperatures,
            y_train=y_train,
        )
        fitted_data = k * self.temperatures + b
        error = mean_squared_error(y_train, fitted_data, squared=False)
        return fitted_data, error

    def plot_linear_regression(
            self,
            y_train,
            fitted_data, error,
            k, b, r,
            is_saved,
            figsize,
    ):
        _, ax = plt.subplots(figsize=figsize)
        ax.errorbar(
            self.temperatures,
            y_train,
            yerr=error,
            fmt='o',
            label=fr'$g(r = {r}\sigma)$',
        )
        ax.plot(
            self.temperatures,
            fitted_data,
            label=fr'{k:.4f}$T$ {"+" if b >= 0 else "-"} {abs(b):.4f}'
        )
        ax.legend()
        ax.set_xlabel(r'Temperature, $\epsilon / k_B$')
        ax.set_ylabel('$g(r)$')
        if is_saved:
            self.post_processor.save_plot(
                f'rdf_point_{self.post_processor.plot_filename_postfix}.png'
            )
        plt.show()
        plt.close()

    def get_fitted_column(self, y_train):
        k, b = self.get_linear_coefficients(
            x_train=self.temperatures,
            y_train=y_train,
        )
        fitted_data, error = self.fit_data(y_train)
        return k, b, fitted_data, error

    def run_linear_regressions(
            self,
            is_printed: bool,
            is_plotted: bool,
            figure_size,
    ):
        predicted_points = {key: [] for key in self.test_temperatures}
        for column in self.rdf_table.columns:
            if not (self.rdf_table[column] == 0).all():
                y = self.rdf_table[column]
                k, b, fitted_data, error = self.get_fitted_column(y_train=y)
                for key, _ in predicted_points.items():
                    new_point = k * key + b
                    predicted_points[key].append(new_point)

                if is_printed:
                    print(
                        f'r = {column:.2f}; '
                        f'k = {k:8.5f}; '
                        f'b = {b:8.5f}; '
                        f'RMSE = {error:.5f}'
                    )

                if is_plotted:
                    self.plot_linear_regression(
                        y_train=y,
                        fitted_data=fitted_data,
                        error=error,
                        k=k, b=b, r=column,
                        is_saved=False,
                        figsize=figure_size,
                    )
            else:
                for key, _ in predicted_points.items():
                    predicted_points[key].append(0.0)
        return predicted_points
