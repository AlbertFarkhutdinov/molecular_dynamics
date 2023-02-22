from math import log10
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pretty_repr import RepresentableObject


FONTSIZE = 20
CUSTOM_STYLE = {
    'axes.labelpad': 0,
    'lines.linewidth': 2,
    'axes.linewidth': 2,
    'figure.dpi': 300,
    'font.family': 'Times New Roman',
    'figure.figsize': [i / 2.54 for i in (15, 15)],
    'mathtext.fontset': 'stix',
    'mathtext.it': 'Times New Roman',
    'xtick.labelsize': FONTSIZE,
    'ytick.labelsize': FONTSIZE,
    'legend.fontsize': FONTSIZE,
    'axes.titlesize': FONTSIZE,
    'axes.labelsize': FONTSIZE,
    'font.size': FONTSIZE,
}


def plot_rdf(file_name: str):
    rdf_data = []
    with open('rdf_file.txt', mode='r', encoding='utf8') as file:
        for line in file:
            rdf_data.append(np.array(line.rstrip().split()).astype(np.float))

    rdf_data = np.array(rdf_data).transpose()
    plt.plot(*rdf_data)
    plt.xlabel(r'r/$\sigma$')
    plt.ylabel('g(r)')
    plt.ylim(bottom=0, top=100)
    plt.savefig(file_name)


def get_temperature_legend(
        temperature: float,
        accuracy: Optional[int],
) -> str:
    _temp = temperature if accuracy is None else round(temperature, accuracy)
    if _temp == 0.0:
        _exp = int(round(log10(temperature)))
        _temp = round(temperature, -_exp) / 10 ** _exp
        return fr'$T$ = $10^-{{}}^{abs(_exp)} \epsilon / k_B$'
    return fr'$T$ = {_temp} $\epsilon / k_B$'


class Labels(RepresentableObject):

    def __init__(self, x_label: str, y_label: str) -> None:
        self.x_label = self.process_label(x_label)
        self.y_label = self.process_label(y_label)

    @staticmethod
    def process_label(label: str) -> str:
        return {
            'radius': r'$r$, $\sigma$',
            'temperature': r'$T$, $\epsilon / k_B$',
            'rdf': r'$g(r)$',
            'cooling_rate': r'$\gamma$, $\varepsilon / k_B\tau$',
            'diffusion': r'$D$, $\sigma^2 / \tau$',
            'time': r'$t$, $\tau$',
        }.get(label, label)


class Plotter(RepresentableObject):

    def __init__(
            self,
            path_to_plots: Optional[str] = None,
            size: Optional[tuple[int, int]] = None,
            limits: Optional[dict[str, float]] = None,
            labels: Optional[tuple[str, str]] = None,
    ):
        plt.style.use('seaborn')
        plt.rcParams.update(CUSTOM_STYLE)
        self.path_to_plots = path_to_plots
        if self.path_to_plots:
            try:
                os.mkdir(self.path_to_plots)
            except FileExistsError:
                pass
        self.fig, self.ax = plt.subplots(figsize=size)
        if limits is not None:
            self.ax.set_xlim(
                left=limits.get('left'),
                right=limits.get('right'),
            )
            self.ax.set_ylim(
                bottom=limits.get('bottom'),
                top=limits.get('top'),
            )
        if labels is not None:
            _labels = Labels(x_label=labels[0], y_label=labels[1])
            self.ax.set_xlabel(_labels.x_label)
            self.ax.set_ylabel(_labels.y_label)

    def get_legend(self, *args, **kwargs):
        self.ax.legend(*args, **kwargs)

    def set_title(self, title: str):
        self.ax.set_title(title)

    def set_major_locators(self, x_step: float, y_step: float):
        self.ax.xaxis.set_major_locator(plt.MultipleLocator(x_step))
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(y_step))

    def set_minor_locators(self, x_step: float, y_step: float):
        self.ax.xaxis.set_minor_locator(plt.MultipleLocator(x_step))
        self.ax.yaxis.set_minor_locator(plt.MultipleLocator(y_step))

    def save_plot(self, filename):
        if self.path_to_plots:
            formats = ('.png', '.eps')
            _filename = filename
            for fmt in formats:
                if filename.endswith(fmt):
                    _filename = _filename.removesuffix(fmt)
            for fmt in formats:
                plt.savefig(
                    os.path.join(self.path_to_plots, f'{_filename}{fmt}'),
                    bbox_inches='tight',
                    pad_inches=0,
                )
