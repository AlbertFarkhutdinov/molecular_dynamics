import matplotlib.pyplot as plt
import numpy as np


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
