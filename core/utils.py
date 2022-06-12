import numpy as np
import pandas as pd
import core.info as info
import random
import core.elements as el
from core.parameters import *
import matplotlib.pyplot as plt


# function for constructing the data frame
def weighted_lines_build(network):
    from_to = []
    paths = []
    signal_pow = []
    noise = []
    latency = []
    path_vector = []
    snr = []
    # building vector of paths
    for i in network.nodes.keys():
        for j in network.nodes.keys():
            if j != i:
                paths.append(network.find_paths(i, j))
    for i in range(0, len(paths)):
        for j in range(0, len(paths[i])):
            strng = ''
            # building the string in the format x->y
            for k in range(0, len(paths[i][j])):
                strng += paths[i][j][k]
                if k < len(paths[i][j]) - 1:
                    strng += '->'
            # saving all important data in lists to be inserted in the dataframe
            path_vector.append(strng)
            from_to.append(paths[i][j][0] + '-' + paths[i][j][len(paths[i][j]) - 1])
            current_path = list(paths[i][j])
            signal_info = info.SignalInformation(current_path)
            network.probe(signal_info)
            signal_pow.append(signal_info.average_power)
            noise.append(signal_info.noise_power)
            latency.append(signal_info.latency)
            snr.append(10 * np.log10(np.power(signal_info.isnr, -1)))
    # building the dataframe
    data = list(zip(from_to, path_vector, signal_pow, noise, snr, latency))
    df = pd.DataFrame(data, columns=['FromTo', 'Path', 'Power', 'NoisePower', 'SNR', 'Latency'])
    return df


# function for building and updating the route space dataframe
def route_space_build(network):
    labels = []
    occupancy = []
    # building the dataframe
    for i in network.weighted_lines['Path']:
        labels.append(i)
        occupancy.append(np.ones(10, dtype=int))
    df = pd.DataFrame(occupancy)
    df.insert(0, 'Path', labels, True)
    for j in network.weighted_lines['Path']:
        path_string = j
        current_path = list(j.replace('->', ''))
        elements = []
        update_rs = np.ones(10, dtype=int)
        # saving all nodes and all lines in the element vector
        for i in range(0, len(current_path) - 1):
            if i != 0:
                elements.append(current_path[i])
            elements.append(current_path[i] + current_path[i + 1])
        # compute the update array
        for i in range(0, len(elements)):
            if len(elements[i]) == 1:  # it's a node
                update_rs *= network.nodes[elements[i]].switching_matrix[elements[i - 1][0]][elements[i + 1][1]]
            elif len(elements[i]) == 2:  # it's a line
                update_rs *= network.lines[elements[i]].state
        # modify the corresponding data frame row
        row_index = df.index[df['Path'] == path_string].tolist()
        col_index = df.columns != 'Path'
        df.loc[row_index, col_index] = update_rs
    return df


# function for updating the route space dataframe after the deployment of a connection
def route_space_update(network, path):
    line_string = ''
    for index in range(0, len(path) - 1):  # loop on all the lines of the path
        line_string = path[index] + '->' + path[index + 1]
        for row in network.route_space.itertuples():  # loop on all possible paths
            if row.Path.find(line_string) != -1:
                # if it's not found the find gives back -1, while it gives back the index at which it was found (not -1)
                path_string = row.Path
                current_path = list(row.Path.replace('->', ''))
                elements = []
                update_rs = np.ones(10, dtype=int)
                # saving all nodes and all lines in the element vector
                for i in range(0, len(current_path) - 1):
                    if i != 0:
                        elements.append(current_path[i])
                    elements.append(current_path[i] + current_path[i + 1])
                # compute the update array
                for i in range(0, len(elements)):
                    if len(elements[i]) == 1:  # it's a node
                        update_rs *= network.nodes[elements[i]].switching_matrix[elements[i - 1][0]][elements[i + 1][1]]
                    elif len(elements[i]) == 2:  # it's a line
                        update_rs *= network.lines[elements[i]].state
                # modify the corresponding data frame row
                row_index = network.route_space.index[network.route_space['Path'] == path_string].tolist()
                col_index = network.route_space.columns != 'Path'
                network.route_space.loc[row_index, col_index] = update_rs
    return


# Function for computing the area of the network
def polygon_area(x, y):
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


# Function for generating the random stream of connections
def generate_random_connections(network):
    nodes = list(network.nodes.keys())
    connections = []
    for i in range(0, number_of_connections):
        new_nodes = list(nodes)
        strt = random.choice(new_nodes)
        new_nodes.remove(strt)
        end = random.choice(new_nodes)
        connections.append(el.Connection(strt, end))
    return connections


# Function for computing the capacity and average bit rate
def compute_network_capacity_and_avg_bit_rate(connections):
    total_capacity = np.nansum([connections[k].bit_rate for k in range(0, len(connections))])
    avg_bit_rate = total_capacity / len(connections)
    return total_capacity, avg_bit_rate


# Function for automatic plots
def plot_histogram(figure_num, list_data, nbins, edge_color, color, label, title='', ylabel='', xlabel='',
                   bbox_to_anchor=None, loc=None, bottom=None, nan_display=False):
    if nan_display:
        list_data = list(np.nan_to_num(list_data))  # replace NaN with 0

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=bottom)
    ax.set_axisbelow(True)
    plt.grid()
    plt.hist(list_data, bins=nbins, edgecolor=edge_color, color=color, label=label)
    plt.title(title)
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, framealpha=1)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)


# Generate fresh traffic matrix as a dictionary
def generate_traffic_matrix(network, M):
    traffic_matrix = {}
    for node in network.nodes:
        traffic_matrix[node] = {}
    for input_n in network.nodes:
        for output_n in network.nodes:
            if output_n == input_n:
                traffic_matrix[input_n][output_n] = 0
            elif output_n != input_n:
                traffic_matrix[input_n][output_n] = 100 * M
    return traffic_matrix
