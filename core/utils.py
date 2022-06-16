import numpy as np
import json
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
def route_space_build(network):  # update route space analyzing the state of each line
    data_f = pd.DataFrame(columns=['Path', 'Availability'])
    # define lists for dataframe
    availabilities_dataframe = []
    titles = []
    # take the paths from weighted paths and analyze the availability of channels
    for path in network.weighted_lines['Path']:  # extracts each path from weighted paths
        titles.append(path)  # save each path label for route space definition
        path = path.replace('->', '')  # removes arrows
        availability_per_channel = np.ones(number_of_channels, dtype='int')
        # Each time will be updated this availability along path
        # if len(path)==2:
        #     availability_per_channel = self.lines[path].state
        # else:
        start = True
        previous_node = ''
        while len(path) > 1:  # as propagate does, let's analyze path until there is at least a line
            if start:  # if it is the first node, let's define availability only by line states
                availability_per_channel = np.array(network.lines[path[:2]].state)
                start = False
            else:
                # switching matrix element for current node
                block = np.array(network.nodes[path[0]].switching_matrix[previous_node][path[1]])
                # array of line states for current line
                line_state = np.array(network.lines[path[:2]].state)
                # new state array
                availability_per_channel *= block * line_state
            # update path to go on the path and have next line
            previous_node = path[0]
            path = path[1:]
        # save the availabilities of the channels
        availabilities_dataframe.append(availability_per_channel)
    # produce route space dataframe
    data_f['Path'] = titles
    data_f['Availability'] = availabilities_dataframe
    return data_f


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
        [strt, end] = random_node_pair(nodes)
        connections.append(el.Connection(strt, end))
    return connections


# Function for generating pairs of nodes
def random_node_pair(nodes):
    new_nodes = list(nodes)
    strt = random.choice(new_nodes)
    new_nodes.remove(strt)
    end = random.choice(new_nodes)
    return strt, end


# Function for computing the capacity and average bit rate
def compute_network_capacity_and_avg_bit_rate(connections):
    total_capacity = np.nansum([connections[k].bit_rate for k in range(0, len(connections))])
    avg_bit_rate = total_capacity / len(connections)
    return total_capacity, avg_bit_rate


# Function for counting the number of occurrencies of the various lines
def count_line_usage(connections, net_lines):
    list_inputs = [connection.input for connection in connections]
    list_outputs = [connection.output for connection in connections]
    lines = [line for line in net_lines.keys()]
    occurrences_in = [0 for i in range(0, len(lines))]
    occurrences_out = [0 for i in range(0, len(lines))]
    for ind in list_inputs:
        occurrences_in[lines.index(ind)] += 1
    for ind in list_outputs:
        occurrences_out[lines.index(ind)] += 1
    return lines, occurrences_in, occurrences_out


# Function for computing the number of successful connections and blocking events
def compute_successful_blocking_events(connections):
    blocking_events = 0
    successful_connections = 0
    for k in range(0, len(connections)):
        if connections[k].connection_status:
            blocking_events += 1
        elif not connections[k].connection_status:
            successful_connections += 1
    return successful_connections, blocking_events


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


# Generate fresh traffic matrix as a dictionary. Each entry represents an input node and it's a dictionary whose entries
# are the connected nodes (themselves dictionaries containing the traffic request in Gbps).
def generate_traffic_matrix(network, M):
    traffic_matrix = {}
    for node in network.nodes:
        traffic_matrix[node] = {}
    for input_n in network.nodes:
        for output_n in network.nodes:
            if output_n != input_n:
                traffic_matrix[input_n][output_n] = 100 * M
    return traffic_matrix


# Free lines and switching matrix
def free_lines_and_switch_matrix(file, network):
    for lbl in network.lines.keys():
        network.lines[lbl].state = [1] * number_of_channels  # at the end of the stream, the lines are again free
        with open(file) as json_file:
            nodes_from_file = json.load(json_file)
        # rewrite the switching matrix
        for i, j in nodes_from_file.items():
            for k in j:
                if k == 'switching_matrix':
                    network.nodes[i].switching_matrix = nodes_from_file[i][k]
    network.route_space = route_space_build(network)


# check if traffic matrix is saturated (True = saturated, False = some traffic can still be allocated)
# def is_matrix_saturated(matrix):
#     saturated = True
#     for i in matrix:
#         for j in matrix[i]:
#             if matrix[i][j] != 0 and matrix[i][j] != np.inf:
#                 saturated = False
#                 return saturated
#     return saturated


def plot_bar(figure_num=None, list_data=None, x_ticks=None, edge_color='k', color=None, label=None, title='', ylabel='',
             xlabel='', savefig_path=None, bbox_to_anchor=None, loc=None, bottom=None, NaN_display=False, myalpha=None,
             remove_y_ticks=None):
    if NaN_display:
        list_data = list(np.nan_to_num(list_data))  # replace NaN with 0

    x = np.arange(len(x_ticks)) if x_ticks else 1

    # fig = plt.figure(figure_num)
    fig, ax = plt.subplots()
    # ax = plt.gca()
    fig.subplots_adjust(bottom=bottom)
    for index in range(0, len(list_data)):
        width = 0.15
        x_i = x + width * (index + 0.5 - len(list_data) / 2)
        plt.bar(x=x_i, width=width, height=list_data[index], edgecolor=edge_color,
                color=color[index] if color else None, alpha=myalpha)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    ax.set_axisbelow(True)
    plt.grid()
    if x_ticks:
        # also define the labels we'll use (note this MUST have the same size as `xticks`!)
        xtick_labels = ['M='+str(M) for M in x_ticks]
        # add the ticks and labels to the plot
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
    else:
        plt.tick_params(axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)
    if remove_y_ticks:
        plt.tick_params(axis='y',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        left=False, right=False, labelleft=False)
    ax.legend(labels=label, bbox_to_anchor=bbox_to_anchor, loc=loc)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)

    # savefig_path = None ##### AVOIDED SAVE AS DEBUG
    if savefig_path:  # if None avoid save
        # if not os.path.isdir('../Results/Lab9'): # if Results doesn't exists, it creates it
        #     os.makedirs('../Results/Lab9')
        plt.savefig(savefig_path)

