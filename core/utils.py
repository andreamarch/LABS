import numpy as np
import json
import pandas as pd
import core.info as info
import random
import core.elements as el
from core.parameters import *
import matplotlib.pyplot as plt
import math


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
    br_list = np.array([connections[k].bit_rate for k in range(0, len(connections))])
    total_capacity = np.nansum(br_list)
    avg_bit_rate = total_capacity / (len(br_list) - np.count_nonzero(np.isnan(br_list)))
    max_bit_rate = max(br_list)
    min_bit_rate = min(br_list)
    return total_capacity, avg_bit_rate, max_bit_rate, min_bit_rate


# Function for computing average, minimum and maximum snr
def compute_average_max_min_snr(connections):
    snr_list = np.array([connections[k].snr for k in range(0, len(connections))])
    snr_list = 10 ** (snr_list[np.nonzero(snr_list)] / 10)
    snr_average = 10 * np.log10(sum(snr_list) / len(snr_list))
    snr_min = 10 * np.log10(min(snr_list))
    snr_max = 10 * np.log10(max(snr_list))
    return snr_average, snr_max, snr_min


# Function for computing the usage of each line
def count_line_usage(network):
    lines = network.lines
    line_list = []
    occupation = []
    for line in lines.keys():
        line_list.append(line)
        occupation.append(10 - sum(lines[line].state))
    return line_list, occupation

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
             remove_y_ticks=None, y_range=None):
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
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    ax.set_axisbelow(True)
    plt.grid()
    if x_ticks:
        if len(x_ticks) == 30 or len(x_ticks) == 42 or len(x_ticks) == 24:
            xtick_labels = [str(M) for M in x_ticks]
        else:
            xtick_labels = ['M=' + str(M) for M in x_ticks]
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


# Function for the computation of the intersection of two given lines
def line_intersections(line1, line2, network):
    coord_line1 = [[network.nodes[line1[0]].position[0] / 1000, network.nodes[line1[0]].position[1] / 1000],
                   [network.nodes[line1[1]].position[0] / 1000, network.nodes[line1[1]].position[1] / 1000]]
    coord_line2 = [[network.nodes[line2[0]].position[0] / 1000, network.nodes[line2[0]].position[1] / 1000],
                   [network.nodes[line2[1]].position[0] / 1000, network.nodes[line2[1]].position[1] / 1000]]
    xdiff = (coord_line1[0][0] - coord_line1[1][0], coord_line2[0][0] - coord_line2[1][0])
    ydiff = (coord_line1[0][1] - coord_line1[1][1], coord_line2[0][1] - coord_line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*coord_line1), det(*coord_line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# Function for computing the average node degree
def compute_node_degree(network):
    nodes = network.nodes
    connected_nodes_dictionary = {n: nodes[n].connected_nodes for n in nodes.keys()}
    node_degree_dictionary = {n: len(connected_nodes_dictionary[n]) for n in connected_nodes_dictionary.keys()}
    average_node_degree = sum(list(node_degree_dictionary.values())) / len(nodes)
    return average_node_degree


# Function for the analysis of the length of the lines (max, min, average)
def length_line_analysis(network):
    lines = dict(network.lines)
    line_labels = [line_label for line_label in lines.keys()]
    line_lengths = [lines[n].length / 1000 for n in lines.keys()]  # in [km]
    max_length = max(line_lengths)
    position = line_lengths.index(max_length)
    max_length = ("{:.2f}".format(max_length), line_labels[position])
    min_length = min(line_lengths)
    position = line_lengths.index(min_length)
    min_length = ("{:.2f}".format(min_length), line_labels[position])
    avg_length = "{:.2f}".format(sum(line_lengths) / len(line_lengths))
    return avg_length, max_length, min_length


# Function for the computation of the average line occupation
def average_line_occupation(network):
    lines = network.lines
    number_of_lines = len(lines)
    active_channels = 0
    for line in lines.values():
        active_channels += 10 - sum(line.state)
    avg_active_channels = active_channels / number_of_lines
    avg_occupancy = avg_active_channels / number_of_channels * 100  # [%]
    return avg_active_channels, avg_occupancy


# Function for computing the occurrences of every input output pair.
def compute_in_out_node_distribution(network, connection_list):
    nodes = network.nodes
    pairs_list = [n1 + n2 for n1 in nodes.keys() for n2 in nodes.keys() if n1 != n2]
    occurrence = [0] * len(pairs_list)
    for connection in connection_list:
        pair = connection.input + connection.output
        location = pairs_list.index(pair)
        occurrence[location] += 1
    return pairs_list, occurrence


# Function for computing the average, max and minimum latency
def compute_avg_max_min_latency(connections):
    latency_list = [connections[k].latency for k in range(0, len(connections)) if not math.isnan(connections[k].latency)]
    length_list = len(latency_list)
    avg_latency = sum(latency_list) / length_list
    max_latency = max(latency_list)
    min_latency = min(latency_list)
    return avg_latency, max_latency, min_latency
