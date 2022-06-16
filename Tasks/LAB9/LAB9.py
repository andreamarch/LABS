import core.elements as el
import matplotlib.pyplot as plt
from pathlib import Path
import core.utils as utls
import time
from core.parameters import *

# import/export paths
root = Path(__file__).parent.parent.parent
in_directory = root / 'resources'
out_directory = root / 'results'

# input files
file_fixed = in_directory / 'nodes_full_fixed_rate.json'
file_flex = in_directory / 'nodes_full_flex_rate.json'
file_shannon = in_directory / 'nodes_full_shannon.json'

# output file
outp_plot_snr = out_directory / '9d5_snr.png'
outp_plot_lat = out_directory / '9d5_lat.png'
outp_plot_br = out_directory / '9d5_br.png'

# network generation
network_fixed = el.Network(file_fixed)
network_fixed.connect()
network_flex = el.Network(file_flex)
network_flex.connect()
network_shan = el.Network(file_shannon)
network_shan.connect()

if not outp_plot_snr.is_file() and not outp_plot_lat.is_file() and not outp_plot_br.is_file():
    connection_list_fixed = utls.generate_random_connections(network_fixed)
    print('Starting fixed rate stream...')
    t_start = time.time()
    connection_list_fixed = network_fixed.stream(connection_list_fixed, 'snr')
    t_end = time.time()
    [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_list_fixed)
    [total_capacity_fixed, avg_bit_rate_fixed] = utls.compute_network_capacity_and_avg_bit_rate(connection_list_fixed)
    print('Number of successful connections:', successful_connections)
    print('Number of blocking events:', blocking_events)
    print('Total capacity allocated =', total_capacity_fixed, 'Gbps.')
    print('Average bit rate =', avg_bit_rate_fixed, 'Gbps.')
    print('... Fixed rate stream done (elapsed', t_end - t_start, 's).')
    print()

if not outp_plot_snr.is_file() and not outp_plot_lat.is_file() and not outp_plot_br.is_file():
    connection_list_flex = utls.generate_random_connections(network_flex)
    print('Starting flexible rate stream...')
    t_start = time.time()
    connection_list_flex = network_flex.stream(connection_list_flex, 'snr')
    t_end = time.time()
    [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_list_flex)
    [total_capacity_flex, avg_bit_rate_flex] = utls.compute_network_capacity_and_avg_bit_rate(connection_list_flex)
    print('Number of successful connections:', successful_connections, '.')
    print('Number of blocking events:', blocking_events, '.')
    print('Total capacity allocated =', total_capacity_flex, 'Gbps.')
    print('Average bit rate =', avg_bit_rate_flex, 'Gbps.')
    print('... Flexible rate stream done (elapsed', t_end - t_start, 's).')
    print()

if not outp_plot_snr.is_file() and not outp_plot_lat.is_file() and not outp_plot_br.is_file():
    connection_list_shan = utls.generate_random_connections(network_shan)
    print('Starting Shannon rate stream...')
    t_start = time.time()
    connection_list_shan = network_shan.stream(connection_list_shan, 'snr')
    t_end = time.time()
    [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_list_shan)
    [total_capacity_shan, avg_bit_rate_shan] = utls.compute_network_capacity_and_avg_bit_rate(connection_list_shan)
    print('Number of successful connections:', successful_connections, '.')
    print('Number of blocking events:', blocking_events, '.')
    print('Total capacity allocated =', total_capacity_shan, 'Gbps.')
    print('Average bit rate =', avg_bit_rate_shan, 'Gbps.')
    print('... Shannon rate stream done (elapsed', t_end - t_start, 's).')
    print()

# snr plot
if not outp_plot_snr.is_file():
    snr_fixed = [connection.snr for connection in connection_list_fixed]
    snr_flex = [connection.snr for connection in connection_list_flex]
    snr_shan = [connection.snr for connection in connection_list_shan]
    snr_list = [snr_fixed, snr_flex, snr_shan]

    utls.plot_histogram(1, snr_list, 20, 'k', ['r', 'b', 'g'], ['Fixed rate', 'Flexible rate', 'Shannon rate'],
                        'SNR distribution for the various transreceiver simulations', 'Number of results', 'SNR, dB')
    plt.savefig(outp_plot_snr)

# latency plot
if not outp_plot_lat.is_file():
    lat_fixed = [connection.latency for connection in connection_list_fixed]
    lat_flex = [connection.latency for connection in connection_list_flex]
    lat_shan = [connection.latency for connection in connection_list_shan]
    lat_list = [lat_fixed, lat_flex, lat_shan]

    utls.plot_histogram(2, lat_list, 20, 'k', ['r', 'b', 'g'], ['Fixed rate', 'Flexible rate', 'Shannon rate'],
                        'Latency for the various transreceiver simulations', 'Number of results', 'Latency, s')
    plt.savefig(outp_plot_lat)

# bit rate plot
if not outp_plot_br.is_file():
    br_fixed = [connection.bit_rate for connection in connection_list_fixed]
    br_flex = [connection.bit_rate for connection in connection_list_flex]
    br_shan = [connection.bit_rate for connection in connection_list_shan]
    br_list = [br_fixed, br_flex, br_shan]

    utls.plot_histogram(3, br_list, 15, 'k', ['r', 'b', 'g'], ['Fixed rate', 'Flexible rate', 'Shannon rate'],
                        'Bit rate for the various transreceiver simulations', 'Number of results', 'Bit rate, Gbps')
    plt.savefig(outp_plot_br)

# traffic_matrix = utls.generate_traffic_matrix(network_fixed, 1)
# connection_list_fixed = network_fixed.deploy_traffic_matrix(traffic_matrix)

connections_fixed_per_M = []
number_connections_fixed_rate_per_M = []
number_blocking_events_fixed_rate_per_M = []
capacities_fixed_rate_per_M = []
average_bit_rate_fixed_rate_per_M = []

connections_flex_per_M = []
number_connections_flex_rate_per_M = []
number_blocking_events_flex_rate_per_M = []
capacities_flex_rate_per_M = []
average_bit_rate_flex_rate_per_M = []

connections_shannon_per_M = []
number_connections_shannon_per_M = []
number_blocking_events_shannon_per_M = []
capacities_shannon_per_M = []
average_bit_rate_shannon_per_M = []

M_list = []
for M in range(1, 52, 5):
    print('----------------------')
    print('Simulation with M =', M)
    print('----------------------')
    print()

    snr_string = '9d7_snr_M' + str(M) + '_N_' + str(max_number_of_iterations)
    lat_string = '9d7_lat_M' + str(M) + '_N_' + str(max_number_of_iterations)
    br_string = '9d7_br_M' + str(M) + '_N_' + str(max_number_of_iterations)
    outp_plot_snr = out_directory / 'LAB9_POINT7_res' / snr_string
    outp_plot_lat = out_directory / 'LAB9_POINT7_res' / lat_string
    outp_plot_br = out_directory / 'LAB9_POINT7_res' / br_string

    M_list.append(M)
    traffic_matrix_fixed = utls.generate_traffic_matrix(network_fixed, M)
    [connection_tm_fixed, saturation_fixed] = network_fixed.deploy_traffic_matrix(traffic_matrix_fixed)
    [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_fixed)
    [total_capacity, avg_bit_rate] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_fixed)
    connections_fixed_per_M.append(successful_connections + blocking_events)
    number_connections_fixed_rate_per_M.append(successful_connections)
    number_blocking_events_fixed_rate_per_M.append(blocking_events)
    capacities_fixed_rate_per_M.append(total_capacity)
    average_bit_rate_fixed_rate_per_M.append(avg_bit_rate)
    if saturation_fixed:
        print('Fixed network for M =', M, ': network was saturated.')
    elif not saturation_fixed and len(connection_tm_fixed) < max_number_of_iterations:
        print('Fixed network for M =', M, ': traffic matrix was saturated')
    elif not saturation_fixed and len(connection_tm_fixed) == max_number_of_iterations:
        print('Fixed network for M =', M, ': maximum number of connections was reached.')
    print('Number of successful connections:', successful_connections, '.')
    print('Number of blocking events:', blocking_events, '.')
    print('Total capacity allocated =', total_capacity / 1000, 'Tbps.')
    print('Average bit rate =', avg_bit_rate, 'Gbps.')
    print()

    traffic_matrix_flex = utls.generate_traffic_matrix(network_flex, M)
    [connection_tm_flex, saturation_flex] = network_flex.deploy_traffic_matrix(traffic_matrix_flex)
    [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_flex)
    [total_capacity, avg_bit_rate] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_flex)
    connections_flex_per_M.append(successful_connections + blocking_events)
    number_connections_flex_rate_per_M.append(successful_connections)
    number_blocking_events_flex_rate_per_M.append(blocking_events)
    capacities_flex_rate_per_M.append(total_capacity)
    average_bit_rate_flex_rate_per_M.append(avg_bit_rate)
    if saturation_flex:
        print('Flexible network for M =', M, ': network was saturated.')
    elif not saturation_flex and len(connection_tm_flex) < max_number_of_iterations:
        print('Flexible network for M =', M, ': traffic matrix was saturated')
    elif not saturation_flex and len(connection_tm_flex) == max_number_of_iterations:
        print('Flexible network for M =', M, ': maximum number of connections was reached.')
    print('Number of successful connections:', successful_connections, '.')
    print('Number of blocking events:', blocking_events, '.')
    print('Total capacity allocated =', total_capacity / 1000, 'Tbps.')
    print('Average bit rate =', avg_bit_rate, 'Gbps.')
    print()

    traffic_matrix_shan = utls.generate_traffic_matrix(network_shan, M)
    [connection_tm_shan, saturation_shan] = network_shan.deploy_traffic_matrix(traffic_matrix_shan)
    [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_shan)
    [total_capacity, avg_bit_rate] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_shan)
    connections_shannon_per_M.append(successful_connections + blocking_events)
    number_connections_shannon_per_M.append(successful_connections)
    number_blocking_events_shannon_per_M.append(blocking_events)
    capacities_shannon_per_M.append(total_capacity)
    average_bit_rate_shannon_per_M.append(avg_bit_rate)
    if saturation_shan:
        print('Shannon network for M =', M, ': network was saturated.')
    elif not saturation_shan and len(connection_tm_shan) < max_number_of_iterations:
        print('Shannon network for M =', M, ': traffic matrix was saturated')
    elif not saturation_shan and len(connection_tm_shan) == max_number_of_iterations:
        print('Fixed network for M =', M, ': maximum number of connections was reached.')
    print('Number of successful connections:', successful_connections, '.')
    print('Number of blocking events:', blocking_events, '.')
    print('Total capacity allocated =', total_capacity / 1000, 'Tbps.')
    print('Average bit rate =', avg_bit_rate, 'Gbps.')
    print()
    print()

    lat_fixed = [connection.latency for connection in connection_tm_fixed]
    lat_flex = [connection.latency for connection in connection_tm_flex]
    lat_shan = [connection.latency for connection in connection_tm_shan]
    lat_list = [lat_fixed, lat_flex, lat_shan]

    snr_fixed = [connection.snr for connection in connection_tm_fixed]
    snr_flex = [connection.snr for connection in connection_tm_flex]
    snr_shan = [connection.snr for connection in connection_tm_shan]
    snr_list = [snr_fixed, snr_flex, snr_shan]

    br_fixed = [connection.bit_rate for connection in connection_tm_fixed]
    br_flex = [connection.bit_rate for connection in connection_tm_flex]
    br_shan = [connection.bit_rate for connection in connection_tm_shan]
    br_list = [br_fixed, br_flex, br_shan]

    utls.plot_histogram(1, snr_list, 20, 'k', ['r', 'b', 'g'], ['Fixed rate', 'Flexible rate', 'Shannon rate'],
                        'SNR distribution, M = ' + str(M), 'Number of results', 'SNR, dB')
    plt.savefig(outp_plot_snr)

    utls.plot_histogram(2, lat_list, 20, 'k', ['r', 'b', 'g'], ['Fixed rate', 'Flexible rate', 'Shannon rate'],
                        'Latency, M = ' + str(M), 'Number of results', 'Latency, s')
    plt.savefig(outp_plot_lat)

    utls.plot_histogram(3, br_list, 15, 'k', ['r', 'b', 'g'], ['Fixed rate', 'Flexible rate', 'Shannon rate'],
                        'Bit rate, M = ' + str(M), 'Number of results', 'Bit rate, Gbps')
    plt.savefig(outp_plot_br)

    utls.free_lines_and_switch_matrix(file_fixed, network_fixed)
    utls.free_lines_and_switch_matrix(file_flex, network_flex)
    utls.free_lines_and_switch_matrix(file_shannon, network_shan)

    # plt.show()

# Number Connections
utls.plot_bar(figure_num=13, list_data=[[number_connections_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                        [number_connections_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                        [number_connections_shannon_per_M[i] for i in range(0, len(M_list))]],
              x_ticks=[M for M in M_list], bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center',
              edge_color='k', color=['r', 'b', 'g'], label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'],
              xlabel='M', ylabel='Number of connections',
              title='Total number of connections per M', myalpha=1)
plt.savefig(out_directory / 'LAB9_POINT7_res' / '9d7_connections_per_M')

# Number Blocking Events
utls.plot_bar(figure_num=14, list_data=[[number_blocking_events_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                        [number_blocking_events_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                        [number_blocking_events_shannon_per_M[i] for i in range(0, len(M_list))]],
              x_ticks=[M for M in M_list], bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center',
              edge_color='k', color=['r', 'b', 'g'], label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'],
              xlabel='M', ylabel='Number of blocking events',
              title='Number of blocking events per M', myalpha=1)
plt.savefig(out_directory / 'LAB9_POINT7_res' / '9d7_blocking_events_per_M')

# Capacity
utls.plot_bar(figure_num=14, list_data=[[capacities_fixed_rate_per_M[i]/1000 for i in range(0, len(M_list))],
                                        [capacities_flex_rate_per_M[i]/1000 for i in range(0, len(M_list))],
                                        [capacities_shannon_per_M[i]/1000 for i in range(0, len(M_list))]],
              x_ticks=[M for M in M_list], bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center',
              edge_color='k', color=['r', 'b', 'g'], label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'],
              xlabel='M', ylabel='Total Capacity, Tbps',
              title='Total network capacity per M', myalpha=1)
plt.savefig(out_directory / 'LAB9_POINT7_res' / '9d7_capacity_per_M')

# Average Bit Rate
utls.plot_bar(figure_num=14, list_data=[[average_bit_rate_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                        [average_bit_rate_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                        [average_bit_rate_shannon_per_M[i] for i in range(0, len(M_list))]],
              x_ticks=[M for M in M_list], bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center',
              edge_color='k', color=['r', 'b', 'g'], label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'],
              xlabel='M', ylabel='Average bit rate, Gbps',
              title='Average bit rate per M', myalpha=1)
plt.savefig(out_directory / 'LAB9_POINT7_res' / '9d7_avg_bit_rate_per_M')
plt.show()
