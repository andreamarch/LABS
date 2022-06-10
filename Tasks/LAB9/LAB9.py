import core.elements as el
import random
import matplotlib.pyplot as plt
from pathlib import Path
import core.utils as utls
import time

# import/export paths
root = Path(__file__).parent.parent.parent
in_directory = root / 'resources'
out_directory = root / 'results'

# input files
file_fixed = in_directory / 'nodes_full_fixed_rate.json'
file_flex = in_directory / 'nodes_full_flex_rate.json'
file_shannon = in_directory / 'nodes_full_shannon.json'

# output file
outp_plot_snr = out_directory / '9d5_snr_fixed'
outp_plot_lat = out_directory / '9d5_lat_fixed'
outp_plot_br = out_directory / '9d5_br_fixed'


#network generation
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
    [total_capacity_fixed, avg_bit_rate_fixed] = utls.compute_network_capacity_and_avg_bit_rate(connection_list_fixed)
    print('Total capacity allocated =', total_capacity_fixed, 'Gbps.')
    print('Average bit rate =', avg_bit_rate_fixed, 'Gbps.')
    print('... Fixed rate stream done (elapsed', t_end-t_start, 's).')
    print()

if not outp_plot_snr.is_file() and not outp_plot_lat.is_file() and not outp_plot_br.is_file():
    connection_list_flex = utls.generate_random_connections(network_flex)
    print('Starting flexible rate stream...')
    t_start = time.time()
    connection_list_flex = network_flex.stream(connection_list_flex, 'snr')
    t_end = time.time()
    [total_capacity_flex, avg_bit_rate_flex] = utls.compute_network_capacity_and_avg_bit_rate(connection_list_flex)
    print('Total capacity allocated =', total_capacity_flex, 'Gbps.')
    print('Average bit rate =', avg_bit_rate_flex, 'Gbps.')
    print('... Flexible rate stream done (elapsed', t_end-t_start, 's).')
    print()

if not outp_plot_snr.is_file() and not outp_plot_lat.is_file() and not outp_plot_br.is_file():
    connection_list_shan = utls.generate_random_connections(network_shan)
    print('Starting Shannon rate stream...')
    t_start = time.time()
    connection_list_shan = network_shan.stream(connection_list_shan, 'snr')
    t_end = time.time()
    [total_capacity_shan, avg_bit_rate_shan] = utls.compute_network_capacity_and_avg_bit_rate(connection_list_shan)
    print('Total capacity allocated =', total_capacity_shan, 'Gbps.')
    print('Average bit rate =', avg_bit_rate_shan, 'Gbps.')
    print('... Shannon rate stream done (elapsed', t_end-t_start, 's).')
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

plt.show()

