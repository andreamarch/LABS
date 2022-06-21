import core.elements as el
import matplotlib.pyplot as plt
from pathlib import Path
import core.utils as utls
import time
from core.parameters import *

# import/export paths
root = Path(__file__).parent.parent.parent
in_directory = root / 'resources'
out_directory = root / 'results' / 'LAB10_res'

# input files
file_fixed = in_directory / 'nodes_full_fixed_rate.json'
file_flex = in_directory / 'nodes_full_flex_rate.json'
file_shannon = in_directory / 'nodes_full_shannon.json'

# output file
outp_plot_snr = out_directory / '9d5_snr.png'
outp_plot_lat = out_directory / '9d5_lat.png'
outp_plot_br = out_directory / '9d5_br.png'

save_my_figure = False
verbose = False
do_simulations = [True, True]

# network generation
network_fixed = el.Network(file_fixed)
network_fixed.connect()
network_flex = el.Network(file_flex)
network_flex.connect()
network_shan = el.Network(file_shannon)
network_shan.connect()

# ----------------- POINT 1 (Single Traffic Matrix) -----------------
if do_simulations[0]:
    connections_fixed_per_M = []
    number_connections_fixed_rate_per_M = []
    number_blocking_events_fixed_rate_per_M = []
    capacities_fixed_rate_per_M = []
    average_bit_rate_fixed_rate_per_M = []
    minimum_snr_fixed_per_M = []
    maximum_snr_fixed_per_M = []
    average_snr_fixed_per_M = []

    connections_flex_per_M = []
    number_connections_flex_rate_per_M = []
    number_blocking_events_flex_rate_per_M = []
    capacities_flex_rate_per_M = []
    average_bit_rate_flex_rate_per_M = []
    minimum_snr_flex_per_M = []
    maximum_snr_flex_per_M = []
    average_snr_flex_per_M = []

    connections_shannon_per_M = []
    number_connections_shannon_per_M = []
    number_blocking_events_shannon_per_M = []
    capacities_shannon_per_M = []
    average_bit_rate_shannon_per_M = []
    minimum_snr_shannon_per_M = []
    maximum_snr_shannon_per_M = []
    average_snr_shannon_per_M = []

    average_average_snr_fixed = 0
    average_max_snr_fixed = 0
    average_min_snr_fixed = 0
    average_total_capacity_fixed = 0
    average_average_bit_rate_fixed = 0
    average_min_bit_rate_fixed = 0
    average_max_bit_rate_fixed = 0
    average_blocking_events_fixed = 0
    average_successful_connections_fixed = 0

    average_average_snr_flex = 0
    average_max_snr_flex = 0
    average_min_snr_flex = 0
    average_total_capacity_flex = 0
    average_average_bit_rate_flex = 0
    average_min_bit_rate_flex = 0
    average_max_bit_rate_flex = 0
    average_blocking_events_flex = 0
    average_successful_connections_flex = 0

    average_average_snr_shannon = 0
    average_max_snr_shannon = 0
    average_min_snr_shannon = 0
    average_total_capacity_shannon = 0
    average_average_bit_rate_shannon = 0
    average_min_bit_rate_shannon = 0
    average_max_bit_rate_shannon = 0
    average_blocking_events_shannon = 0
    average_successful_connections_shannon = 0

    M_list = [16 for i in range(0, 100)]
    for counter, M in enumerate(M_list):
        if counter == 0:  # save these plots just for the first iteration
            snr_string = '10d1_snr_M' + str(M)
            lat_string = '10d1_lat_M' + str(M)
            br_string = '10d1_br_M' + str(M)

            outp_plot_snr = out_directory / 'Point1' / snr_string
            outp_plot_lat = out_directory / 'Point1' / lat_string
            outp_plot_br = out_directory / 'Point1' / br_string

        traffic_matrix_fixed = utls.generate_traffic_matrix(network_fixed, M)
        [connection_tm_fixed, saturation_fixed] = network_fixed.deploy_traffic_matrix(traffic_matrix_fixed)
        [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_fixed)
        [total_capacity, avg_bit_rate, max_br, min_br] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_fixed)
        [avg_snr, max_snr, min_snr] = utls.compute_average_max_min_snr(connection_tm_fixed)
        print()
        print('FIXED TRANSCEIVER STRATEGY')
        print('-------------------------------')
        print('Simulation number', counter + 1, 'with M =', M)
        print('-------------------------------')
        if verbose:
            if saturation_fixed:
                print('Fixed network for M =', M, ': network was saturated.')
            elif not saturation_fixed:
                print('Fixed network for M =', M, ': traffic matrix was saturated')
        connections_fixed_per_M.append(successful_connections + blocking_events)
        number_connections_fixed_rate_per_M.append(successful_connections)
        number_blocking_events_fixed_rate_per_M.append(blocking_events)
        capacities_fixed_rate_per_M.append(total_capacity)
        average_bit_rate_fixed_rate_per_M.append(avg_bit_rate)
        average_snr_fixed_per_M.append(avg_snr)
        maximum_snr_fixed_per_M.append(max_snr)
        minimum_snr_fixed_per_M.append(min_snr)
        percentage_blocking_events = blocking_events / (blocking_events + successful_connections) * 100
        if verbose:
            print('Number of successful connections:', successful_connections)
            print('Number of blocking events:', blocking_events)
            print('Percentage of blocking events =', percentage_blocking_events, '%')
            print('Total capacity allocated =', total_capacity / 1000, 'Tbps')
            print('Average bit rate =', avg_bit_rate, 'Gbps')
            print('Average SNR =', avg_snr, 'dB')
        average_average_snr_fixed = (average_average_snr_fixed * counter + avg_snr) / (counter + 1)
        average_min_snr_fixed = (average_min_snr_fixed * counter + min_snr) / (counter + 1)
        average_max_snr_fixed = (average_max_snr_fixed * counter + max_snr) / (counter + 1)
        average_total_capacity_fixed = (average_total_capacity_fixed * counter + total_capacity) / (counter + 1)
        average_average_bit_rate_fixed = (average_average_bit_rate_fixed * counter + avg_bit_rate) / (counter + 1)
        average_min_bit_rate_fixed = (average_min_bit_rate_fixed * counter + min_br) / (counter + 1)
        average_max_bit_rate_fixed = (average_max_bit_rate_fixed * counter + max_br) / (counter + 1)
        average_blocking_events_fixed = (average_blocking_events_fixed * counter + blocking_events) / (counter + 1)
        average_successful_connections_fixed = (average_successful_connections_fixed * counter + successful_connections) / (counter + 1)
        print('Average average SNR at run', counter + 1, '=', average_average_snr_fixed, 'dB')
        print('Average minimum SNR at run', counter + 1, '=', average_min_snr_fixed, 'dB')
        print('Average maximum SNR at run', counter + 1, '=', average_max_snr_fixed, 'dB')
        print('Average capacity at run', counter + 1, '=', average_total_capacity_fixed / 1000, 'Tbps')
        print('Average average bit rate at run', counter + 1, '=', average_average_bit_rate_fixed, 'Gbps')
        print('Average minimum bit rate at run', counter + 1, '=', average_min_bit_rate_fixed, 'Gbps')
        print('Average maximum bit rate at run', counter + 1, '=', average_max_bit_rate_fixed, 'Gbps')
        print('Average number of blocking events at run', counter + 1, '=', average_blocking_events_fixed)
        print('Average number of successful connections at run', counter + 1, '=', average_successful_connections_fixed)
        utls.free_lines_and_switch_matrix(file_fixed, network_fixed)

        traffic_matrix_flex = utls.generate_traffic_matrix(network_flex, M)
        [connection_tm_flex, saturation_flex] = network_flex.deploy_traffic_matrix(traffic_matrix_flex)
        [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_flex)
        [total_capacity, avg_bit_rate, max_br, min_br] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_flex)
        [avg_snr, max_snr, min_snr] = utls.compute_average_max_min_snr(connection_tm_flex)
        print()
        print('FLEXIBLE TRANSCEIVER STRATEGY')
        print('-------------------------------')
        print('Simulation number', counter + 1, 'with M =', M)
        print('-------------------------------')
        if verbose:
            if saturation_flex:
                print('Flexible network for M =', M, ': network was saturated.')
            elif not saturation_flex:
                print('Flexible network for M =', M, ': traffic matrix was saturated')
        connections_flex_per_M.append(successful_connections + blocking_events)
        number_connections_flex_rate_per_M.append(successful_connections)
        number_blocking_events_flex_rate_per_M.append(blocking_events)
        capacities_flex_rate_per_M.append(total_capacity)
        average_bit_rate_flex_rate_per_M.append(avg_bit_rate)
        average_snr_flex_per_M.append(avg_snr)
        maximum_snr_flex_per_M.append(max_snr)
        minimum_snr_flex_per_M.append(min_snr)
        percentage_blocking_events = blocking_events / (blocking_events + successful_connections) * 100
        if verbose:
            print('Number of successful connections:', successful_connections)
            print('Number of blocking events:', blocking_events)
            print('Percentage of blocking events =', percentage_blocking_events, '%')
            print('Total capacity allocated =', total_capacity / 1000, 'Tbps')
            print('Average bit rate =', avg_bit_rate, 'Gbps')
            print('Average SNR =', avg_snr, 'dB')
        average_average_snr_flex = (average_average_snr_flex * counter + avg_snr) / (counter + 1)
        average_min_snr_flex = (average_min_snr_flex * counter + min_snr) / (counter + 1)
        average_max_snr_flex = (average_max_snr_flex * counter + max_snr) / (counter + 1)
        average_total_capacity_flex = (average_total_capacity_flex * counter + total_capacity) / (counter + 1)
        average_average_bit_rate_flex = (average_average_bit_rate_flex * counter + avg_bit_rate) / (counter + 1)
        average_min_bit_rate_flex = (average_min_bit_rate_flex * counter + min_br) / (counter + 1)
        average_max_bit_rate_flex = (average_max_bit_rate_flex * counter + max_br) / (counter + 1)
        average_blocking_events_flex = (average_blocking_events_flex * counter + blocking_events) / (counter + 1)
        average_successful_connections_flex = (average_successful_connections_flex * counter + successful_connections) / (counter + 1)
        print('Average average SNR at run', counter + 1, '=', average_average_snr_flex, 'dB')
        print('Average minimum SNR at run', counter + 1, '=', average_min_snr_flex, 'dB')
        print('Average maximum SNR at run', counter + 1, '=', average_max_snr_flex, 'dB')
        print('Average capacity at run', counter + 1, '=', average_total_capacity_flex / 1000, 'Tbps')
        print('Average average bit rate at run', counter + 1, '=', average_average_bit_rate_flex, 'Gbps')
        print('Average minimum bit rate at run', counter + 1, '=', average_min_bit_rate_flex, 'Gbps')
        print('Average maximum bit rate at run', counter + 1, '=', average_max_bit_rate_flex, 'Gbps')
        print('Average number of blocking events at run', counter + 1, '=', average_blocking_events_flex)
        print('Average number of successful connections at run', counter + 1, '=', average_successful_connections_flex)
        utls.free_lines_and_switch_matrix(file_flex, network_flex)

        traffic_matrix_shan = utls.generate_traffic_matrix(network_shan, M)
        [connection_tm_shan, saturation_shan] = network_shan.deploy_traffic_matrix(traffic_matrix_shan)
        [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_shan)
        [total_capacity, avg_bit_rate, max_br, min_br] = utls.compute_network_capacity_and_avg_bit_rate(
            connection_tm_shan)
        [avg_snr, max_snr, min_snr] = utls.compute_average_max_min_snr(connection_tm_shan)
        print()
        print('SHANNON TRANSCEIVER STRATEGY')
        print('-------------------------------')
        print('Simulation number', counter + 1, 'with M =', M)
        print('-------------------------------')
        if verbose:
            if saturation_shan:
                print('Shannon network for M =', M, ': network was saturated.')
            elif not saturation_shan:
                print('Shanno network for M =', M, ': traffic matrix was saturated')
        connections_shannon_per_M.append(successful_connections + blocking_events)
        number_connections_shannon_per_M.append(successful_connections)
        number_blocking_events_shannon_per_M.append(blocking_events)
        capacities_shannon_per_M.append(total_capacity)
        average_bit_rate_shannon_per_M.append(avg_bit_rate)
        average_snr_shannon_per_M.append(avg_snr)
        maximum_snr_shannon_per_M.append(max_snr)
        minimum_snr_shannon_per_M.append(min_snr)
        percentage_blocking_events = blocking_events / (blocking_events + successful_connections) * 100
        if verbose:
            print('Number of successful connections:', successful_connections)
            print('Number of blocking events:', blocking_events)
            print('Percentage of blocking events =', percentage_blocking_events, '%')
            print('Total capacity allocated =', total_capacity / 1000, 'Tbps')
            print('Average bit rate =', avg_bit_rate, 'Gbps')
            print('Average SNR =', avg_snr, 'dB')
        average_average_snr_shannon = (average_average_snr_shannon * counter + avg_snr) / (counter + 1)
        average_min_snr_shannon = (average_min_snr_shannon * counter + min_snr) / (counter + 1)
        average_max_snr_shannon = (average_max_snr_shannon * counter + max_snr) / (counter + 1)
        average_total_capacity_shannon = (average_total_capacity_shannon * counter + total_capacity) / (counter + 1)
        average_average_bit_rate_shannon = (average_average_bit_rate_shannon * counter + avg_bit_rate) / (counter + 1)
        average_min_bit_rate_shannon = (average_min_bit_rate_shannon * counter + min_br) / (counter + 1)
        average_max_bit_rate_shannon = (average_max_bit_rate_shannon * counter + max_br) / (counter + 1)
        average_blocking_events_shannon = (average_blocking_events_shannon * counter + blocking_events) / (counter + 1)
        average_successful_connections_shannon = (average_successful_connections_shannon * counter + successful_connections) / (counter + 1)
        print('Average average SNR at run', counter + 1, '=', average_average_snr_shannon, 'dB')
        print('Average minimum SNR at run', counter + 1, '=', average_min_snr_shannon, 'dB')
        print('Average maximum SNR at run', counter + 1, '=', average_max_snr_shannon, 'dB')
        print('Average capacity at run', counter + 1, '=', average_total_capacity_shannon / 1000, 'Tbps')
        print('Average average bit rate at run', counter + 1, '=', average_average_bit_rate_shannon, 'Gbps')
        print('Average minimum bit rate at run', counter + 1, '=', average_min_bit_rate_shannon, 'Gbps')
        print('Average maximum bit rate at run', counter + 1, '=', average_max_bit_rate_shannon, 'Gbps')
        print('Average number of blocking events at run', counter + 1, '=', average_blocking_events_shannon)
        print('Average number of successful connections at run', counter + 1, '=', average_successful_connections_shannon)
        utls.free_lines_and_switch_matrix(file_shannon, network_shan)

