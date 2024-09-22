import csv
import math
import random
import pickle
import re
from bs4 import BeautifulSoup
import json

# This function reads the information of the graph from the CSV file and returns in a dict
def get_graph_info(path_to_file, num_clients):
    graph_info = {}  # Dict that holds all the graph info
    comp_time = {}  # Dict for the computation time for all the clients

    with open(path_to_file, 'r') as file:
        csvreader = csv.reader(file, )
        next(csvreader)
        for row in csvreader:
            client_neighbours = {}
            if row[0] == 'Comp. Time':
                for i in range(1, num_clients + 1):
                    comp_time[i] = float(row[i])
                    # print(f'Client {i} -> {row[i]} ')
                graph_info['comp_time'] = comp_time
            else:
                # print(f'Neighbours of client {row[0]}')
                for neighbour in range(1, num_clients + 1):
                    if row[0] != str(neighbour) and row[neighbour] != '':
                        # print(f' ({neighbour} : {row[neighbour]}) ', end='')
                        client_neighbours[neighbour] = float(row[neighbour])
                graph_info[int(row[0])] = client_neighbours
    return graph_info


# This function returns the information (computation time, neighbours and communication time to them) of a client
def get_client_graph_info(graph_info, client_id):
    comp_time = graph_info['comp_time'][client_id]
    neighbours = graph_info[client_id]
    return comp_time, neighbours


# This function takes a list of clients and returns a dict that has the graph topology and new calculated edge weights
# based on communication time and computation time
def get_processed_graph_info(clients):
    processed_graph_info = {}
    for client in clients:
        processed_graph_info[client.client_id] = client.neighbours_with_weights
    return processed_graph_info


#   This function takes the graph info and schedule the clients as aggregators and workers
def schedule_clients(graph_info_, target_client_id):
    # path_table is a dict that stores the shortest path of each client to the target client Each value in path_table
    # is a list, where the first element is the cost and the second element is the path to the target client (
    # aggregator)
    path_table, unvisited = {}, []
    for key in graph_info_:
        if key == target_client_id:
            path_table[target_client_id] = [0, target_client_id]
        else:
            path_table[key] = [math.inf, None]
        unvisited.append(key)

    # The target node which represent the final aggregator
    current_client = target_client_id
    # This loop runs dijkstra's algorithm and returns the path table which is used to find the aggregators and the
    # workers attached with each aggregator
    while unvisited:
        neighbours = get_client_neighbours(graph_info_[current_client])  # neighbours of current the client
        # Looping through the neighbours to see if the path to the target client should be updated
        for neighbour in neighbours:
            cost_of_new_path = get_cost_of_new_path(neighbour, current_client, graph_info_, path_table)
            if cost_of_new_path < path_table[neighbour][0]:  # if the cost of the new path for neighbour is lower
                path_table[neighbour][0] = cost_of_new_path
                path_table[neighbour][1] = current_client

        unvisited.remove(current_client)

        if unvisited:
            current_client = next_client(path_table, unvisited)

    # Finding the aggregators and their workers and saving them in a dict
    schedule = {}
    for client in path_table:  # If the aggregator has not been added to the schedule add it and create a workers' list
        if client != target_client_id:  # do not add the target client (main aggregator to workers)
            if not path_table[client][1] in schedule:
                schedule[path_table[client][1]] = [client]
            else:  # If the aggregator has been added to the schedule just add the client to the workers
                schedule[path_table[client][1]].append(client)
    return sort_schedule(schedule)


# Sort the aggregation sequence based on dependencies between aggregators
def sort_schedule(schedule):
    ordered_schedule = {}
    aggregators = list(schedule.keys())
    for i in range(len(aggregators) + 1):
        for j in range(i + 1, len(aggregators)):
            if aggregators[i] != aggregators[j] and aggregators[j] in schedule[aggregators[i]]:
                aggregators[i], aggregators[j] = aggregators[j], aggregators[i]
    for aggregator in aggregators:
        ordered_schedule[aggregator] = schedule[aggregator]
    return ordered_schedule


def get_client_neighbours(client_info):
    neighbours = []
    for key in client_info:
        neighbours.append(key)
    return neighbours


# The cost of the new path of the neighbour is the cost from the neighbour to current client + the cost from the current
# client to the target client (which represents the cost from the neighbour to the target node)
def get_cost_of_new_path(neighbour, current_client, graph_info_, path_table):
    return graph_info_[neighbour][current_client] + path_table[current_client][0]


def next_client(path_table, unvisited):
    next_client_ = unvisited[0]  # assuming that the next unvisited node will be the next node
    for client_id in path_table:
        if client_id in unvisited and path_table[client_id][0] < path_table[next_client_][0]:
            next_client_ = client_id
    return next_client_


# This function returns the time required to completed on FL run
def time_of_run_round(schedule, graph_info, comm_method):
    aggregators_time = {}
    for aggregator in schedule:
        aggregators_time[aggregator] = graph_info['comp_time'][aggregator]
        for worker in schedule[aggregator]:  # getting one worker at a time to calculable the max time at an aggregator
            # If the worker is not an aggregator combine its com. time and com. time to the aggregator.
            if worker not in schedule or comm_method != 'proposed':
                # If the aggregator's time is less than the time a worker took to complete the process and send the
                # result, update the max time at the aggregator
                if graph_info['comp_time'][worker] + graph_info[worker][aggregator] > aggregators_time[aggregator]:
                    aggregators_time[aggregator] = graph_info['comp_time'][worker] + graph_info[worker][aggregator]
            else:
                if aggregators_time[worker] + graph_info[worker][aggregator] > aggregators_time[aggregator]:
                    aggregators_time[aggregator] = aggregators_time[worker] + graph_info[worker][aggregator]

    if comm_method == 'proposed':
        # The last aggregator in the schedule is the target client as the main aggregator
        main_aggregator = list(schedule.keys())[-1]
        return aggregators_time[main_aggregator]
    else:
        return max(aggregators_time.values())


# Scheduling the clients without any heuristic. Start from main aggregator add all neighbours as workers.
# Then select one of the worker to be an aggregator. loop until not more clients need to be placed
# The aggregators list will hold main client (aggregator) when called
def rand_scheduling(clients, graph_info, aggregators):
    # Get the all the clients' IDes.
    clients_in_network = list(graph_info.keys())[1:]
    schedule = {}
    # Keep looping until no more clients are available and there is no aggregator in the aggregator list
    while clients_in_network and aggregators:
        # Take the first element from the list and consider it to be a possible aggregator
        aggregator = aggregators[0]
        # Put all the neighbours of this aggregator and check them if they can be workers for this aggregator
        workers = list(clients[aggregator - 1].neighbours.keys())
        real_workers = []  # This list holds the valid workers of each aggregator
        for worker in workers:
            if worker in clients_in_network:
                real_workers.append(worker)
                clients_in_network.remove(worker)
        # if there is/are worker/s update the schedule with the new aggregator and its workers
        if real_workers:
            schedule[aggregator] = real_workers

        # remove the aggregator from the clients in the network if it is still in it
        if aggregator in clients_in_network:
            clients_in_network.remove(aggregator)

        aggregators.remove(aggregator)
        aggregators.extend(real_workers)  # Add the workers of the aggregator to be nominated as aggregators

    return sort_schedule(schedule)


def cal_time_Perc(metrics):
    total_time_schedule = 0
    total_time_rand = 0
    for round in metrics:
        total_time_schedule += metrics[round]['schedule_time']
        total_time_rand += metrics[round]['rand_schedule_time']
        perc = ((total_time_rand - total_time_schedule) / total_time_rand) * 100
    return total_time_schedule, total_time_rand, perc


def p2p_scheduling(clients, graph_info):
    # Get the all the clients' IDes.
    clients_in_network = list(graph_info.keys())[1:]
    schedule = {}
    # Keep looping until no more clients are available and there is no aggregator in the aggregator list
    for client in clients_in_network:
        # Take the first element from the list and consider it to be an aggregator
        # Put all the neighbours of this aggregator and check them as  workers for this aggregator
        workers = list(clients[client - 1].neighbours.keys())

        # if there is/are worker/s update the schedule with the new aggregator and its workers
        schedule[client] = workers
    return schedule


def gossip_scheduling(clients, graph_info):
    # Get the all the clients' IDes.
    clients_in_network = list(graph_info.keys())[1:]
    schedule = {}
    scheduled_clients = []  # this list is used to indicate the scheduled clients to be skipped when scheduling
    # Keep looping until no more clients are available and there is no aggregator in the aggregator list
    for client in clients_in_network:
        if client not in scheduled_clients:
            # Take the first element from the list and consider it to be an aggregator
            # Put all the neighbours of this aggregator and check them as  workers for this aggregator

            workers = list(clients[client - 1].neighbours.keys())
            # if there is/are worker/s update the schedule with the new aggregator and its workers
            # Select a random worker
            selected_worker = random.choice(workers)
            try_num = len(workers) * 2
            while selected_worker in scheduled_clients and try_num != 0:
                selected_worker = random.choice(workers)
                try_num -= 1

            schedule[client] = [selected_worker]
            if selected_worker not in scheduled_clients:
                schedule[selected_worker] = [client]
            scheduled_clients.append(client)  # Adding client and worker to be avoided when scheduling the rest.
            scheduled_clients.append(
                selected_worker)  # Adding client and worker to be avoided when scheduling the rest.
    return schedule


# Exporting the results to a csv file
def export_all_results_to_csv(outputs):
    with open('local_results.csv', 'w', newline='') as csv_local:
        local_writer = csv.writer(csv_local)
        local_writer.writerow(
            ['client_id', 'method', 'graph', 'dataset', 'train_sample_num', 'valid_sample_num',
             'neighbors_num', 'round', 'local_train_losses', 'local_test_losses', 'local_test_accuracies',
             'glb_test_losses', 'glb_test_accuracies', 'communication overhead'])
    with open('global_results.csv', 'w', newline='') as csv_glb:
        glb_writer = csv.writer(csv_glb)
        glb_writer.writerow(['round', 'method', 'graph', 'dataset', 'loss', 'accuracy', 'schedule_time',
                             'highest_active_client_num', 'clients_num', 'max_client_comm_overhead', 'total_num_of_comm'])

    for output in outputs:
        print(f"{output['dataset']} - {output['method']} - {output['graph']} ")
        path = f"outputs/{output['dataset']}/{output['method']}/{output['path']}"
        export_result_2_csv(path, output['method'], output['dataset'], output['graph'])
def export_result_2_csv(input_path, method, dataset, graph):

    # Exporting the local metrics of each client
    path = input_path + "/results.pkl"
    with open(path, 'rb') as file:
        clients = pickle.load(file)
    num_of_clients = len(clients)
    clients_records = []
    for client in clients:
        neighbors_num = len(client.neighbours)
        for round in client.local_test_accuracies:
            if method == 'proposed':
                # using the glb_metrics to get the accuracy of each client after each round
                path = input_path + "/glb_metrics.pkl"
                with open(path, 'rb') as file:
                    glb_metrics = pickle.load(file)
                glb_test_losses = glb_metrics[round]['loss']
                glb_test_accuracies = glb_metrics[round]['accuracy']

                # Calculating the comm. overhead on each client
                # getting the rand schedule
                rand_schedule = glb_metrics[round]['rand_schedule']
                if client.client_id in rand_schedule:
                    comm_overhead = len(rand_schedule[client.client_id]) / (num_of_clients-1)
                else:
                    comm_overhead = 1 / (num_of_clients-1)
                data_entry = [client.client_id, "random schedule", graph, dataset, client.training_sample_num,
                              client.validation_sample_num,
                              neighbors_num, round, client.local_train_losses[round][0],
                              client.local_test_losses[round],
                              client.local_test_accuracies[round], glb_test_losses, glb_test_accuracies, comm_overhead]
                clients_records.append(data_entry)

                # getting the schedule
                schedule = glb_metrics[round]['schedule']
                if client.client_id in schedule:
                    comm_overhead = len(schedule[client.client_id]) / (num_of_clients-1)
                else:
                    comm_overhead = 1 / (num_of_clients-1)
            else:
                glb_test_losses = client.glb_test_losses[round]
                glb_test_accuracies = client.glb_test_accuracies[round]

                # Calculating the comm. overhead on each client
                if method == 'p2p':
                    comm_overhead = len(client.neighbours) / (num_of_clients-1)
                else:
                    comm_overhead = 1 / (num_of_clients-1)

            data_entry = [client.client_id, method, graph, dataset, client.training_sample_num,
                          client.validation_sample_num,
                          neighbors_num, round, client.local_train_losses[round][0], client.local_test_losses[round],
                          client.local_test_accuracies[round], glb_test_losses, glb_test_accuracies, comm_overhead]
            clients_records.append(data_entry)

    with open('local_results.csv', 'a', newline='') as csv_local:
        local_writer = csv.writer(csv_local)
        local_writer.writerows(clients_records)

    # Exporting the global metrics
    path = input_path + "/glb_metrics.pkl"
    with open(path, 'rb') as file:
        glb_metrics = pickle.load(file)

        global_records = []
        rand_global_records = []

        # Calculating the highest_active_client_num
        total_num_of_comm = 0
        highest_active_client_num = 0

        if method == 'p2p':
            # if graph == 'randomly connected': we will calculate it from the connection between clients
            if graph == 'randomly connected':
                for client in clients:
                    if len(client.neighbours) > highest_active_client_num:
                        highest_active_client_num = len(client.neighbours)
                    total_num_of_comm += len(client.neighbours)
            elif graph == 'fully connected':
                highest_active_client_num = num_of_clients-1
                total_num_of_comm = highest_active_client_num * num_of_clients
            max_overhead = highest_active_client_num / (num_of_clients-1)
        elif method == 'gossip':
            highest_active_client_num = 1
            total_num_of_comm = num_of_clients
            max_overhead = highest_active_client_num / (num_of_clients-1)

        for round in glb_metrics:
            if method == 'proposed':
                # Calculating the highest_active_client_num and total_num_of_comm when random algo is used
                total_num_of_comm = 0
                highest_active_client_num = 0
                # getting the rand schedule
                rand_schedule = glb_metrics[round]['rand_schedule']
                for client_sc in rand_schedule:
                    total_num_of_comm += len(rand_schedule[client_sc])
                    if len(rand_schedule[client_sc]) > highest_active_client_num:
                        highest_active_client_num = len(rand_schedule[client_sc])

                max_overhead = highest_active_client_num / (num_of_clients-1)
                # adding the entry of the round in random algo

                rand_data_entry = [round, 'random scheduling', graph, dataset, glb_metrics[round]['loss'],
                              glb_metrics[round]['accuracy'], glb_metrics[round]['rand_schedule_time'],
                            highest_active_client_num, num_of_clients, max_overhead, total_num_of_comm]
                rand_global_records.append(rand_data_entry)

                # Calculating the highest_active_client_num and total_num_of_comm when random algo is used
                total_num_of_comm = 0
                highest_active_client_num = 0
                # getting the schedule
                schedule = glb_metrics[round]['schedule']

                # Calculating the highest_active_client_num and total_num_of_comm when proposed algo is used
                for client_sc in schedule:
                    total_num_of_comm += len(schedule[client_sc])
                    if len(schedule[client_sc]) > highest_active_client_num:
                        highest_active_client_num = len(schedule[client_sc])

                max_overhead = highest_active_client_num / (num_of_clients-1)

            # adding the entry of the round
            data_entry = [round, method, graph, dataset, glb_metrics[round]['loss'],
                          glb_metrics[round]['accuracy'], glb_metrics[round]['schedule_time'],
                          highest_active_client_num, num_of_clients, max_overhead, total_num_of_comm]

            global_records.append(data_entry)

    with open('global_results.csv', 'a', newline='') as csv_glb:
        glb_writer = csv.writer(csv_glb)
        glb_writer.writerows(global_records)
        if method == 'proposed':
            glb_writer.writerows(rand_global_records)

