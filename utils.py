import csv
import math


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
def time_of_run_round(schedule, graph_info):
    aggregators_time = {}
    for aggregator in schedule:
        aggregators_time[aggregator] = graph_info['comp_time'][aggregator]
        for worker in schedule[aggregator]:  # getting one worker at a time to calculable the max time at an aggregator
            # If the worker is not an aggregator combine its com. time and com. time to the aggregator.
            if worker not in schedule:
                # If the aggregator's time is less than the time a worker took to complete the process and send the
                # result, update the max time at the aggregator
                if graph_info['comp_time'][worker] + graph_info[worker][aggregator] > aggregators_time[aggregator]:
                    aggregators_time[aggregator] = graph_info['comp_time'][worker] + graph_info[worker][aggregator]
            else:
                if aggregators_time[worker] + graph_info[worker][aggregator] > aggregators_time[aggregator]:
                    aggregators_time[aggregator] = aggregators_time[worker] + graph_info[worker][aggregator]

    # The last aggregator in the schedule is the target client as the main aggregator
    main_aggregator = list(schedule.keys())[-1]
    return aggregators_time[main_aggregator]


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
        workers = list(clients[aggregator-1].neighbours.keys())
        real_workers = [] # This list holds the valid workers of each aggregator
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
        perc = ((total_time_rand-total_time_schedule)/total_time_rand) * 100
    return total_time_schedule, total_time_rand, perc
