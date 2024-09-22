from pyvis.network import Network
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


def schedule(graph_info, schedule_to_vis, save_path, run_round=0, rand_schedule=False):
    # Create a directed graph
    graph = Network(height='1000px', width='100%', directed=True)
    graph.repulsion(node_distance=50, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)
    aggregators = list(schedule_to_vis.keys())
    for aggregator in schedule_to_vis:
        label = f'{aggregator} ({graph_info["comp_time"][aggregator]})'
        graph.add_node(aggregator, label=label, color='lightgreen',
                       shape='ellipse')  # Add the aggregator client to graph
        for worker in schedule_to_vis[aggregator]:
            label = f'{worker} ({graph_info["comp_time"][worker]})'
            if worker in aggregators:
                graph.add_node(worker, label=label, color='lightgreen',
                               shape='ellipse')  # Add the worker client to graph
            else:
                graph.add_node(worker, label=label, shape='ellipse')  # Add the worker client to graph
            graph.add_edge(worker, aggregator,width=4, label=str(graph_info[worker][aggregator]))

    # Visualize the graph
    # if run_round = 0, that means there is one schedule for all run round of FL
    if run_round == 0:
        if rand_schedule:
            result_path = Path(save_path) / 'rand_schedule.html'
        else:
            result_path = Path(save_path) / 'schedule.html'
    else:
        if rand_schedule:
            result_path = Path(save_path) / f'{run_round}_rand_schedule.html'
        else:
            result_path = Path(save_path) / f'{run_round}_schedule.html'
    graph.show(str(result_path), notebook=False)


def org_graph(graph_info, save_path, num_clients):
    graph = Network(height='1000px', width='100%', directed=False)
    graph.repulsion(node_distance=100, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
    # add the clients in a list excluding the first list which has comp. time
    clients = list(graph_info.keys())[1:num_clients]
    for client in clients:
        label = f'{client} ({graph_info["comp_time"][client]})'
        graph.add_node(client, label=label, shape='ellipse')  # Add the aggregator client to graph
        for nb in graph_info[client]:
            label = f'{nb} ({graph_info["comp_time"][nb]})'
            graph.add_node(nb, label=label, shape='ellipse')  # Add the aggregator client to graph
            graph.add_edge(client, nb, width=4, physics=True, label=str(graph_info[client][nb]), )

    result_path = Path(save_path) / 'graph.html'
    graph.show(str(result_path), notebook=False)


def directed_graph(graph_info, process_time, save_path):
    graph = Network(height='1000px', width='100%', directed=True)
    graph.repulsion(node_distance=30, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)

    for client in graph_info:
        label = f'{client} ({process_time[client]})'
        graph.add_node(client, label=label, shape='ellipse')  # Add the aggregator client to graph
        for nb in graph_info[client]:
            label = f'{nb} ({process_time[nb]})'
            graph.add_node(nb, label=label, shape='ellipse')  # Add the aggregator client to graph
            graph.add_edge(client, nb,width=4, physics=True, label=str(graph_info[client][nb]))

    result_path = Path(save_path) / 'processed_graph.html'
    graph.show(str(result_path), notebook=False)


def local_training(client_id, res_file_path):
    # Specify the path to your pickle file
    client_id = client_id - 1  # the first client in the position zero in clients list
    res_file_path += '/results.pkl'
    # Load data from the pickle file
    with open(res_file_path, 'rb') as file:
        clients = pickle.load(file)

    run_rounds = []
    losses = []
    accuracies = []
    for run_round in clients[client_id].local_test_losses:
        run_rounds.append(run_round)
        losses.append(clients[client_id].local_test_losses[run_round])
        accuracies.append(clients[client_id].local_test_accuracies[run_round])

    fig, loss = plt.subplots()
    loss.set_title(f'Client: {client_id + 1}')
    loss.set_xlabel('Epoch')
    loss.set_ylabel('Loss')
    loss.plot(run_rounds, losses, color='orange')

    fig, accuracy = plt.subplots()
    accuracy.set_title(f'Client: {client_id + 1}')
    accuracy.set_xlabel('Epoch')
    accuracy.set_ylabel('Accuracy')
    accuracy.plot(run_rounds, accuracies)

    plt.show()


def all_local_clients_losses(res_file_path):
    res_file_path += '/results.pkl'

    # Load data from the pickle file
    with open(res_file_path, 'rb') as file:
        clients = pickle.load(file)
    fig, ax = plt.subplots(layout='constrained')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    for client in clients:
        run_rounds = []
        losses = []
        for run_round in client.local_test_losses:
            run_rounds.append(run_round)
            losses.append(client.local_test_losses[run_round])
        cl_label = f'client: {client.client_id}'
        ax.plot(run_rounds, losses, label=cl_label)
    ax.legend()
    plt.show()


def all_local_clients_accuracies(res_file_path):
    res_file_path += '/results.pkl'

    # Load data from the pickle file
    with open(res_file_path, 'rb') as file:
        clients = pickle.load(file)
    fig, ax = plt.subplots(layout='constrained')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    for client in clients:
        run_rounds = []
        accuracies = []
        for run_round in client.local_test_losses:
            run_rounds.append(run_round)
            accuracies.append(client.local_test_accuracies[run_round])
        cl_label = f'client: {client.client_id}'
        ax.plot(run_rounds, accuracies, label=cl_label)
    ax.legend()
    plt.show()


def global_model_performance(file_path):
    file_path += '/glb_metrics.pkl'

    # Load data from the pickle file
    with open(file_path, 'rb') as file:
        global_model_metrics = pickle.load(file)
    run_rounds = []
    losses = []
    accuracies = []

    for run_round in global_model_metrics:
        run_rounds.append(run_round)
        losses.append(global_model_metrics[run_round]['loss'])
        accuracies.append(global_model_metrics[run_round]['accuracy'])

    fig, glb_loss = plt.subplots()
    glb_loss.set_title('Global Model')
    glb_loss.set_xlabel('Epoch')
    glb_loss.set_ylabel('Loss')
    glb_loss.plot(run_rounds, losses, color='orange')

    fig, glb_accuracy = plt.subplots()
    glb_accuracy.set_title('Global Model')
    glb_accuracy.set_xlabel('Epoch')
    glb_accuracy.set_ylabel('Accuracy')
    glb_accuracy.plot(run_rounds, accuracies)

    plt.show()


def data_dist(data_dist_path, data_type, classes):
    import numpy as np

    data_dist_path += '/data_dist.pkl'

    # Load data from the pickle file
    with open(data_dist_path, 'rb') as file:
        all_data_dist = pickle.load(file)

    if data_type == 'train':
        title = 'Training'
    else:
        title = 'Validation'

    data_dist = all_data_dist[data_type]
    print(data_dist)
    clients = list(data_dist.keys())
    num_clients = len(clients)
    num_classes = len(classes)

    client_data = []
    for i in range(num_classes):
        dist = []
        for client in clients:
            dist.append(data_dist[client][i])
        dist.reverse()
        client_data.append(dist)

    fig, ax = plt.subplots(figsize=(10, num_classes))
    # Loop through each class and plot stacked horizontal bars for each client

    for i in range(num_classes):
        left = np.sum(client_data[:i], axis=0) if i > 0 else None
        bars = ax.barh(range(num_clients), client_data[i], left=left, label=f'Class {classes[i]}', alpha=0.5)



        # Annotate each bar with its value
        for bar, val in zip(bars, client_data[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f'{val}', ha='center', va='center', color='black')

    # Adding labels and title
    ax.set_ylabel('Clients')
    ax.set_xlabel('Frequency')
    ax.set_title(f'{title} data distribution')

    # Set y-axis tick labels to show client names
    client_names = [f'Client {num_clients - i}' for i in range(num_clients)]
    plt.yticks(range(num_clients), client_names)

    # Adding legend
    ax.legend()
    # Show plot
    plt.tight_layout()
    plt.show()


# This function shows the confusion matrix
def confusion_matrix(pred_labels_file_path, classes):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    pred_labels_file_path += '/pred_labels.pkl'
    # Load data from the pickle file
    with open(pred_labels_file_path, 'rb') as file:
        pred_labels = pickle.load(file)

    cm = confusion_matrix(pred_labels['labels'], pred_labels['pred'])
    # Plot confusion matrix
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False, xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
