import pickle
from pathlib import Path
from random import randint

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import copy

import torch

from dataset import prepare_dataset
from models import Net, train, test, test_global, VGG11, ResNet18
from client import Client
from fed import FedAvg
from utils import (get_graph_info, get_client_graph_info, get_processed_graph_info, schedule_clients, time_of_run_round,
                   rand_scheduling, cal_time_Perc)
import visulize as vis
from ResNet import ResNet50

# A decorator for Hydra. This tells hydra to by default load the conf in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    #
    classes = {'MNIST': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
               'PathMNIST': ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'],
               'OrganAMNIST': ['heart', ' left lung', 'right lung', 'liver', 'spleen', 'pancreas', 'left kidney',
                               'right kidney', 'bladder', 'L femoral head', 'R femoral head']}
    # vis.local_training(1, cfg.res_file_path)
    # vis.all_local_clients_losses(cfg.res_file_path)
    # vis.all_local_clients_accuracies(cfg.res_file_path)
    # vis.global_model_performance(cfg.res_file_path)
    # vis.data_dist(cfg.res_file_path, 'val', classes[cfg.dataset])
    # vis.confusion_matrix(cfg.res_file_path, classes[cfg.dataset])
    # return True
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.dataset, save_path, cfg.NonIID, cfg.num_classes,
                                                                  cfg.input_size, cfg.num_clients, cfg.batch_size)

    # Getting the graph information form the csv file
    graph_info = get_graph_info(cfg.path_to_csv, cfg.num_clients)

    # Initializing the clients
    clients = []
    if cfg.model == 'Net':
        model = Net(cfg.num_classes)
    elif cfg.model == 'ResNet50':
        model = ResNet50('3', cfg.num_classes)
    elif cfg.model == 'ResNet18':
        model = ResNet18(cfg.num_classes)
    else:
        model = VGG11(cfg.num_classes)

    for client_id in range(1, cfg.num_clients + 1):
        comp_time, neighbors = get_client_graph_info(graph_info, client_id)  # The info about this client in the graph
        client = Client(client_id, copy.deepcopy(model), trainloaders[client_id - 1], validationloaders[client_id - 1],
                        cfg.local_epoch, comp_time, neighbors)
        # client = Client(client_id, model, None, None, cfg.local_epoch, comp_time, neighbors)
        client.aggr_model = copy.deepcopy(model)

        client.prepare_client_for_scheduling()
        clients.append(client)

    # processed graph info means combining the computation and communication time and reforming a directed graph with
    # new weighted edges
    print(graph_info)
    processed_graph_info = get_processed_graph_info(clients)
    print(processed_graph_info)
    vis.org_graph(graph_info, save_path, cfg.num_clients)
    vis.directed_graph(processed_graph_info, graph_info['comp_time'], save_path)
    if not cfg.random_main_aggregator:
        schedule = schedule_clients(processed_graph_info, cfg.target_client)
        schedule_time = time_of_run_round(schedule, graph_info)  # Calculate the time spent to complete a run round of FL
        print(f'schedule: {schedule} / Time to complete: {schedule_time}')
        rand_schedule = rand_scheduling(clients, graph_info, [cfg.target_client])
        vis.schedule(graph_info, schedule, save_path)
        rand_schedule_time = time_of_run_round(rand_schedule, graph_info)  # Calculate the time spent to complete a run round of FL
        print(f'rand_schedule: {rand_schedule} / Time to complete: {rand_schedule_time}')
        vis.schedule(graph_info, rand_schedule, save_path, rand_schedule=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check if the GPU available
    global_model_metrics = {}

    # Training all the clients in each run
    for run_round in range(1, cfg.num_rounds + 1):
        # selecting a random main aggregator
        if cfg.random_main_aggregator:
            main_aggregator = randint(1, cfg.num_clients)
            # schedule clients in each round
            schedule = schedule_clients(processed_graph_info, main_aggregator)
            # Calculate the time spent to complete a run round of FL
            schedule_time = time_of_run_round(schedule, graph_info)
            print(f'Run_round: {run_round} / schedule: {schedule} / Time to complete: {schedule_time}')
            vis.schedule(graph_info, schedule, save_path, run_round)

            rand_schedule = rand_scheduling(clients, graph_info, [main_aggregator])
            # Calculate the time spent to complete a run round of FL
            rand_schedule_time = time_of_run_round(rand_schedule, graph_info)
            print(f'Rand_schedule: {rand_schedule} / Time to complete: {rand_schedule_time}')
            vis.schedule(graph_info, rand_schedule, save_path, run_round, True)
        else:
            main_aggregator = cfg.target_client

        print(f'Working in round : {run_round}')
        for client in clients:
            # print(f'Training client: {client_id + 1} in round: {run_round}')
            train_losses = train(client, cfg.dataset, device, run_round)
            client.local_train_losses[run_round] = train_losses

            loss, accuracy = test(client, cfg.dataset, device)
            client.local_test_losses[run_round] = loss
            client.local_test_accuracies[run_round] = accuracy

            print(f'local_train_losses: {client.local_train_losses}')
            print(f'local_test_losses: {client.local_test_losses}')
            print(f'local_test_accuracies: {client.local_test_accuracies}')
            print('-------------------------')

        # Aggregating based on the schedule
        for aggregator in schedule:
            clients_to_aggregate = [clients[aggregator - 1]]
            for worker in schedule[aggregator]:
                clients_to_aggregate.append(clients[worker - 1])
            weighted_averaged_model, total_samples_number = FedAvg(clients_to_aggregate, schedule, device,
                                                                   cfg.num_classes, cfg.model)
            clients[clients_to_aggregate[0].client_id - 1].update_aggr_model_and_num_of_samples(weighted_averaged_model,
                                                                                                total_samples_number)
        # In each round will evaluate the global model on the test dataset
        glb_model = copy.deepcopy(clients[main_aggregator - 1].aggr_model)

        # Check if this is the last run_round get the predictions and labels to compute the confusion matrix
        last_round = run_round == cfg.num_rounds
        if last_round:
            glb_loss, glb_accuracy, all_predictions, all_true_labels = test_global(glb_model, cfg.dataset, testloader,
                                                                                   device, last_round)
            pred_labels = {'labels': all_true_labels, 'pred': all_predictions}
            # Saving the predictions and the Ture labels
            result_path = Path(save_path) / 'pred_labels.pkl'
            with open(str(result_path), "wb") as h:
                pickle.dump(pred_labels, h, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            glb_loss, glb_accuracy = test_global(glb_model, cfg.dataset, testloader, device)

        global_model_metrics[run_round] = {'loss': glb_loss, 'accuracy': glb_accuracy, 'schedule_time': schedule_time,
                                           'schedule': schedule, 'rand_schedule': rand_schedule,
                                           'rand_schedule_time': rand_schedule_time}
        print(f'Global model evaluating / run round: {run_round} / loss: {glb_loss} / accuracy: {glb_accuracy} '
              f'/ Schedule_ime: {schedule_time} / Rand_schedule_time: {rand_schedule_time}')

        print('===============================')

        # # Aggregating based on the rand schedule
        # clients2 = copy.deepcopy(clients)
        # for aggregator in rand_schedule:
        #     clients_to_aggregate = [clients2[aggregator - 1]]
        #     for worker in rand_schedule[aggregator]:
        #         clients_to_aggregate.append(clients2[worker - 1])
        #     weighted_averaged_model, total_samples_number = FedAvg(clients_to_aggregate, rand_schedule, device,
        #                                                            cfg.num_classes, cfg.model)
        #     clients2[clients_to_aggregate[0].client_id - 1].update_aggr_model_and_num_of_samples(weighted_averaged_model,
        #                                                                                         total_samples_number)
        # # In each round will evaluate the global model on the test dataset
        # print('Evaluating the  global model')
        # glb_model2 = copy.deepcopy(clients[main_aggregator - 1].aggr_model)
        # glb_loss, glb_accuracy = test_global(glb_model2, testloader, device)
        # # global_model_metrics[run_round] = {'loss': glb_loss, 'accuracy': glb_accuracy, 'schedule_time': schedule_time,
        # #                                    'schedule': schedule, 'rand_schedule': rand_schedule,
        # #                                    'rand_schedule_time': rand_schedule_time}
        # print(f'Global model evaluating / run round: {run_round} / loss: {glb_loss} / accuracy: {glb_accuracy} '
        #       f'/ Schedule_ime: {schedule_time} / Rand_schedule_time: {rand_schedule_time}')
        #
        # print('===============================')

        # after each run round, distribute the global model to all clients to update their models
        for client in clients:
            client.model = copy.deepcopy(glb_model)
        # for client in clients:
        #     client.model = glb_model

    # Saving the clients as a pickle file
    result_path = Path(save_path) / 'results.pkl'
    with open(str(result_path), "wb") as h:
        pickle.dump(clients, h, protocol=pickle.HIGHEST_PROTOCOL)
    # Saving the global model metrics
    result_path = Path(save_path) / 'glb_metrics.pkl'
    with open(str(result_path), "wb") as h:
        pickle.dump(global_model_metrics, h, protocol=pickle.HIGHEST_PROTOCOL)


    schedule_time, rand_schedule_time, perc = cal_time_Perc(global_model_metrics)
    print(f'Schedule Time: {schedule_time} | Rand. Schedule time: {rand_schedule_time} | Less by: {perc}%' )


if __name__ == "__main__":
    main()

    # # Load data from the pickle file
    # with open("outputs/OrganMNIST/OrganAMNIST_Ne1_SGD_80/glb_metrics.pkl", 'rb') as file:
    #     global_model_metrics = pickle.load(file)
    #     print(global_model_metrics)
    # schedule_time, rand_schedule_time, perc = cal_time_Perc(global_model_metrics)
    # print(f'Schedule Time: {schedule_time} | Rand. Schedule time: {rand_schedule_time} | Less by: {perc}%' )
