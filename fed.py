import copy
import torch
from models import Net, VGG11, ResNet18
from ResNet import ResNet50


def FedAvg(clients, schedule, device, num_classes, model, comm_method, in_channels):
    if model == 'Net':
        weighted_averaged_model = Net(num_classes, in_channels)
    elif model == 'ResNet50':
        weighted_averaged_model = ResNet50(num_classes, in_channels)
    elif model == 'ResNet18':
        weighted_averaged_model = ResNet18(num_classes)
    else:
        weighted_averaged_model = VGG11(num_classes)


    weighted_averaged_model.to(device)
    # Iterate through the layers of the models
    for param in weighted_averaged_model.parameters():
        param.data = torch.zeros_like(param.data)

    total_samples_number = total_samples_num(clients, schedule, comm_method)
    for client in clients:
        if client.client_id in schedule[clients[0].client_id] and client.client_id in schedule and comm_method == 'proposed':
            model = copy.deepcopy(client.aggr_model)
            samples_num = client.aggr_num_of_samples
        else:
            model = copy.deepcopy(client.model)
            samples_num = client.training_sample_num
            model.to(device)
        for param, param_avg in zip(model.parameters(), weighted_averaged_model.parameters()):
            param_avg.data += param.data * (samples_num / total_samples_number)
    # return weighted_averaged_model
    return copy.deepcopy(weighted_averaged_model), total_samples_number


def total_samples_num(clients, schedule, comm_method):
    total = 0
    for client in clients:
        if client.client_id in schedule[clients[0].client_id] and client.client_id in schedule and comm_method == 'proposed':
            total += client.aggr_num_of_samples
        else:
            total += client.training_sample_num
    return total
