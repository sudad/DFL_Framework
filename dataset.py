import torch
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.datasets import MNIST, CIFAR10
from pathlib import Path
import pickle
from medmnist import PathMNIST, TissueMNIST, OCTMNIST, OrganAMNIST

import matplotlib.pyplot as plt

def get_dataset(dataset_name: str, input_size, data_path: str = "./data"):
    """Download MNIST and apply minimal transformation."""
    if dataset_name == 'MNIST':
        tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        # tr = Compose([ToTensor(), Resize((224, 224), antialias=True), Normalize((0.1307,), (0.3081,))])
        train_set = MNIST(data_path, train=True, download=True, transform=tr)
        test_set = MNIST(data_path, train=False, download=True, transform=tr)
    elif dataset_name == 'CIFAR10':
        tr = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = CIFAR10(data_path, train=True, download=True, transform=tr)
        test_set = CIFAR10(data_path, train=False, download=True, transform=tr)
    elif dataset_name == 'PathMNIST':
        tr = Compose([ToTensor(), Normalize(mean=[.5], std=[.5])])
        train_set = PathMNIST(split="train", download=True, transform=tr, size=input_size)
        val_set = PathMNIST(split='val', transform=tr, download=True, size=input_size)
        test_set = PathMNIST(split='test', transform=tr, download=True, size=input_size)
    elif dataset_name == 'TissueMNIST':
        # tr = Compose([ToTensor(),  Resize((224, 224), antialias=True), Normalize(mean=[.5], std=[.5])])
        tr = Compose([ToTensor(), Normalize(mean=[.5], std=[.5])])
        train_set = TissueMNIST(split="train", download=True, transform=tr, size=input_size)
        val_set = TissueMNIST(split='val', transform=tr, download=True, size=input_size)
        test_set = TissueMNIST(split='test', transform=tr, download=True, size=input_size)
    elif dataset_name == 'OCTMNIST':
        # tr = Compose([ToTensor(),  Resize((224, 224), antialias=True), Normalize(mean=[.5], std=[.5])])
        tr = Compose([ToTensor(), Normalize(mean=[.5], std=[.5])])
        train_set = OCTMNIST(split="train", download=True, transform=tr, size=input_size)
        val_set = OCTMNIST(split='val', transform=tr, download=True, size=input_size)
        test_set = OCTMNIST(split='test', transform=tr, download=True, size=input_size)
    elif dataset_name == 'OrganAMNIST':
        # tr = Compose([ToTensor(),  Resize((224, 224), antialias=True), Normalize(mean=[.5], std=[.5])])
        tr = Compose([ToTensor(), Normalize(mean=[.5], std=[.5])])
        train_set = OrganAMNIST(split="train", download=True, transform=tr, size=input_size)
        val_set = OrganAMNIST(split='val', transform=tr, download=True, size=input_size)
        test_set = OrganAMNIST(split='test', transform=tr, download=True, size=input_size)
    else:
        return False

    if dataset_name == 'PathMNIST' or dataset_name == 'TissueMNIST' or dataset_name == 'OCTMNIST' or dataset_name == 'OrganAMNIST':
        return train_set, val_set, test_set
    else:
        return train_set, None, test_set


def prepare_dataset(dataset_name, save_path, NonIID, num_classes, input_size, num_partitions: int,
                    batch_size: int, val_ratio: float = 0.1):
    """Download MNIST and generate IID partitions.
    :type dataset_name: object
    """
    # download MNIST in case it's not already in the system
    if dataset_name == 'PathMNIST' or dataset_name == 'TissueMNIST' or dataset_name == 'OCTMNIST' or dataset_name == 'OrganAMNIST':
        train_set, val_set, test_set = get_dataset(dataset_name, input_size)
    else:
        train_set, _, test_set = get_dataset(dataset_name, input_size)

    train_loaders = []
    val_loaders = []
    if NonIID:
        # processing the validation dataset for each client
        val_indices = {}
        for i in range(num_partitions):
            val_indices[i] = []
        num_images = len(train_set) // num_partitions
        num_val = int(0.1 * num_images)
        for i in range(num_val * num_partitions):
            client = i // num_val
            val_indices[client].append(i)
        subset_indices = []
        for index in val_indices:
            subset_indices.append(val_indices[index])
        clients_val_datasets = [Subset(train_set, indices) for indices in subset_indices]

        # processing the training dataset for each client
        indices = {}
        # Creating a list of indices for each class
        for i in range(num_partitions):
            indices[i] = []
        # Add the index of each image to its list based on the image class
        # loop will exclude the images used in creating validation dataset
        for i in range(num_val * num_partitions, len(train_set)):
            _, label = train_set[i]
            indices[label].append(i)
        # clients_training_datasets will hold the images of each class, They can be assigned to clients
        subset_indices = []
        for index in indices:
            subset_indices.append(indices[index])
        clients_training_datasets = [Subset(train_set, indices) for indices in subset_indices]

        # Creating the training and testing dataloaders for all clients
        for i, client_dataset in enumerate(clients_training_datasets):
            train_loaders.append(DataLoader(client_dataset, batch_size=batch_size, num_workers=2))
        for i, client_dataset in enumerate(clients_val_datasets):
            val_loaders.append(DataLoader(client_dataset, batch_size=batch_size, num_workers=2))
    # If IID
    else:
        clients_val_datasets = []
        clients_training_datasets = []

        # split trainset into `num_partitions` trainsets (one per client)
        # figure out number of training examples per partition
        num_images = len(train_set) // num_partitions
        # a list of partition lenghts (all partitions are of equal size)
        partition_len = [num_images] * num_partitions

        # for each train set, let's put aside some training examples for validation
        if dataset_name == 'PathMNIST' or dataset_name == 'TissueMNIST' or dataset_name == 'OCTMNIST' or dataset_name == 'OrganAMNIST':
            # train_set.montage(length=3)
            client_indices = []
            subset_indices = []
            for i in range(len(train_set)):
                if i % num_images == 0 and len(client_indices) != 0:
                    subset_indices.append(client_indices)
                    client_indices = []
                client_indices.append(i)
                # clients_training_datasets will hold the images of each class, They can be assigned to clients

            clients_training_datasets = [Subset(train_set, indices) for indices in subset_indices]
            for client_dataset in clients_training_datasets:
                train_loaders.append(DataLoader(client_dataset, batch_size=batch_size, num_workers=2))

            #validation dataset
            client_indices = []
            subset_indices = []
            num_val_images = len(val_set) // num_partitions

            for i in range(len(val_set)+1):
                if i % num_val_images == 0 and len(client_indices) != 0:
                    subset_indices.append(client_indices)
                    client_indices = []
                client_indices.append(i)
                # clients_training_datasets will hold the images of each class, They can be assigned to clients
            clients_val_datasets = [Subset(val_set, indices) for indices in subset_indices]
            for i, client_dataset in enumerate(clients_val_datasets):
                val_loaders.append(DataLoader(client_dataset, batch_size=batch_size, num_workers=2))
        else:
            trainsets = random_split(train_set, partition_len, torch.Generator().manual_seed(2024))

            for trainset_ in trainsets:
                num_total = len(trainset_)
                num_val = int(val_ratio * num_total)
                num_train = num_total - num_val

                for_train, for_val = random_split(
                    trainset_, [num_train, num_val], torch.Generator().manual_seed(2024)
                )
                clients_training_datasets.append(for_train)
                clients_val_datasets.append(for_val)
                # construct data loaders and append to their respective list.
                # In this way, the i-th client will get the i-th element in the train_loaders list and the i-th element
                # in the val_loaders list
                train_loaders.append(
                    DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
                )
                val_loaders.append(
                    DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
                )
    data_distribution = {}
    train_set_dist = {}
    # Display the distribution of labels for each client
    print("The distribution of clients' training dataset ")

    for i, client_dataset in enumerate(clients_training_datasets):
        label_counts = {label: 0 for label in range(num_classes)}
        for _, label in client_dataset:
            if dataset_name == 'MNIST' or dataset_name == "CIFAR10":
                label_counts[label] += 1
            else:
                label_counts[label[0]] += 1
        print(f"Client {i + 1} Label Distribution: {label_counts}")
        train_set_dist[i + 1] = label_counts
    print('===================================')
    print("The distribution of clients' validation dataset ")
    val_set_dist = {}
    # Display the distribution of labels for each client
    for i, client_dataset in enumerate(clients_val_datasets):
        label_counts = {label: 0 for label in range(num_classes)}
        for _, label in client_dataset:
            if dataset_name == 'MNIST' or dataset_name == "CIFAR10":
                label_counts[label] += 1
            else:
                label_counts[label[0]] += 1
        print(f"Client {i + 1} Label Distribution: {label_counts}")
        val_set_dist[i + 1] = label_counts

    # Adding the training and validation dataset to the data distribution set to save it as a pickle file
    data_distribution['train'] = train_set_dist
    data_distribution['val'] = val_set_dist
    # Saving the clients as a pickle file
    result_path = Path(save_path) / 'data_dist.pkl'
    with open(str(result_path), "wb") as h:
        pickle.dump(data_distribution, h, protocol=pickle.HIGHEST_PROTOCOL)

    # We leave the test set intact (i.e. we don't partition it)
    test_loader = DataLoader(test_set, batch_size=64)

    # examples = enumerate(train_loaders[0])
    # batch_idx, (example_data, example_targets) = next(examples)
    # fig = plt.figure()
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # fig.show()

    return train_loaders, val_loaders, test_loader
