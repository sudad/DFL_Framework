class Client:
    def __init__(self, client_id, model, train_loader, validation_loader, epoch, comp_time, neighbors):
        self.client_id = client_id  # The client id
        self.training_sample_num = len(train_loader.dataset)  # len(train_loader.dataset)  # Number of samples that the client will use for training
        self.validation_sample_num = len(validation_loader.dataset)   # len(validation_loader.dataset) # Number of samples that the client will use for
        # validating (testing)
        self.model = model  # The client model
        self.train_loader = train_loader  # The training dataset that the client has
        self.validation_loader = validation_loader  # The validating dataset that the client has
        # self.optimizer = optimizer  # The optimizer used in each client
        self.local_epochs = epoch  # Local epoch each client will iterate training the model
        self.local_train_losses = {}  # Storing the train losses of all local training rounds
        self.local_test_losses = {}  # Storing the test losses of all local testing rounds
        self.local_test_accuracies = {}  # Storing the test accuracies of all local testing rounds

        self.glb_test_losses = {}  # Storing the test losses of all global testing rounds
        self.glb_test_accuracies = {}  # Storing the test accuracies of all global testing rounds

        # Graph info
        self.neighbours = neighbors  # Dict of all neighbours and the communication time with them
        self.comp_time = comp_time  # The time the client takes to complete the training

        self.neighbours_with_weights = {}
        # adding the comp. time and communication time to represent the cost for sending info to other neighbours

        self.aggr_model = None
        self.aggr_num_of_samples = None

    def update_aggr_model_and_num_of_samples(self, weighted_averaged_model, total_samples_number):
        self.aggr_model = weighted_averaged_model
        self.aggr_num_of_samples = total_samples_number

    def prepare_client_for_scheduling(self):
        for key in self.neighbours:
            self.neighbours_with_weights[key] = self.neighbours[key] + self.comp_time
