import pandas as pd
import numpy as np
import itertools as it
import time
import matplotlib.pyplot as plt

from network import *
from functions import *
from utility import *

dizionario = {
'layers_size': [[len(training[0][0]), 15, 1]],
'activation_function': [leaky_relu, sigmoid, softplus],
'last_activation_function': [tanh],
'regularization': [0.001, 0.0001, 0.01],
'initialization_w': ['range2'],
'momentum': [0.5, 0.6, 0.7, 0.8, 0.9],
'learning_rate': [0.4, 0.5, 0.6, 0.7, 0.8],
'loss_function': [euclideanLoss],
'max_epochs': [500],
'batch': [6]
}
dizionario_cup = {
'layers_size': [[len(training[0][0]), 40, 40, 40, 2]],
'activation_function': [sigmoid],
'last_activation_function': [identity],
'regularization': [0.01],
'initialization_w': ['range7'],
'momentum': [0.9],
'learning_rate': [0.01],
'loss_function': [euclideanLoss],
'max_epochs': [300],
'batch': [30]
}



'''
    create_combinations takes a dictionary with hyperparameters and their possible values
    It creates all the possible combinations of these hyper-parameters values.
    Returns a list of network.
'''
def create_combinations(dict_of_hyperparameters):
    models = []
    #create the list of all parameters combinations
    combinations = list(it.product(*(dict_of_hyperparameters[k] for k in dict_of_hyperparameters.keys())))
    print('Combinazioni da provare: ', len(combinations))
    print()
    #create the network for each combination of parameters
    for combination in combinations:
        network = Network(layers_size = combination[0],
            activation_function= combination[1], last_activation_function = combination[2], regularization=combination[3],
            initialization_w =combination[4], learning_rate = combination[6], loss_function = combination[7], max_epochs= combination[8], momentum = combination[5])
        models.append((network, combination[-1]))
    return models



'''
    try_model takes a network, a dataset and the batch size and train the model
'''
def try_model(network, data, batch):
    network = network.compile()
    net_copy = network.get_null_copy()

    for epoch in range(network.max_epochs):
        random.shuffle(data)
        error_at_epoch = []

        #cycle for the batch size
        for b in range (int(math.ceil(len(data)/batch))):
            error_deriv = 0.
            grad_net = network.get_null_copy()
            for x, target in data[b*batch : (b+1)*batch]:
                out = network.feed_forward(x) #[12, 30]
                error_deriv += network.loss_function.derivative(out, target)
                network.back_propagation(error_deriv)

                # update gradients
                for j, layer in enumerate(network.layers):
                    gradient_w, gradient_bias = layer.compute_gradients()
                    grad_net.layers[j].w += gradient_w - 2*network.regularization * net_copy.layers[j].w
                    grad_net.layers[j].bias += gradient_bias - 2*network.regularization * net_copy.layers[j].bias

            #apply the momentum and then the delta rule
            for j, layer in enumerate(network.layers):
                grad_net.layers[j].w /= batch
                grad_net.layers[j].bias /= batch

                net_copy.layers[j].w = network.momentum * net_copy.layers[j].w + (1 - network.momentum) * grad_net.layers[j].w

                net_copy.layers[j].bias *= network.momentum
                net_copy.layers[j].bias += (1 - network.momentum) * grad_net.layers[j].bias

                network.layers[j].w -= network.learning_rate * net_copy.layers[j].w
                network.layers[j].bias -= network.learning_rate * net_copy.layers[j].bias

    loss = network.avg_loss(data) #oppure va bene anche avg_loss RMSE_loss
    return loss



'''
    k_fold_crossvalidation takes:
    - k: the number of folds
    - data: dataset
    - network
    - batch
    - acc: True if we are interesting in computing also the accuracy

    It performes a k-fold cross validation.
    It returns:
    - avg_loss_train: average loss on the training,
    - avg_loss_val: average loss on the validation,
    - std_loss_train: standard deviation of the loss on the training,
    - std_loss_val: standard deviation of the loss on the validation,
'''
def k_fold_crossvalidation(k, data, network, batch, acc):
    #Creating k folds
    length_fold = math.ceil(len(data)/k)
    folds = []
    for i in range(k):
        folds.append(data[length_fold*i : length_fold* (i+1)])

    losses_CV_train = []
    losses_CV_val = []
    if(acc):
        acc_CV_train = []
        acc_CV_val = []
    # cycling k times
    for i in range(k):
        train = []
        val = []
        for j in range(k):
            if(i!=j):
                train.append(folds[j])
            else:
                val.append(folds[j])
        net = network.compile()
        print('.')
        #calcoliamo l'errore sul validation e training
        if (acc):
            losses_train, losses_val, acc_train, acc_val = net.fit( train[0], val[0], batch, acc)
        else:
            losses_train, losses_val = net.fit(train[0], val[0], batch, acc)
        if (net.layers_size[-1] > 1):
            losses_train = np.mean(losses_train[-1])
            losses_val = np.mean(losses_val[-1])
            losses_CV_train.append(losses_train)
            losses_CV_val.append(losses_val)
            if(acc):
                acc_train = np.mean(acc_train[-1])
                acc_val = np.mean(acc_val[-1])
                acc_CV_train.append(acc_train)
                acc_CV_val.append(acc_val)
        else:
            losses_CV_train.append(losses_train[-1])
            losses_CV_val.append(losses_val[-1])
            if(acc):
                acc_CV_train.append(acc_train[-1])
                acc_CV_val.append(acc_val[-1])

    # calcolo media e deviazione standard degli errori sul train e sul validation
    avg_loss_train = np.mean(losses_CV_train)
    avg_loss_val = np.mean(losses_CV_val)

    if(acc):
        avg_acc_train = np.mean(acc_CV_train)
        avg_acc_val = np.mean(acc_CV_val)

    std_loss_train = np.std(losses_CV_train)
    std_loss_val = np.std(losses_CV_val)
    if(acc):
        return avg_loss_train, avg_loss_val, std_loss_train, std_loss_val, avg_acc_train, avg_acc_val
    else:
        return avg_loss_train, avg_loss_val, std_loss_train, std_loss_val


'''
    save_gs_results takes a list of results of a grid search and save them into a txt file
'''
def save_gs_results(list_of_trials_gs, acc):
    path_file = f"results/{name_file}/_grid_search_details.txt"
    # save the results of each network into a txt file
    if(acc):
        with open(path_file, "w") as infile:
            for trial in list_of_trials_gs:
                infile.write("Network "f"{trial[0]}" + "\n" +
                    f"{trial[1].get_network_description()}" +
                    "\n Batch size:  " + f"{trial[2]}"  +
                    "\n Avg Error on validation:  " + f"{trial[3]}" +
                    "\n Avg Errors on training :  " + f"{trial[4]}"+
                    "\n Std Error on validation:  " + f"{trial[5]}" +
                    "\n Std Errors on training :  " + f"{trial[6]}"+
                    "\n Avg accuracy on training :  " + f"{trial[7]}"+
                    "\n Avg accuracy on validation :  " + f"{trial[8]}"+
                    "\n" + "\n"
                )
    else:
        with open(path_file, "w") as infile:
            for trial in list_of_trials_gs:
                infile.write("Network "f"{trial[0]}" + "\n" +
                    f"{trial[1].get_network_description()}" +
                    "\n Batch size:  " + f"{trial[2]}"  +
                    "\n Avg Error on validation:  " + f"{trial[3]}" +
                    "\n Avg Errors on training :  " + f"{trial[4]}"+
                    "\n Std Error on validation:  " + f"{trial[5]}" +
                    "\n Std Errors on training :  " + f"{trial[6]}"+
                    "\n" + "\n"
                )


'''
    A function that takes a list of networks and trains each of them.
    It returns a list such that:
        - 0: network index
        - 1: network
        - 2: average of the loss of the network on the validation
        - 3: loss of the network for each attempt on the validation
        - 4: computational time of fitting the network on the validation
    and the best network performing on the validation set
'''
def exhaustive_grid_search(dizionario, cv, acc):
    start_gs = time.time()
    list_of_models = create_combinations(dizionario)
    coarse_gs_results = []

    for i, model in enumerate(list_of_models):
        print()
        print('** Network', i)

        if(cv):
            if(acc):
                avg_train_loss, avg_val_loss, std_train_loss, std_val_loss, acc_train, acc_val = k_fold_crossvalidation(5, training, model[0], model[1], acc)
            else:
                avg_train_loss, avg_val_loss, std_train_loss, std_val_loss = k_fold_crossvalidation(5, training, model[0], model[1], acc)

            print('Error of this network (training): \t', avg_train_loss)
            print('Error of this network (validation): \t', avg_val_loss)

            if (math.isnan(avg_train_loss) or math.isnan(avg_val_loss)):
                avg_train_loss = math.inf
                avg_val_loss = math.inf
            if(acc):
                coarse_gs_results.append((i, model[0], model[1], avg_val_loss, avg_train_loss, std_val_loss, std_train_loss, acc_train, acc_val))
            else:
                coarse_gs_results.append((i, model[0], model[1], avg_val_loss, avg_train_loss, std_val_loss, std_train_loss))

        else:
            avg_loss = 0.
            for _ in range(5):
                print('.')
                loss_val = try_model(model[0], validation, batch = model[1])
                if(model[0].layers_size[-1] > 1):
                    loss_val = np.mean(loss_val)
                avg_loss += loss_val
            print('Avg error of this network: ', avg_loss/5)
            coarse_gs_results.append((i, model[0], model[1], avg_loss/5, float("nan"), np.std(avg_loss), float("nan")))

    coarse_gs_results.sort(key=lambda tup: tup[3])
    end_gs = time.time()
    print()
    print()
    print('Grid search executed in ', int((end_gs - start_gs)/60), 'minutes and ', int(end_gs - start_gs)%60, ' seconds')
    print()

    save_gs_results(coarse_gs_results, acc)
    return coarse_gs_results


def plot_learning_curve(training_losses, validation_losses):
    #Plotting the errors on the training and validation sets
    plt.figure()
    plt.plot(validation_losses, label="validation", color = 'red', ls='--')
    plt.plot(training_losses, label="training", color='blue')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MEE)")
    #plt.title("Error of the best network on the training and validation")
    plt.savefig(f"results/{name_file}/{l}learning_curve_bestNet.png", dpi=300)

def plot_accuracy(training_accuracies, validation_accuracies):
    plt.figure()
    plt.plot(validation_accuracies, label="validation", color = 'red', ls='--')
    plt.plot(training_accuracies, label="training", color = 'blue')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy ")
    #plt.title("Accuracy of the best network on the training and validation")
    plt.savefig(f"results/{name_file}/{l}accuracy_curve_bestNet.png", dpi=300)

def save_predictions(network, l):
    #compute and save predictions
    predictions = network.predict(blind_test)
    path_output_file = f"results/{name_file}/{l}_predictions.csv"
    with open(path_output_file, 'w') as outputfile:
        print('# Marica  Massari	Francesco Melighetti', file=outputfile)
        print('# name team', file=outputfile)
        print('# ML-CUP20', file=outputfile)
        print('# Submission Date 15/06/2021', file=outputfile)
        for i in range(len(predictions)):
            print(str(i + 1) + ',' + str(predictions[i][0]) + ',' + str(predictions[i][1]), file=outputfile)



'''
    use_best_network
     - takes a model and retrain it,
     - print the training and validation errors
     - print the learning accuracy_curve
     - predict the output for the cup
'''
def use_best_network(best_network, batch_size, training, validation, acc_output, l):
    #retraining the best_network
    if (len(validation) < 1):
        #validation = []
        n = int( (25* len(training))/100)
        validation = training[:n]
        training = training[n:]
    if(acc_output):
        training_losses, validation_losses, training_accuracies, validation_accuracies =  best_network.fit( training, validation, batch_size)
    else:
        training_losses, validation_losses  =  best_network.fit(training, validation, batch_size, acc_output)
    n_outputs = best_network.layers_size[-1]
    if(n_outputs>1):
        for i in range(len(training_losses)):
            training_losses[i] = np.mean(training_losses[i])
        for i in range(len(validation_losses)):
            validation_losses[i] = np.mean(validation_losses[i])

    print()
    print(f"*** Network {l} - {name_file} ***")
    print('   - Error on the training:      ', training_losses[-1])
    plot_learning_curve(training_losses, validation_losses)
    if(acc_output):
        print('   - Accuracy on the training: ', training_accuracies[-1])
        plot_accuracy(training_accuracies, validation_accuracies)

    #compute the test error
    loss_test = best_network.avg_loss(test)
    if(n_outputs > 1):
        loss_test = np.mean(loss_test)
    print('   - Error on the test set:      ', loss_test)
    if (acc_output):
        acc_test = best_network.avg_accuracy(test, binary = True)
        print('   - Accuracy on the test set: ',  acc_test)

    #save network configuration and errors
    path_file = f"results/{name_file}/{l}_models-details.txt"
    params_network = best_network.get_network_description()
    with open(path_file, "w") as infile:
        infile.write("Network " + str(l) + "\n" +
        "parameters:   " f"{params_network}" + "\n" +
        "\n" +
        "Error on training:" + str(training_losses[-1])+ "\n" +
        "Error on validation:   " + str(validation_losses[-1]) + "\n" +
        "Error on test:   " f"{loss_test}" + "\n")
    if(not acc_output):
        save_predictions(best_network, l)



if (name_file.startswith('monk')):
    accuracy_out = True
else:
    accuracy_out = False

if(len(validation) < 1):
    k_cv = True
else:
    k_cv = False

results_from_gs = exhaustive_grid_search(dizionario_cup, k_cv, accuracy_out)

for l in range(4):
    best_model = results_from_gs[l][1]
    batch = results_from_gs[l][2]
    #best_model_params = best_model.get_network_description()
    net = best_model.compile()

    use_best_network(net, batch, training, validation, accuracy_out, l)
