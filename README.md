# Functionality

This package provides functionality to built user-defined feedforward
Neural Networks for classification and regression <br> You can

-   use different activation functions
-   choose between different algorithms
-   split data into test- and trainsets
-   evaluate the results
-   visualize the Neural Network

<br>

# Usage

## 1. Create a Neural Network

First you have to create a Neural Network.

This will generate a Neural Network for classification with two input
nodes, two output nodes and two hidden layers with four nodes each:

`nn <- NeuralNet$new(c(2,4,4,2), category = "classification")`

<br>

This will generate a Neural Network for regression using the predefined
sigmoid function as activation function:

    nn <- NeuralNet$new(c(2,4,4,1), activationfct = "sigmoid", 
                        category = "regression")

## 2. Configure Optimizer

You can choose between three optimizer algorithms:

-   Stochastic Gradient Descent

-   Stochastic Gradient Descent with Momentum

-   Nesterov accelerated gradient

This will generate a Stochastic Gradient Descent with Momentum Optimizer
with a learning_rate of 0.0005, regularization_rate of 0 and a
momentum_term of 0.9:

`optimizer <- OptimizerMomentum$new(0.0005, 0, 0.9)`

<br>

The previous set values can be changed:

       optimizer$setLearningRate(0.0001)
       optimizer$setRegularizationRate(0.00001)
       optimizer$setMomentumTerm(0.8)

## 3. Create a Trainer

The trainer class manages the training and the testing process of a
given network for a given optimizer.

This generates a trainer:

    trainer <- Trainer$new(nn, optimizer)

If you want, you can already give the trainer the training data and the
test data that you want to use. The training data and the test data each
have to be lists in the format that you can see below.

    training_data <- list(
        list(input = 1:784, expectedOutput = 2),
        list(input = 784:1, expectedOutput = 10) # ,
        # ...
    )
    test_data <- list(
        list(input = 1:784 * 2, expectedOutput = 1) # ,
        # ...
    )
    trainer <- Trainer$new(nn, optimizer,
                           training_data,
                           test_data))

If your neural network uses regression, you must specify an accuracy
tester like so:

    accuracy_tester <- accuracy_tester_regression_rel(0.1)
    trainer <- Trainer$new(nn, optimizer, accuracy_tester = accuracy_tester)

That means that the trainer will consider outputs by the network that
don’t deviate from the expected result by more than 10% (relative) as
accurate. <br> You can also use an absolute accuracy tester:

    accuracy_tester <- accuracy_tester_regression_abs(5)
    trainer <- Trainer$new(nn, optimizer, accuracy_tester = accuracy_tester)

In this case the trainer will consider outputs by the network that don’t
deviate from the expected result by more than 5 (absolute) as accurate.

## 4. Prepare Data

Often times the training / test data is divided into a list of inputs
and target values. You can combine these lists to one list, that the
trainer will be able to read using the combineData method:

    training_inputs <-
       list(c(0.5, 0.5), c(0.1, 0.8))
    training_targets <-
       list(2, 1)
    training_data <- combineData(training_inputs, training_targets)
    test_inputs <-
       list(c(0.5, 0.5), c(0.1, 0.8))
    test_targets <-
       list(2, 1)
    test_data <- combineData(test_inputs, test_targets)
    trainer$setTrainingData(training_data)
    trainer$setTestData(test_data)

You can use the seperateData method to let the trainer devide your data
into training and test data:

    inputs <-
       list(c(0.5, 0.5), c(0.1, 0.8))
    targets <-
       list(2, 1)
    data <- combineData(inputs, targets)
    trainer$seperateData(data, test_percentage = 0.15)

You can combine the methods combineData and seperateData with the method
gererateTrainingTest:

    inputs <-
       list(c(0.5, 0.5), c(0.1, 0.8))
    targets <-
       list(2, 1)
    data <- combineData(inputs, targets)
    trainer$generateTrainingTest(inputs, targets, test_percentage = 0.15)

## 5. Training

You can train your data with the train method:

    trainer$train(epochs = 10, training_per_epoch = 10000)

The parameter “epochs” is the number of epochs, the network will be
trained The parameter ” training_per_epoch” is the amount of training
data that will be used for an epoch (if you don’t want to use all of
your training data in an epoch). <br> The train method supports early
stopping:

    trainer$train(10, training_per_epoch = 10000, use_early_stopping = T,
                  es_test_frequency = 1000, es_test_size = 100,
                  es_minimal_improvement = -0.05)

If use_early_stopping is set to true, the method will test the network
for every es_test_frequency (here: 1000) training data points with
es_test_size (here: 1000) many test data points. If the accuracy of the
network is smaller than the best accuracy (overall of this network with
this trainer) of the network + es_minimal_improvement (here: -0.05), the
method aborts the training. The method will return true, if the training
was finished without early stopping and false if the training was
aborted Examples of training with different parameters:

    trainer$train(1, use_early_stopping = T,
                  es_test_frequency = 1000, es_test_size = 100,
                  es_minimal_improvement = -0.05)
    trainer$train(1, use_early_stopping = T,
                  es_test_frequency = 1000, es_test_size = 100,
                  es_minimal_improvement = -0.02)
    trainer$train(1, 500)
    trainer$train(1, 1000)
    trainer$train(1, 100)

You can test the network with the test function:

    trainer$test(N = 500)

The parameter N specifies with how many test data points the neural
network is supposed to be tested. If you don’t specify N, the network
will be tested with all test data available. The method returns the
ratio of correctly calculated outputs divided by the number of tests
performed. You can swap the currently used neural network of the with
the copy of the neural network, that scored the best using the following
code:

    trainer$swapWithBestNeuralnet()

You can also get the best neural network with the following code:

    trainer$getBestNeuralnet()

## 5. Visualization

This will give you a visualization of the Neural Network:

    nn$plot()
