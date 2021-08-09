Trainer <- R6::R6Class("Trainer",
private = list(
  optimiser = NULL,
  training_data = list(),
  test_data = list()
),
public = list(
  initialize = function(optimiser, training_data, test_data) {
    private$optimiser <- optimiser
    private$training_data <- training_data
    private$test_data <- test_data
  },

  setTrainingData = function(training_data) {
    private$training_data <- training_data
  },
  setTestData = function(test_data) {
    private$test_data <- test_data
  },

  getTrainingData = function() private$training_data,
  getTestData = function() private$test_data,

  test = function(neuralnet, N) {
    N <- min(N, length(private$test_data))
    accuracyFunc <- NULL
    if(neuralnet$category == "classification") {
      accuracyFunc <- function(netOutput, expectedOutput)
        as.double(netOutput == expectedOutput)
    } else {
      accuracyFunc <- function(netOutput, expectedOutput)
        min((netOutput - expectedOutput)**2, 1)
    }
    accuracyVals <- sapply(sample(private$test_data, N), function(td) {
      #print(td)
      netResult <- neuralnet$calculate(td$input)
      netOutput <- netResult$output
      #print(netResult)
      #print(netOutput)
      #print(td$expectedOutput)
      accuracyFunc(netOutput, td$expectedOutput)
    })
    sum(accuracyVals) / N
  },

  #' train - Stochastic Gradient Descent
  #'
  #' train implements an Algorithm for training a \code{?NeuralNet} Neural Network
  #' using Stochastic Gradient Descent.
  #'
  #' @param neuralnet A R6 Neural Network that will be trained with the given data
  #' @param training_data a data set used to train the Neural Network
  #' @param learing_rate the learning rate to be used by the Algorithm
  #' @param lamda a lambda to be used by the Algorithm
  #' @seealso ?NeuralNet
  #' @export
  train = function(neuralnet, learning_rate, lambda, N) {
    private$optimiser$setLearningRate(learning_rate)
    private$optimiser$setLambda(lambda)
    private$optimiser$optim(neuralnet, sample(private$training_data, N, replace = T) )
  }
))
