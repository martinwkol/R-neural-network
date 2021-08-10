#' Trainer class
#'
#' @description
#' This class is responsible for managing the training
#' process of a given network for a given optimizer.
#'
Trainer <- R6::R6Class("Trainer",
private = list(
  neuralnet = NULL,
  optimizer = NULL,
  training_data = list(),
  test_data = list(),
  accuracy_measurement = NULL,

  last_test_result = NULL,
  best_test_result = -Inf,
  best_neuralnet = NULL
),
public = list(
  initialize = function(neuralnet, optimizer,
                        training_data = NULL, test_data = NULL,
                        accuracy_measurement = NULL) {
    stopifnot("The neural network is null" = !is.null(neuralnet))
    stopifnot("The optimizer is null" = !is.null(optimizer))
    stopifnot("An accuracy_measurement function is missing" =
                neuralnet$category == "classification" ||
                !is.null(accuracy_measurement))

    private$neuralnet <- neuralnet
    private$optimizer <- optimizer
    private$training_data <- training_data
    private$test_data <- test_data

    if (is.null(accuracy_measurement)) {
      if (neuralnet$category == "classification") {
        private$accuracy_measurement <- measurement_classification()
      } else {
        stop("An accuracy_measurement function is missing")
      }
    } else {
      private$accuracy_measurement <- accuracy_measurement
    }

    self$reset()
  },

  setNeuralnet = function(neuralnet) {
    stopifnot("The neural network is null" = !is.null(neuralnet))
    private$neuralnet <- neuralnet
    self$reset()
  },
  setOptimizer = function(optimizer) {
    stopifnot("The optimizer is null" = !is.null(optimizer))
    private$optimizer <- optimizer
    private$optimizer$reset()
  },
  setTrainingData = function(training_data) {
    private$training_data <- training_data
  },
  setTestData = function(test_data) {
    private$test_data <- test_data
  },

  getTrainingData = function() private$training_data,
  getTestData = function() private$test_data,
  getBestNeuralnet = function() private$best_neuralnet,

  swapWithBestNeuralnet = function() {
    if (!is.null(private$best_neuralnet)) {
      self$setNeuralnet(private$best_neuralnet)
    }
  },

  seperate = function(data, test_percentage = 0.15) {
    stopifnot("Test percentage out of range" = 0 <= test_percentage && test_percentage <= 1)

    shuffled <- sample(data)
    test_data_length <- round(length(data) * test_percentage)
    private$test_data <- shuffled[1:test_data_length]
    private$training_data <- shuffled[(test_data_length + 1):length(data)]
  },

  test = function(N) {
    stopifnot("No test data is given" = !is.null(private$test_data))

    N <- min(N, length(private$test_data))
    accuracyVals <- sapply(private$test_data[seq(N)], function(td) {
      #print(td)
      netResult <- private$neuralnet$calculate(td$input)
      netOutput <- netResult$output
      #print(netResult)
      #print(netOutput)
      #print(td$expectedOutput)
      private$accuracy_measurement(netOutput, td$expectedOutput)
    })
    private$last_test_result <- sum(accuracyVals) / N
    if (private$last_test_result > private$best_test_result) {
      private$best_test_result <- private$last_test_result
      private$best_neuralnet <- rlang::duplicate(private$neuralnet)
    }
    private$last_test_result
  },

  #'
  #' @description
  #' train trains the neural network of the trainer with the given optimizer
  #'
  #' @param epochs integer; the number of epochs, the network will be trained
  #' @param training_per_epoch integer; the amout of training data that will be used for an epoch
  #' @param use_early_stopping logical; if true, early stopping is enabled
  #' @param es_test_frequency integer; if early stopping is enabled,
  #' test the performance of the network every \code{es_test_frequency} training sessions
  #' @param es_test_size integer; if early stopping is enabled and the performance of
  #' the network is to be evaluated, test the network with \code{es_test_size} test objects
  #' @param es_minimal_improvement double; if early stopping is enabled, stop the training,
  #' if the measured success ratio of the network didn't improve by at least
  #' \code{es_minimal_improvement} in comparison to the best measured success ratio
  #' for this network. Negative values are allowed.
  #' @seealso ?NeuralNet
  #' @export
  train = function(epochs, training_per_epoch = Inf, use_early_stopping = F, es_test_frequency = 5000,
                   es_test_size = 500, es_minimal_improvement = 0) {
    stopifnot("No trainung data is given" = !is.null(private$training_data))
    stopifnot("The number of epochs has to be as least 0" = epochs >= 0)
    stopifnot("The amount of trained data per epoch has to be bigger than 0"
              = training_per_epoch > 0)
    if (use_early_stopping) {
      stopifnot("No test data is given" = !is.null(private$test_data))
      stopifnot("Test frequency has to be bigger than 0" = es_test_frequency > 0)
      stopifnot("Test size has to be bigger than 0" = es_test_size > 0)
    }

    training_per_epoch <- min(training_per_epoch, length(private$training_data))

    if (use_early_stopping) {

      trained_data_since_test <- 0
      for (epoch in seq(epochs)) {
        shuffled_td <- sample(private$training_data)[seq(training_per_epoch)]

        start_index <- 1
        while(start_index <= length(shuffled_td)) {
          end_index <- min(start_index + es_test_frequency - 1, length(shuffled_td))

          private$optimizer$optim(private$neuralnet,
                                  shuffled_td[start_index:end_index],
                                  N = length(shuffled_td))

          trained_data_since_test <- trained_data_since_test +
                                    (end_index - start_index + 1)
          if(trained_data_since_test >= es_test_frequency) {
            trained_data_since_test <- trained_data_since_test - es_test_frequency
            test_result <- self$test(es_test_size)
            diff <- test_result - private$best_test_result
            if (diff < es_minimal_improvement) {
              return (T)
            }
          }

          start_index <- end_index + 1
        }
      }

    } else {

      for (epoch in seq(epochs)) {
        shuffled_td <- sample(private$training_data)[seq(training_per_epoch)]
        private$optimizer$optim(private$neuralnet, shuffled_td)
      }

    }

    return(F)
  },

  reset = function() {
    private$last_test_result <- NULL
    private$best_test_result <- -Inf
    private$best_neuralnet <- NULL
    private$optimizer$reset()
  }
))
