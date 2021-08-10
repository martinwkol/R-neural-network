#' Trainer class
#'
#' @description
#' This class is responsible for managing the training
#' process of a given network for a given optimizer.
#'
#' @export
Trainer <- R6::R6Class("Trainer",
private = list(
  # The neural network to be trained
  neuralnet = NULL,

  # The optimizer used for training the neural network
  optimizer = NULL,

  # The list of the training data
  training_data = list(),

  # The list of the test data
  test_data = list(),

  # A method that compares the output of the
  # neural network for a given input
  # with the expected output.
  # If the output of the network is considered correct,
  # the method returns true, otherwise false
  accuracy_tester = NULL,



  # Stores the last test result scored
  # with the neural network
  last_test_result = NULL,

  # Stores the best test result scored
  # with the neural network (since the last reset)
  best_test_result = -Inf,

  # Stores a copy of the neural network, that scored the
  # best test result (since the last reset)
  best_neuralnet = NULL
),
public = list(
  #' @description
  #' Initializes a new Trainer. It resets the given
  #' optimizer.
  #'
  #' @param neuralnet The neural network to be trained
  #' @param optimizer The optimizer used for training the neural network
  #' @param training_data The list of the training data
  #' @param test_data The list of the test data
  #' @param accuracy_tester A method that must compare
  #' the output of the neural network for a given input
  #' with the expected output.
  #' If the output of the network is considered correct,
  #' the method must returns true, otherwise false.
  #' If \code{accuracy_tester} is \code{NULL}, and the
  #' given neural network uses classification, the Trainer
  #' generates the standard \code{accuracy_tester} method
  #' for classification (recomended).
  #' If the neural network uses regression, a custom
  #' \code{accuracy_tester} method must be given
  #' @seealso ?NeuralNet
  #' @export
  initialize = function(neuralnet, optimizer,
                        training_data = NULL, test_data = NULL,
                        accuracy_tester = NULL) {
    stopifnot("The neural network is null" = !is.null(neuralnet))
    stopifnot("The optimizer is null" = !is.null(optimizer))
    stopifnot("An accuracy_tester function is missing" =
                neuralnet$category == "classification" ||
                !is.null(accuracy_tester))

    private$neuralnet <- neuralnet
    private$optimizer <- optimizer
    private$training_data <- training_data
    private$test_data <- test_data

    if (is.null(accuracy_tester)) {
      if (neuralnet$category == "classification") {
        private$accuracy_tester <- accuracy_tester_classification()
      } else {
        stop("An accuracy_tester function is missing")
      }
    } else {
      private$accuracy_tester <- accuracy_tester
    }

    self$reset()
  },

  #' @description
  #' Changes the neural network to train.
  #' After the change of the network, the method
  #' resets the optimiser and forgets the last test score,
  #' the best test score and the best neural network
  #'
  #' @param neuralnet The new neural network to be trained
  #'
  #' @seealso ?NeuralNet
  #' @export
  setNeuralnet = function(neuralnet) {
    stopifnot("The neural network is null" = !is.null(neuralnet))
    private$neuralnet <- neuralnet
    self$reset()
  },

  #' @description
  #' Changes the optimizer used for training the neural network
  #' The method also resets the new optimizer
  #'
  #' @param optimizer The optimizer used for training
  #' the neural network
  #'
  #' @export
  setOptimizer = function(optimizer) {
    stopifnot("The optimizer is null" = !is.null(optimizer))
    private$optimizer <- optimizer
    private$optimizer$reset()
  },

  #' @description
  #' Sets the training data
  #'
  #' @param training_data The list of the training data
  #'
  #' @export
  setTrainingData = function(training_data) {
    private$training_data <- training_data
  },

  #' @description
  #' Sets the test data
  #'
  #' @param test_data The list of the test data
  #'
  #' @export
  setTestData = function(test_data) {
    private$test_data <- test_data
  },


  #' @return The list of the training data
  #' If the Trainer doesn't have training data, it
  #' returns \code{NULL}
  #'
  #' @export
  getTrainingData = function() private$training_data,
  #' @return The list of the test data
  #' If the Trainer doesn't have test data, it
  #' returns \code{NULL}
  #'
  #' @export
  getTestData = function() private$test_data,
  #' @return A copy of the neural network, that scored the
  #' best test result. If no test was performed with the
  #' current neural network (since the last reset), the
  #' method returns \code{NULL}
  #'
  #' @export
  getBestNeuralnet = function() private$best_neuralnet,

  #' @description
  #' Swaps the currently used neural network with the
  #' copy of the neural network, that scored the best
  #' test result. If such a copy does not exists, the
  #' method does nothing
  #'
  #' @export
  swapWithBestNeuralnet = function() {
    if (!is.null(private$best_neuralnet)) {
      self$setNeuralnet(private$best_neuralnet)
    }
  },

  #' @description
  #' Separates the given data into training data and test data
  #' The parameter \code{test_percentage} determines what portion
  #' of the data will be used for testing and consequently
  #' what portion will be used for training.
  #' If the trainer already has training and / or test data,
  #' the old data will be overwritten.
  #'
  #' @param data A list of lists that must each contain the
  #' named components \code{input} and \code{expectedOutput};
  #' The data to be separated into training data and test data
  #' @param test_percentage The portion
  #' of the data that will be used for testing
  #'
  #' @examples
  #' data <- list(
  #'   list(input = c(0.5, 0.5),
  #'        expectedOutput = 2),
  #'   list(input = c(0.1, 0.8),
  #'        expectedOutput = 1)
  #' )
  #' trainer$separateData(data,
  #'            test_percentage = 0.5)
  #'
  #' @export
  separateData = function(data, test_percentage = 0.15) {
    stopifnot("data is not a list" = is.list(data))
    stopifnot("Test percentage out of range" = 0 <= test_percentage && test_percentage <= 1)
    for (d in data) {
      stopifnot("Not every element of the datalist
                has an input and an expected output
                value" = all(c("input", "expectedOutput") %in% names(d)))
    }

    shuffled <- sample(data)
    test_data_length <- round(length(data) * test_percentage)
    private$test_data <- shuffled[1:test_data_length]
    private$training_data <- shuffled[(test_data_length + 1):length(data)]
  },

  #' @description
  #' Creates training data list and test data list from the given
  #' inputs and targets
  #' The parameter \code{test_percentage} determines what portion
  #' of the data will be used for testing and consequently
  #' what portion will be used for training.
  #' If the trainer already has training and / or test data,
  #' the old data will be overwritten.
  #'
  #' @param inputs A list with the input data
  #' @param targets A list with the target data
  #' @param test_percentage The portion
  #' of the data that will be used for testing
  #'
  #' @examples
  #' inputs <-
  #'   list(c(0.5, 0.5), c(0.1, 0.8))
  #' targets <-
  #'   list(2, 1)
  #' trainer$generateTrainingTest
  #'   (inputs, targets,
  #'    test_percentage = 0.5)
  #'
  #' @export
  generateTrainingTest = function(inputs, targets, test_percentage = 0.15) {
    combined <- combineData(inputs, targets)
    self$separateData(combinded, test_percentage)
  },

  #' @description
  #' Tests the neural network on the first \code{N} elements
  #' of the test data list. It will do so by calculating the
  #' output of the network for the first \code{N} inputs in the
  #' test data list and comparing the network output with the
  #' expected output using the \code{accuracy_tester} method.
  #' The method will then calculate the proportion of correctly
  #' calculated outputs and save the result. If the test result
  #' is better than the previous best test result, the method
  #' will store the new best test result and save the current
  #' network as the best network.
  #' If the trainer does not have a test data list,
  #' the method throws an error.
  #'
  #' @param N Number of tests to perform. If N is bigger than
  #' the length of the test data list, the method will replace
  #' N with the length of the test data list
  #'
  #' @return proportion of correctly calculated outputs
  #'
  #' @export
  test = function(N = Inf) {
    stopifnot("No test data is given" = !is.null(private$test_data))

    N <- min(N, length(private$test_data))
    accuracyVals <- sapply(private$test_data[seq(N)], function(td) {
      #print(td)
      netResult <- private$neuralnet$calculate(td$input)
      netOutput <- netResult$output
      #print(netResult)
      #print(netOutput)
      #print(td$expectedOutput)
      as.integer(private$accuracy_tester(netOutput, td$expectedOutput))
    })
    private$last_test_result <- sum(accuracyVals) / N
    if (private$last_test_result > private$best_test_result) {
      private$best_test_result <- private$last_test_result
      private$best_neuralnet <- rlang::duplicate(private$neuralnet)
    }
    private$last_test_result
  },

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
  #'
  #' @return True if the training was completed, false otherwise
  #' (early stopping)
  #'
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
                                  shuffled_td[start_index:end_index])

          trained_data_since_test <- trained_data_since_test +
                                    (end_index - start_index + 1)
          if(trained_data_since_test >= es_test_frequency) {
            trained_data_since_test <- trained_data_since_test - es_test_frequency
            test_result <- self$test(es_test_size)
            diff <- test_result - private$best_test_result
            if (diff < es_minimal_improvement) {
              return (F)
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

    return(T)
  },

  #' @description
  #' The trainer forgets the last test result, the best test
  #' result and the best network. The optimizer gets reset-ted.
  #'
  #' @export
  reset = function() {
    private$last_test_result <- NULL
    private$best_test_result <- -Inf
    private$best_neuralnet <- NULL
    private$optimizer$reset()
  }
))

#' Combine input and target lists
#'
#' @description
#' Combines the given inputs and targets into one list that
#' can then be given to the trainer as a training / test data
#' list.
#'
#' @param inputs A list with the input data
#' @param targets A list with the target data
#'
#' @return The combined data list
#'
#' @examples
#' inputs <-
#'   list(c(0.5, 0.5), c(0.1, 0.8))
#' targets <-
#'   list(2, 1)
#' combined <-
#'   combineData
#'     (inputs, targets)
#'
#' @export
combineData <- function(inputs, targets) {
  stopifnot("'inputs' and 'targets' have a different length" = length(inputs) == length(targets))

  combinded <- mapply(function(i, t) list(input = i, expectedOutput = t),
                      inputs, targets, SIMPLIFY = F)
  combinded
}
