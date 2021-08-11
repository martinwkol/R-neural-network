#' Optimizer Stochastic Gradient Descent
#'
#' @description
#' This class is responsible for performing the
#' optimization method "Stochastic Gradient Descent"
#' on a neural network for a given list of training data
#'
OptimizerSGD <- R6::R6Class("OptimizerSGD",
private = list(
  learning_rate = 0,
  regularization_rate = 0
),
public = list(
  #' @description
  #' Initializes a new OptimiserSGD object
  #'
  #' @param learning_rate The learning rate used by the optimization method
  #' @param regularization_rate The regularization rate used by the optimization method
  #'
  #' @export
  initialize = function(learning_rate, regularization_rate) {
    private$learning_rate <- learning_rate
    private$regularization_rate <- regularization_rate
  },

  #' @description
  #' Sets the learning rate
  #'
  #' @param learning_rate The new learning rate
  #'
  #' @export
  setLearningRate = function(learning_rate) {
    private$learning_rate <- learning_rate
  },

  #' @description
  #' Sets the regularization rate
  #'
  #' @param regularization_rate The new regularization rate
  #'
  #' @export
  setRegularizationRate = function(regularization_rate) {
    private$regularization_rate <- regularization_rate
  },

  #' @description
  #' Returns the learning rate
  #'
  #' @return The learning rate
  #'
  #' @export
  getLearningRate = function() private$learning_rate,

  #' @description
  #' Returns the regularization rate
  #'
  #' @return The regularization rate
  #'
  #' @export
  getRegularizationRate = function() private$regularization_rate,


  #' @description
  #' Performs the optimization algorithm
  #'
  #' @param neuralnet The neural network to be trained
  #' @param training_data The training data used for training the
  #' network
  #'
  optim = function(neuralnet, training_data) {
    layer2nvIndex <- function(layer) layer + 1

    L <- length(neuralnet$weights) - 1

    learning_rate <- private$learning_rate
    regularization_rate <- private$regularization_rate

    getLastXInfluence <-
      getLastXInfluenceL[[neuralnet$category]]
    getLastWeightsInfluence <-
      getLastWeightsInfluenceL[[neuralnet$category]]

    for(training_data in sample(training_data)) {
      netCalcResult <- neuralnet$calculate(training_data$input)

      expectedOutput <- training_data$expectedOutput
      rawNodeValue <- netCalcResult$rawNodeValues
      nodeValue <- netCalcResult$nodeValues
      output <- netCalcResult$output


      deltaList <- list()
      weightsInfluenceList <- list()

      if (L > 0) {
        lastXInfluence <-
          getLastXInfluence(expectedOutput,
                            nodeValue[[layer2nvIndex(L + 1)]],
                            neuralnet$weights[[L + 1]])

        deltaList[[L]] <-
          getLastDelta(lastXInfluence,
                       rawNodeValue[[layer2nvIndex(L)]],
                       neuralnet$dActfct)

        stopifnot(dim(deltaList[[L]]) == dim(neuralnet$bias[[L]]))
        if(!all(!is.nan(deltaList[[L]]))) {
          print(str_c("Delta: ", L))
          print(deltaList[[L]])
          stop()
        }
      }

      weightsInfluenceList[[L + 1]] <-
        getLastWeightsInfluence( expectedOutput,
                                 nodeValue[[layer2nvIndex(L + 1)]],
                                 nodeValue[[layer2nvIndex(L)]])
      stopifnot(dim(weightsInfluenceList[[L + 1]]) == dim(neuralnet$weights[[L + 1]]))

      if(!all(!is.nan(weightsInfluenceList[[L + 1]]))) {
        print(str_c("Weights: ", L + 1))
        print(weightsInfluenceList[[L + 1]])
        stop()
      }

      for(l in rev(seq_len(L))) {
        weightsInfluenceList[[l]] <-
          getPrevWeightsInfluence(deltaList[[l]],
                                          nodeValue[[layer2nvIndex(l - 1)]])
        stopifnot(dim(weightsInfluenceList[[l]]) == dim(neuralnet$weights[[l]]))
        if(!all(!is.nan(weightsInfluenceList[[l]]))) {
          print(str_c("Weights: ", l))
          stop()
        }
        if (l > 1) {
          deltaList[[l - 1]] <-
            getPrevDelta(deltaList[[l]],
                                 rawNodeValue[[layer2nvIndex(l - 1)]],
                                 neuralnet$weights[[l]], neuralnet$dActfct)
          stopifnot(dim(deltaList[[l - 1]]) == dim(neuralnet$bias[[l - 1]]))
          if(!all(!is.nan(weightsInfluenceList[[l - 1]]))) {
            print(str_c("Delta: ", l - 1))
            stop()
          }
        }
      }

      biasUpdates <- list()
      if (L > 0) {
        biasUpdates <-
          mapply(calculateBiasUpdate,
                 deltaList, learning_rate,
                 SIMPLIFY = F)
      }
      weightUpdates <-
        mapply(calculateWeightUpdate, neuralnet$weights,
               weightsInfluenceList, learning_rate, regularization_rate,
               SIMPLIFY = F)

      newBias <- mapply(`-`, neuralnet$bias, biasUpdates,
                        SIMPLIFY = F)
      newWeights <- mapply(`-`, neuralnet$weights, weightUpdates,
                        SIMPLIFY = F)

      neuralnet$bias <- newBias
      neuralnet$weights <- newWeights
    }
  },

  #' @description
  #' Deletes info from earlier optimization processes
  #'
  reset = function() { }
)
)
