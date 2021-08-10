#' Optimizer Stochastic Gradient Descent with Momentum
#'
#' @description
#' This class is responsible for performing the
#' optimization method "Stochastic Gradient Descent"
#' with the momentum optimization
#' on a neural network for a given list of training data
#'
OptimizerMomentum <- R6::R6Class("OptimizerMomentum",
private = list(
  learning_rate = 0,
  regularization_rate = 0,
  momentum_term = 0,

  weightMomentum = NULL,
  biasMomentum = NULL
),
public = list(
  #' @description
  #' Initializes a new OptimizerMomentum object
  #'
  #' @param learning_rate The learning rate used by the optimization method
  #' @param regularization_rate The regularization rate used by the optimization method
  #' @param momentum_term The momentum term used by the optimization method
  #'
  #' @export
  initialize = function(learning_rate, regularization_rate, momentum_term = 0.9) {
    private$learning_rate <- learning_rate
    private$regularization_rate <- regularization_rate
    private$momentum_term <- momentum_term

    self$reset()
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
  #' Sets the momentum term
  #'
  #' @param momentum_term The new momentum term
  #'
  #' @export
  setMomentumTerm = function(momentum_term) {
    private$momentum_term <- momentum_term
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
  #' Returns the momentum term
  #'
  #' @return The momentum term
  #'
  #' @export
  getMomentumTerm = function() private$momentum_term,

  #' @description
  #' Performs the optimization algorithm
  #'
  #' @param neuralnet The neural network to be trained
  #' @param training_data The training data used for training the
  #' network
  #'
  #' @export
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
      #print(training_data)
      netCalcResult <- neuralnet$calculate(training_data$input)

      expectedOutput <- training_data$expectedOutput
      rawNodeValue <- netCalcResult$rawNodeValues
      nodeValue <- netCalcResult$nodeValues
      output <- netCalcResult$output

      deltaList <- list()
      weightsInfluenceList <- list()

      lastXInfluence <-
        getLastXInfluence(expectedOutput,
                          nodeValue[[layer2nvIndex(L + 1)]],
                          neuralnet$weights[[L + 1]])

      deltaList[[L]] <-
        getLastDelta(lastXInfluence,
                     rawNodeValue[[layer2nvIndex(L)]],
                     neuralnet$dActfct)
      stopifnot(dim(deltaList[[L]]) == dim(neuralnet$bias[[L]]))

      weightsInfluenceList[[L + 1]] <-
        getLastWeightsInfluence( expectedOutput,
                                 nodeValue[[layer2nvIndex(L + 1)]],
                                 nodeValue[[layer2nvIndex(L)]])
      #print(nodeValue)
      #print(dim(weightsInfluenceList[[L + 1]]))
      #print(dim(neuralnet$weights[[L + 1]]))
      stopifnot(dim(weightsInfluenceList[[L + 1]]) == dim(neuralnet$weights[[L + 1]]))

      if(!all(!is.nan(weightsInfluenceList[[L + 1]]))) {
        print(str_c("Weights: ", L + 1))
        stop()
      }

      if(!all(!is.nan(deltaList[[L]]))) {
        print(str_c("Delta: ", L))
        stop()
      }

      for(l in rev(seq(L))) {
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

      biasUpdates <-
        mapply(calculateBiasUpdate, neuralnet$bias,
               deltaList, learning_rate,
               SIMPLIFY = F)
      weightUpdates <-
        mapply(calculateWeightUpdate, neuralnet$weights,
               weightsInfluenceList, learning_rate, regularization_rate,
               SIMPLIFY = F)

      # Add momentum
      if (!is.null(private$biasMomentum) && !is.null(private$weightMomentum)) {
        biasUpdates <- mapply(function(bu, bm) bu + bm * private$momentum_term,
                              biasUpdates, private$biasMomentum,
                              SIMPLIFY = F)
        weightUpdates <- mapply(function(wu, wm) wu + wm * private$momentum_term,
                                weightUpdates, private$weightMomentum,
                                SIMPLIFY = F)
      }

      private$biasMomentum <- biasUpdates
      private$weightMomentum <- weightUpdates

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
  #' @export
  reset = function() {
    private$weightMomentum <- NULL
    private$biasMomentum <- NULL
  }
)
)
