OptimiserMomentum <- R6::R6Class("OptimiserMomentum",
private = list(
  learning_rate = 0,
  lambda = 0,
  momentum_term = 0,

  weightMomentum = NULL,
  biasMomentum = NULL
),
public = list(
  initialize = function(learning_rate, lambda, momentum_term = 0.9) {
    private$learning_rate <- learning_rate
    private$lambda <- lambda
    private$momentum_term <- momentum_term

    self$reset()
  },
  setLearningRate = function(learning_rate) {
    private$learning_rate <- learning_rate
  },
  setLambda = function(lambda) {
    private$lambda <- lambda
  },
  setMomentumTerm = function(momentum_term) {
    private$momentum_term <- momentum_term
  },

  getLearningRate = function() private$learning_rate,
  getLambda = function() private$lambda,
  getMomentumTerm = function() private$momentum_term,

  optim = function(neuralnet, training_data, N = 0) {
    layer2nvIndex <- function(layer) layer + 1

    N <- max(N, length(training_data))
    L <- length(neuralnet$weights) - 1

    learning_rate <- private$learning_rate
    lambda <- private$lambda

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
               deltaList, N, learning_rate,
               SIMPLIFY = F)
      weightUpdates <-
        mapply(calculateWeightUpdate, neuralnet$weights,
               weightsInfluenceList, N, learning_rate, lambda,
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

  reset = function() {
    private$weightMomentum <- NULL
    private$biasMomentum <- NULL
  }
)
)
