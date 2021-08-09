OptimiserSGD <- R6::R6Class("OptimiserSGD",
private = list(
  learning_rate = 0,
  lambda = 0,

  getLastXInfluenceL = list(
    regression = function(expectedOutput, netOutput, lastWeights) {
      -2 * (expectedOutput - netOutput) * t(lastWeights)
    },
    classification = function(expectedOutput, netOutput, lastWeights) {
      M <- function(x) exp(x) / sum(exp(x))
      #print(expectedOutput)
      #print(dim(lastWeights[expectedOutput,]))
      #print(dim(t(lastWeights)))
      #print(dim(M(netOutput)))
      #print("\n")
      -(lastWeights[expectedOutput,] - t(lastWeights) %*% M(netOutput))
    }
  ),

  getLastDelta = function(lastXInfluence, secLastRawNodeValues, dActfct) {
    #print(dActfct)
    lastXInfluence * dActfct(secLastRawNodeValues)
  },
  getPrevDelta = function(delta, prevRawNodeValues, weights, dActfct) {
    #print(t(weights))
    #print(delta)
    #print(dActfct)
    (t(weights) %*% delta) * dActfct(prevRawNodeValues)
  },

  getLastWeightsInfluenceL = list(
    regression = function(expectedOutput, netOutput, prevNodeValues) {
      #print(prevNodeValues)
      -2 * (expectedOutput - netOutput) * t(prevNodeValues)
    },
    classification = function(expectedOutput, netOutput, prevNodeValues) {
      #print(prevNodeValues)
      M <- function(x) exp(x) / sum(exp(x))

      'stopifnot(all(!is.nan(as.double(1:length(netOutput) == expectedOutput))))
      stopifnot(all(!is.nan(M(netOutput))))
      stopifnot(all(!is.nan(t(prevNodeValues))))
      stopifnot(all(!is.nan(as.double(1:length(netOutput) == expectedOutput) - M(netOutput))))
      #stopifnot(all(!is.nan(-(as.double(1:length(netOutput) == expectedOutput) - M(netOutput)) %*% t(prevNodeValues))))
      result <- -(as.double(1:length(netOutput) == expectedOutput) - M(netOutput)) %*% t(prevNodeValues)
      if (!all(!is.nan(result))) {
        print("------------------------------------")
        print(as.double(1:length(netOutput) == expectedOutput))
        print("------------------------------------")
        print(netOutput)
        print("------------------------------------")
        print(t(prevNodeValues))
        print("------------------------------------")
        print(result)
        print("------------------------------------")
        print("------------------------------------")

        stop()
      }'

      -(as.double(1:length(netOutput) == expectedOutput) - M(netOutput)) %*% t(prevNodeValues)
    }
  ),
  getPrevWeightsInfluence = function(delta, prevNodeValues) {
    delta %*% t(prevNodeValues)
  },

  calculateNewBias = function(oldBias, delta, N, learning_rate) {
    stopifnot(!all(is.nan(delta)))
    oldBias - learning_rate * delta / N
  },
  calculateNewWeights = function(oldWeights, weightsInfluence, N,
                                 learning_rate, lambda) {
    #print(weightsInfluence)
    #print(oldWeights)
    stopifnot(!all(is.nan(weightsInfluence)))
    change <- weightsInfluence / N + 2 * lambda * oldWeights
    oldWeights - learning_rate * change
  }

),
public = list(
  initialize = function(learning_rate, lambda) {
    private$learning_rate <- learning_rate
    private$lambda <- lambda
  },
  setLearningRate = function(learning_rate) {
    private$learning_rate <- learning_rate
  },
  setLambda = function(lambda) {
    private$lambda <- lambda
  },

  getLearningRate = function() private$learning_rate,
  getLambda = function() private$lambda,

  optim = function(neuralnet, training_data) {
    layer2nvIndex <- function(layer) layer + 1

    N <- length(training_data)
    L <- length(neuralnet$weights) - 1

    learning_rate <- private$learning_rate
    lambda <- private$lambda

    getLastXInfluence <-
      private$getLastXInfluenceL[[neuralnet$category]]
    getLastWeightsInfluence <-
      private$getLastWeightsInfluenceL[[neuralnet$category]]

    for(training_data in sample(training_data)) {
      #print(training_data)
      netCalcResult <- neuralnet$calculate(training_data$input)

      expectedOutput <- training_data$expectedOutput
      rawNodeValue <- netCalcResult$rawNodeValues
      nodeValue <- netCalcResult$nodeValues
      output <- netCalcResult$output

      deltaList <- lapply(seq(L), function(x) NULL)
      weightsInfluenceList <- lapply(seq(L + 1), function(x) NULL)

      lastXInfluence <-
        getLastXInfluence(expectedOutput,
                          nodeValue[[layer2nvIndex(L + 1)]],
                          neuralnet$weights[[L + 1]])

      deltaList[[L]] <-
        private$getLastDelta(lastXInfluence,
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
          private$getPrevWeightsInfluence(deltaList[[l]],
                                          nodeValue[[layer2nvIndex(l - 1)]])
        stopifnot(dim(weightsInfluenceList[[l]]) == dim(neuralnet$weights[[l]]))
        if(!all(!is.nan(weightsInfluenceList[[l]]))) {
          print(str_c("Weights: ", l))
          stop()
        }
        if (l > 1) {
          deltaList[[l - 1]] <-
            private$getPrevDelta(deltaList[[l]],
                                 rawNodeValue[[layer2nvIndex(l - 1)]],
                                 neuralnet$weights[[l]], neuralnet$dActfct)
          stopifnot(dim(deltaList[[l - 1]]) == dim(neuralnet$bias[[l - 1]]))
          if(!all(!is.nan(weightsInfluenceList[[l - 1]]))) {
            print(str_c("Delta: ", l - 1))
            stop()
          }
        }
      }

      newBias <-
        mapply(private$calculateNewBias, neuralnet$bias,
               deltaList, N, learning_rate,
               SIMPLIFY = F)
      newWeights <-
        mapply(private$calculateNewWeights, neuralnet$weights,
               weightsInfluenceList, N, learning_rate, lambda,
               SIMPLIFY = F)

      neuralnet$bias <- newBias
      neuralnet$weights <- newWeights
    }
  }
)
)
