Trainer <- R6::R6Class("Trainer",
private = list(
  training_data = list(),
  test_data = list(),
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

  calculateNewBias = function(oldBias, delta, N) {
    stopifnot(!all(is.nan(delta)))
    oldBias - private$learning_rate * delta / N
  },
  calculateNewWeights = function(oldWeights, weightsInfluence, N) {
    #print(weightsInfluence)
    #print(oldWeights)
    stopifnot(!all(is.nan(weightsInfluence)))
    change <- weightsInfluence / N + 2 * private$lambda * oldWeights
    oldWeights - private$learning_rate * change
  },

  layer2nvIndex = function(layer) layer + 1

),
public = list(
  initialize = function(training_data, test_data, learning_rate, lambda) {
    private$training_data <- training_data
    private$test_data <- test_data
    private$learning_rate <- learning_rate
    private$lambda <- lambda
  },

  setTrainingData = function(training_data) {
    private$training_data <- training_data
  },
  setTestData = function(test_data) {
    private$test_data <- test_data
  },
  setLearningRate = function(learning_rate) {
    private$learning_rate <- learning_rate
  },
  setLambda = function(lambda) {
    private$lambda <- lambda
  },

  getTrainingData = function() private$training_data,
  getTestData = function() private$test_data,
  getLearningRate = function() private$learning_rate,
  getLambda = function() private$lambda,

  test = function(neuralnet) {
    accuracyFunc <- NULL
    if(neuralnet$category == "classification") {
      accuracyFunc <- function(netOutput, expectedOutput)
        as.double(netOutput == expectedOutput)
    } else {
      accuracyFunc <- function(netOutput, expectedOutput)
        min((netOutput - expectedOutput)**2, 1)
    }
    accuracyVals <- sapply(private$test_data, function(td) {
      #print(td)
      netResult <- neuralnet$calculate(td$input)
      netOutput <- netResult$output
      #print(netResult)
      #print(netOutput)
      #print(td$expectedOutput)
      accuracyFunc(netOutput, td$expectedOutput)
    })
    sum(accuracyVals) / length(private$test_data)
  },

  train = function(neuralnet) {
    N <- length(private$training_data)
    L <- length(neuralnet$weights) - 1

    getLastXInfluence <-
      private$getLastXInfluenceL[[neuralnet$category]]
    getLastWeightsInfluence <-
      private$getLastWeightsInfluenceL[[neuralnet$category]]

    for(training_data in sample(private$training_data)) {
      #print(training_data)
      netCalcResult <- neuralnet$calculate(training_data$input)

      expectedOutput <- training_data$expectedOutput
      rawNodeValue <- netCalcResult$rawNodeValues
      nodeValue <- netCalcResult$nodeValues
      output <- netCalcResult$output

      deltaList <- lapply(seq(L), \(x) NULL)
      weightsInfluenceList <- lapply(seq(L + 1), \(x) NULL)

      lastXInfluence <-
        getLastXInfluence(expectedOutput,
                          nodeValue[[private$layer2nvIndex(L + 1)]],
                          neuralnet$weights[[L + 1]])

      deltaList[[L]] <-
        private$getLastDelta(lastXInfluence,
                          rawNodeValue[[private$layer2nvIndex(L)]],
                          neuralnet$dActfct)
      stopifnot(dim(deltaList[[L]]) == dim(neuralnet$bias[[L]]))

      weightsInfluenceList[[L + 1]] <-
        getLastWeightsInfluence( expectedOutput,
                                 nodeValue[[private$layer2nvIndex(L + 1)]],
                                 nodeValue[[private$layer2nvIndex(L)]])
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
                                  nodeValue[[private$layer2nvIndex(l - 1)]])
        stopifnot(dim(weightsInfluenceList[[l]]) == dim(neuralnet$weights[[l]]))
        if(!all(!is.nan(weightsInfluenceList[[l]]))) {
          print(str_c("Weights: ", l))
          stop()
        }
        if (l > 1) {
          deltaList[[l - 1]] <-
            private$getPrevDelta(deltaList[[l]],
                         rawNodeValue[[private$layer2nvIndex(l - 1)]],
                         neuralnet$weights[[l]], neuralnet$dActfct)
          stopifnot(dim(deltaList[[l - 1]]) == dim(neuralnet$bias[[l - 1]]))
          if(!all(!is.nan(weightsInfluenceList[[l - 1]]))) {
            print(str_c("Delta: ", l - 1))
            stop()
          }
        }
      }

      newBias <-
        mapply(private$calculateNewBias, neuralnet$bias, deltaList, N,
               SIMPLIFY = F)
      newWeights <-
        mapply(private$calculateNewWeights, neuralnet$weights,
               weightsInfluenceList, N, SIMPLIFY = F)

      neuralnet$bias <- newBias
      neuralnet$weights <- newWeights
    }
  }
))
