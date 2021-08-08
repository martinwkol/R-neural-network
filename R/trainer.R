Trainer <- R6::R6Class("Trainer",
private = list(
  training_data = list(),
  learning_rate = 0,
  lambda = 0,

  getLastXInfluenceL = list(
    regression = function(expectedOutput, netOutput, lastWeights) {
      -2 * (expectedOutput - netOutput) * t(lastWeights)
    },
    classification = function(expectedOutput, netOutput, lastWeights) {
      M <- function(x) exp(x) / sum(x)
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
      M <- function(x) exp(x) / sum(x)
      -(as.double(1:length(netOutput) == expectedOutput) - M(netOutput)) %*% t(prevNodeValues)
    }
  ),
  getPrevWeightsInfluence = function(delta, prevNodeValues) {
    delta %*% t(prevNodeValues)
  },

  calculateNewBias = function(oldBias, delta, N) {
    oldBias - private$learning_rate * delta / N
  },
  calculateNewWeights = function(oldWeights, weightsInfluence, N) {
    #print(weightsInfluence)
    #print(oldWeights)
    change <- weightsInfluence / N + 2 * private$lambda * oldWeights
    oldWeights - private$learning_rate * change
  },

  layer2nvIndex = function(layer) layer + 1

),
public = list(
  initialize = function(training_data, learning_rate, lambda) {
    private$training_data <- training_data
    private$learning_rate <- learning_rate
    private$lambda <- lambda
  },

  setTrainingData = function(training_data) {
    private$training_data <- training_data
  },
  setLearningRate = function(learning_rate) {
    private$learning_rate <- learning_rate
  },
  setLambda = function(lambda) {
    private$lambda <- lambda
  },

  getTrainingData = function() private$training_data,
  getLearningRate = function() private$learning_rate,
  getLambda = function() private$lambda,


  train = function(neuralnet) {
    training_data_list <- lapply(private$training_data, function(td) {
        c(neuralnet$calculate(td$input),
          list(expectedOutput=td$expectedOutput))
    })
    N <- length(training_data_list)
    L <- length(neuralnet$weights) - 1

    getLastXInfluence <-
      private$getLastXInfluenceL[[neuralnet$category]]
    getLastWeightsInfluence <-
      private$getLastWeightsInfluenceL[[neuralnet$category]]

    for(training_data in training_data_list) {
      #print(training_data)

      expectedOutput <- training_data$expectedOutput
      rawNodeValue <- training_data$rawNodeValues
      nodeValue <- training_data$nodeValues
      output <- training_data$output

      deltaList <- lapply(seq(L), \(x) NULL)
      weightsInfluenceList <- lapply(seq(L + 1), \(x) NULL)

      lastXInfluence <-
        getLastXInfluence(expectedOutput,
                          nodeValue[[private$layer2nvIndex(L + 1)]],
                          neuralnet$weights[[L + 1]])

      deltaList[[L]] <-
        private$getLastDelta(lastXInfluence,
                          rawNodeValue[[L]],
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


      for(l in rev(seq(L))) {
        weightsInfluenceList[[l]] <-
          private$getPrevWeightsInfluence(deltaList[[l]],
                                  nodeValue[[private$layer2nvIndex(l - 1)]])
        stopifnot(dim(weightsInfluenceList[[l]]) == dim(neuralnet$weights[[l]]))
        if (l > 1) {
          deltaList[[l - 1]] <-
            private$getPrevDelta(deltaList[[l]],
                         rawNodeValue[[private$layer2nvIndex(l - 1)]],
                         neuralnet$weights[[l]], neuralnet$dActfct)
          stopifnot(dim(deltaList[[l - 1]]) == dim(neuralnet$bias[[l - 1]]))
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
