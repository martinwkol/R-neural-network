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
      -(lastWeights[expectedOutput,] - t(lastWeights) %*% M(netOutput))
    }
  ),

  getLastDelta = function(lastXInfluence, secLastRawNodeValues, dActfct) {
    lastXInfluence * dActfct(secLastRawNodeValues)
  },
  getPrevDelta = function(delta, prevRawNodeValues, weights, dActfct) {
    #print(t(weights))
    #print(delta)
    (t(weights) %*% delta) * dActfct(prevRawNodeValues)
  },

  getLastWeightsInfluenceL = list(
    regression = function(expectedOutput, netOutput, prevNodeValues) {
      #print(prevNodeValues)
      -2 * (expectedOutput - netOutput) * t(prevNodeValues)
    }
    classification = function(expectedOutput, netOutput, prevNodeValues) {
      #print(prevNodeValues)
      -(as.double(1:length(netOutput) == expectedOutput) - M(netOutput)) %*% t(prevNodeValues)
    }
  ),
  getPrevWeightsInfluence = function(delta, prevNodeValues) {
    delta %*% t(prevNodeValues)
  },

  calculateNewBias = function(oldBias, delta, N) {
    oldBias - self$learning_rate * delta / N
  },
  calculateNewWeights = function(oldWeights, weightsInfluence, N) {
    #print(weightsInfluence)
    #print(oldWeights)
    change <- weightsInfluence / N + 2 * self$lambda * oldWeights
    oldWeights - self$learning_rate * change
  },

  layer2nvIndex = function(layer) layer + 1

),
public = list(
  initialize = function(training_data, learning_rate, lambda) {
    self$training_data <- training_data
    self$learning_rate <- learning_rate
    self$lambda <- lambda
  },

  setTrainingData = function(training_data) {
    self$training_data <- training_data
  },
  setLearningRate = function(learning_rate) {
    self$learning_rate <- learning_rate
  },
  setLambda = function(lambda) {
    self$lambda <- lambda
  },

  getTrainingData <- function() self$training_data
  getLearningRate <- function() self$learning_rate
  getLambda <- function() self$lambda


  train = function(neuralnet) {
    training_data_list <- lapply(self$training_data, function(td) {
        c(neuralnet$calculate(td$input),
          list(expectedOutput=td$expectedOutput))
    })
    N <- length(training_data_list)
    L <- length(neuralnet$weights) - 1

    getLastXInfluence <-
      self$getLastXInfluenceL[[neuralnet$category]]
    getLastWeightsInfluence <-
      self$getLastWeightsInfluenceL[[neuralnet$category]]

    for(training_data in training_data_list) {
      #print(training_data)

      expectedOutput <- training_data$expectedOutput
      rawNodeValue <- training_data$rawNodeValues
      nodeValue <- training_data$nodeValues

      deltaList <- lapply(seq(L), \(x) NULL)
      weightsInfluenceList <- lapply(seq(L + 1), \(x) NULL)

      lastXInfluence <-
        getLastXInfluence(expectedOutput,
                          nodeValue[[layer2nvIndex(L + 1)]],
                          neuralnet$weights[[L + 1]])

      deltaList[[L]] <-
        self$getLastDelta(lastXInfluence,
                          rawNodeValue[[L]],
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


      for(l in rev(seq(L))) {
        weightsInfluenceList[[l]] <-
          self$getPrevWeightsInfluence(deltaList[[l]],
                                  nodeValue[[layer2nvIndex(l - 1)]])
        stopifnot(dim(weightsInfluenceList[[l]]) == dim(neuralnet$weights[[l]]))
        if (l > 1) {
          deltaList[[l - 1]] <-
            srlf$getPrevDelta(deltaList[[l]],
                         rawNodeValue[[layer2nvIndex(l - 1)]],
                         neuralnet$weights[[l]], neuralnet$dActfct)
          stopifnot(dim(deltaList[[l - 1]]) == dim(neuralnet$bias[[l - 1]]))
        }
      }

      newBias <-
        mapply(calculateNewBias, neuralnet$bias, deltaList, N,
               SIMPLIFY = F)
      newWeights <-
        mapply(calculateNewWeights, neuralnet$weights,
               weightsInfluenceList, N, SIMPLIFY = F)

      neuralnet$bias <- newBias
      neuralnet$weights <- newWeights
    }
  }
))
