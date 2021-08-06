
#' sgdAlg - Stochastic Gradient Descent
#'
#' sgdAlg implements an Algorithm for training a \code{?NeuralNet} Neural Network
#' using Stochastic Gradient Descent.
#'
#' @param neuralnet A R6 Neural Network that will be trained with the given data
#' @param training_data a data set used to train the Neural Network
#' @param learing_rate the learning rate to be used by the Algorithm
#' @param lamda a lambda to be used by the Algorithm
#' @seealso ?NeuralNet
#' @export
sgdAlg <- function(neuralnet, training_data, learning_rate, lambda) {
  M <- function(x) exp(x) / sum(x)
  if (neuralnet$category == "regression") {
    getLastXInfluence <- function(expectedOutput, netOutput, lastWeights) {
      -2 * (expectedOutput - netOutput) * t(lastWeights)
    }
  } else if (neuralnet$category == "classification") {
    getLastXInfluence <- function(expectedOutput, netOutput, lastWeights) {
      -(lastWeights[expectedOutput,] - t(lastWeights) %*% M(netOutput))
    }
  }

  getLastDelta <- function(lastXInfluence, secLastRawNodeValues, dActfct) {
    lastXInfluence * dActfct(secLastRawNodeValues)
  }

  getPrevDelta <- function(delta, prevRawNodeValues, weights, dActfct) {
    #print(t(weights))
    #print(delta)
    (t(weights) %*% delta) * dActfct(prevRawNodeValues)
  }

  if (neuralnet$category == "regression") {
    getLastWeightsInfluence <- function(expectedOutput, netOutput, prevNodeValues) {
      #print(prevNodeValues)
      -2 * (expectedOutput - netOutput) * t(prevNodeValues)
    }
  } else if (neuralnet$category == "classification") {
    getLastWeightsInfluence <- function(expectedOutput, netOutput, prevNodeValues) {
      #print(prevNodeValues)
      -(as.double(1:length(netOutput) == expectedOutput) - M(netOutput)) %*% t(prevNodeValues)
    }
  }

  getPrevWeightsInfluence <- function(delta, prevNodeValues) {
    delta %*% t(prevNodeValues)
  }

  calculateNewBias <- function(oldBias, delta, learning_rate, N) {
    oldBias - learning_rate * delta / N
  }

  calculateNewWeights <- function(oldWeights, weightsInfluence, learning_rate, lambda, N) {
    print(weightsInfluence)
    print(oldWeights)
    change <- weightsInfluence / N + 2 * lambda * oldWeights
    oldWeights - learning_rate * change
  }

  layer2nvIndex <- function(layer) layer + 1

  backprop <- function(training_data_list) {
    N <- length(training_data_list)
    L <- length(neuralnet$weights) - 1

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
        getLastDelta( lastXInfluence,
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
          getPrevWeightsInfluence(deltaList[[l]],
                                  nodeValue[[layer2nvIndex(l - 1)]])
        stopifnot(dim(weightsInfluenceList[[l]]) == dim(neuralnet$weights[[l]]))
        if (l > 1) {
          deltaList[[l - 1]] <-
            getPrevDelta(deltaList[[l]],
                         rawNodeValue[[layer2nvIndex(l - 1)]],
                         neuralnet$weights[[l]], neuralnet$dActfct)
          stopifnot(dim(deltaList[[l - 1]]) == dim(neuralnet$bias[[l - 1]]))
        }
      }

      newBias <-
        mapply(calculateNewBias, neuralnet$bias, deltaList,
               learning_rate, N,
               SIMPLIFY = F)
      newWeights <-
        mapply(calculateNewWeights, neuralnet$weights,
               weightsInfluenceList, learning_rate,
               lambda, N, SIMPLIFY = F)

      neuralnet$bias <- newBias
      neuralnet$weights <- newWeights
    }
  }

  training_data_list <- lapply(training_data,
                               \(td) c(neuralnet$calculate(td$input),
                                       list(expectedOutput=td$expectedOutput)))
  #print(training_data_list)
  backprop(training_data_list)
}
