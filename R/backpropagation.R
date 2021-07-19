
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
    (t(weights) %*% delta) * dActfct(prevRawNodeValues)
  }

  if (neuralnet$category == "regression") {
    getLastWeightsInfluence <- function(expectedOutput, netOutput, prevNodeValues) {
      -2 * (expectedOutput - netOutput) * t(prevNodeValues)
    }
  } else if (neuralnet$category == "classification") {
    M <- function(x) e^x / sum(x)
    getLastWeightsInfluence <- function(expectedOutput, netOutput, prevNodeValues) {
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
    change <- weightsInfluence / N + 2 * lambda * oldWeights
    oldWeights - learning_rate * change
  }

  backprop <- function(training_data_list) {
    N <- length(training_data_list)
    aoLayers <- length(neuralnet$weights) + 1

    for(training_data in training_data_list) {
      expectedOutput <- training_data$expectedOutputs
      rawNodeValue <- training_data$rawNodeValues
      nodeValue <- training_data$nodeValues

      deltaList <- lapply(seq(aoLayers - 1), \(x) NULL)
      weightsInfluenceList <- lapply(seq(aoLayers), \(x) NULL)

      lastXInfluence <-
        getLastXInfluence(expectedOutput,
                          nodeValue[[aoLayers]],
                          neuralnet$weights[[aoLayers - 1]])

      deltaList[[aoLayers - 1]] <-
        getLastDelta( lastXInfluence,
                      rawNodeValue[[aoLayers - 1]],
                      neuralnet$dActfct)

      weightsInfluenceList[[aoLayers]] <-
        getLastWeightsInfluence( expectedOutput,
                                 nodeValue[[aoLayers]],
                                 nodeValue[[aoLayers - 1]])

      for(l in rev(seq(aoLayers - 1)[-1])) {
        weightsInfluenceList[[l]] <-
          getPrevWeightsInfluence(deltaList[[l]],
                                  nodeValue[[l - 1]])
        if (l > 2) {
          deltaList[[l - 1]] <-
            getPrevDelta(deltaList[[l]],
                         rawNodeValue[[l - 1]],
                         neuralnet$weights[[l]], neuralnet$dActfct)
        }
      }

      newBias <-
        mapply(calculateNewBias, neuralnet$bias, deltaList[-1],
               learning_rate, N,
               SIMPLIFY = F)
      newWeights <-
        mapply(calculateNewWeights, neuralnet$weights,
               weightsInfluenceList[-1], learning_rate,
               lambda, N, SIMPLIFY = F)

      neuralnet$bias <- newBias
      neuralnet$weights <- newWeights
    }
  }

  training_data_list <- lapply(training_data,
                               \(td) c(neuralnet$calculate(td$input),
                                       list(expectedOutput=td$expectedOutput)))
  backprop(training_data_list)
}
