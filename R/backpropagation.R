sgdAlg <- function(neuralnet, inputs, expectedOutputs, learning_rate, lambda) {
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

  calculateNewBiasSGD <- function(oldBias, delta, learning_rate, N) {
    oldBias - learning_rate * delta / N
  }

  calculateNewWeightsSGD <- function(oldWeights, weightsInfluence, learning_rate, lambda, N) {
    change <- weightsInfluence / N + 2 * lambda * oldWeights
    oldWeights - learning_rate * change
  }

  feed_foreward <- function(input, expectedOutput) {
    stopifnot("input size doesn't fit inputlayer size" = length(input) == neuralnet$inputsize)

    rawNodeValues <- list(input)
    nodeValues <- list(input)
    output <- input
    for(i in 1:neuralnet$nrhiddenlayers) {
      #weights
      output <- neuralnet$weights[[i]]%*%output
      #bias
      output <- output + neuralnet$bias[[i]]
      rawNodeValues <- append(rawNodeValues, output)
      #apply the activation function
      output <- sapply(output, neuralnet$actfct)
      nodeValues <- append(nodeValues, output)
    }

    output <- neuralnet$weights[[neuralnet$nrhiddenlayers + 1]] %*% output
    rawNodeValues <- append(rawNodeValues, output)
    output <- sapply(output, neuralnet$outputfct)
    nodeValues <- append(nodeValues, output)

    list(expectedOutput = expectedOutput,
         rawNodeValues = rawNodeValues,
         nodeValues = nodeValues)
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

  training_data_list <- mapply(feed_foreward, inputs, expectedOutputs)
  backprop(training_data_list)
}
