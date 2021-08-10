lossFunctionList = list(
  regression = function(expectedOutput, netOutput) {
    (expectedOutput - netOutput) ** 2
  },
  classification = function(expectedOutput, netOutput, lastWeights) {
    M <- function(x) exp(x) / sum(exp(x))
    -log(M(netOutput)[expectedOutput])
  }
)

getLastXInfluenceL = list(
  regression = function(expectedOutput, netOutput, lastWeights) {
    -2 * (expectedOutput - netOutput) * t(lastWeights)
  },
  classification = function(expectedOutput, netOutput, lastWeights) {
    M <- function(x) exp(x) / sum(exp(x))
    -(lastWeights[expectedOutput,] - t(lastWeights) %*% M(netOutput))
  }
)

getLastDelta = function(lastXInfluence, secLastRawNodeValues, dActfct) {
  lastXInfluence * dActfct(secLastRawNodeValues)
}

getPrevDelta = function(delta, prevRawNodeValues, weights, dActfct) {
  (t(weights) %*% delta) * dActfct(prevRawNodeValues)
}

getLastWeightsInfluenceL = list(
  regression = function(expectedOutput, netOutput, prevNodeValues) {
    -2 * (expectedOutput - netOutput) * t(prevNodeValues)
  },
  classification = function(expectedOutput, netOutput, prevNodeValues) {
    M <- function(x) exp(x) / sum(exp(x))
    -(as.double(1:length(netOutput) == expectedOutput) - M(netOutput)) %*% t(prevNodeValues)
  }
)

getPrevWeightsInfluence = function(delta, prevNodeValues) {
  delta %*% t(prevNodeValues)
}

calculateBiasUpdate = function(oldBias, delta, learning_rate) {
  stopifnot(!all(is.nan(delta)))
  learning_rate * delta
}

calculateWeightUpdate = function(oldWeights, weightsInfluence,
                                 learning_rate, lambda) {
  stopifnot(!all(is.nan(weightsInfluence)))
  change <- weightsInfluence + 2 * lambda * oldWeights
  learning_rate * change
}
