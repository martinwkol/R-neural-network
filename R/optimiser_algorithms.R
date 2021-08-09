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
)

getLastDelta = function(lastXInfluence, secLastRawNodeValues, dActfct) {
  #print(dActfct)
  lastXInfluence * dActfct(secLastRawNodeValues)
}

getPrevDelta = function(delta, prevRawNodeValues, weights, dActfct) {
  #print(t(weights))
  #print(delta)
  #print(dActfct)
  (t(weights) %*% delta) * dActfct(prevRawNodeValues)
}

getLastWeightsInfluenceL = list(
  regression = function(expectedOutput, netOutput, prevNodeValues) {
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
)

getPrevWeightsInfluence = function(delta, prevNodeValues) {
  delta %*% t(prevNodeValues)
}

calculateBiasUpdate = function(oldBias, delta, N, learning_rate) {
  stopifnot(!all(is.nan(delta)))
  learning_rate * delta / N
}

calculateWeightUpdate = function(oldWeights, weightsInfluence, N,
                                 learning_rate, lambda) {
  #print(weightsInfluence)
  #print(oldWeights)
  stopifnot(!all(is.nan(weightsInfluence)))
  change <- weightsInfluence / N + 2 * lambda * oldWeights
  learning_rate * change
}
