#' getLastXInfluenceL
#'
#' @description
#' List of functions (one for classification and
#' one for regression) that calculate the derivatives
#' of the loss function of the node-values
#' from the last layer (excluding the output layer)
#'
#' @param expectedOutput The output that the network is supposed to
#' produce
#' @param netOutput The output the that network produced
#' @param lastWeights The weights matrix of the last layer
#'
#' @noRd
getLastXInfluenceL = list(
  regression = function(expectedOutput, netOutput, lastWeights) {
    -2 * (expectedOutput - netOutput) * t(lastWeights)
  },
  classification = function(expectedOutput, netOutput, lastWeights) {
    M <- function(x) exp(x) / sum(exp(x))
    -(lastWeights[expectedOutput,] - t(lastWeights) %*% M(netOutput))
  }
)

#' getLastDelta
#'
#' @description
#' \code{getLastDelta} calculates influence of the last
#' layer (excluding the output layer) on the loss function
#'
#' @param lastXInfluence The derivatives
#' of the loss function of the node-values
#' from the last layer
#' @param secLastRawNodeValues The raw node values
#' (= node values before applying the activation function) of the
#' last layer
#' @param dActfct The derivative of the activation function
#'
#' @noRd
getLastDelta = function(lastXInfluence, secLastRawNodeValues, dActfct) {
  lastXInfluence * dActfct(secLastRawNodeValues)
}

#' getPrevDelta
#'
#' @description
#' \code{getPrevDelta} calculates influence of a layer
#' on the loss function out of delta of the following layer.
#'
#' @param delta influence of the following layer on the loss function
#' @param prevRawNodeValues The raw node values
#' (= node values before applying the activation function) of the
#' layer who's delta is to be calculated
#' @param dActfct The derivative of the activation function
#'
#' @noRd
getPrevDelta = function(delta, prevRawNodeValues, weights, dActfct) {
  (t(weights) %*% delta) * dActfct(prevRawNodeValues)
}

#' getLastWeightsInfluenceL
#'
#' @description
#' List of functions (one for classification and
#' one for regression) that calculate influence of the
#' weights of the last layer (excluding the output layer)
#' on the loss function
#'
#' @param expectedOutput The output that the network is supposed to
#' produce
#' @param netOutput The output the that network produced
#' @param prevNodeValues The node-values of the last layer
#' (excluding the output layer)
#'
#' @noRd
getLastWeightsInfluenceL = list(
  regression = function(expectedOutput, netOutput, prevNodeValues) {
    -2 * (expectedOutput - netOutput) * t(prevNodeValues)
  },
  classification = function(expectedOutput, netOutput, prevNodeValues) {
    M <- function(x) exp(x) / sum(exp(x))
    -(as.double(1:length(netOutput) == expectedOutput) - M(netOutput)) %*% t(prevNodeValues)
  }
)

#' getPrevWeightsInfluence
#'
#' @description
#' \code{getPrevWeightsInfluence} calculates influence of the
#' weights of a layer on the loss function out of delta
#' of the following layer.
#'
#' @param delta influence of the following layer on the loss function
#' @param prevRawNodeValues The raw node values
#' (= node values before applying the activation function) of the
#' layer who's delta is to be calculated
#' @param dActfct The derivative of the activation function
#'
#' @noRd
getPrevWeightsInfluence = function(delta, prevNodeValues) {
  delta %*% t(prevNodeValues)
}

#' calculateBiasUpdate
#'
#' @description
#' \code{calculateBiasUpdate} calculates the update vector
#' for the bias for a layer
#'
#' @param delta influence of the layer on the loss function
#' @param learning_rate The learning rate
#'
#' @noRd
calculateBiasUpdate = function(delta, learning_rate) {
  stopifnot(!all(is.nan(delta)))
  learning_rate * delta
}

#' calculateWeightUpdate
#'
#' @description
#' \code{calculateWeightUpdate} calculates the update matrix
#' for the weights for a layer
#'
#' @param weightsInfluence influence of the
#' weights of the layer on the loss function
#' @param learning_rate The learning rate
#' @param lambda the regularization rate
#'
#' @noRd
calculateWeightUpdate = function(oldWeights, weightsInfluence,
                                 learning_rate, lambda) {
  stopifnot(!all(is.nan(weightsInfluence)))
  change <- weightsInfluence + 2 * lambda * oldWeights
  learning_rate * change
}
