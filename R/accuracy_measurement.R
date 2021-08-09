measurement_classification <- function() {
  function(netOutput, expectedOutput) {
    as.integer(netOutput == expectedOutput)
  }
}

measurement_regression_rel <- function(maxDiviation) {
  function(netOutput, expectedOutput) {
    maxDelta <- abs(maxDiviation * expectedOutput)
    absDelta <- abs(expectedOutput - netOutput)
    as.integer( all(absDelta <= maxDelta) )
  }
}

measurement_regression_abs <- function(maxDiviation) {
  function(netOutput, expectedOutput) {
    absDelta <- abs(expectedOutput - netOutput)
    as.integer( all(absDelta <= maxDiviation) )
  }
}

