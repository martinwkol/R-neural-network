accuracy_tester_classification <- function() {
  function(netOutput, expectedOutput) {
    as.integer(netOutput == expectedOutput)
  }
}

accuracy_tester_regression_rel <- function(maxDiviation) {
  function(netOutput, expectedOutput) {
    maxDelta <- abs(maxDiviation * expectedOutput)
    absDelta <- abs(expectedOutput - netOutput)
    as.integer( all(absDelta <= maxDelta) )
  }
}

accuracy_tester_regression_abs <- function(maxDiviation) {
  function(netOutput, expectedOutput) {
    absDelta <- abs(expectedOutput - netOutput)
    as.integer( all(absDelta <= maxDiviation) )
  }
}

