#' Generate an accuracy tester function for classification
#'
#' @description
#' \code{accuracy_tester_classification} returns a function that
#' compares the output of a neural network with the expected output
#' and returns true if they match and false if they don't
#'
#' @return accuracy tester function for classification
#'
#' @export
accuracy_tester_classification <- function() {
  function(netOutput, expectedOutput) {
    netOutput == expectedOutput
  }
}

#' Generate a relative tester function for regression
#'
#' @description
#' \code{accuracy_tester_regression_rel} returns a function that
#' compares the output of a neural network with the expected output
#' and returns true if the relative difference of the neural network
#' output to the expected output does not exceed \code{maxRelativeDiviation}.
#' Otherwise the returned function will return false
#'
#' @param maxRelativeDiviation the maximal allowed relative diviation from
#' the expected output
#'
#' @return relative accuracy tester function for regression
#'
#' @export
accuracy_tester_regression_rel <- function(maxRelativeDiviation) {
  function(netOutput, expectedOutput) {
    maxDelta <- abs(maxRelativeDiviation * expectedOutput)
    absDelta <- abs(expectedOutput - netOutput)
    all(absDelta <= maxDelta)
  }
}

#' Generate an absolute tester function for regression
#'
#' @description
#' \code{accuracy_tester_regression_abs} returns a function that
#' compares the output of a neural network with the expected output
#' and returns true if the absolute difference of the neural network
#' output to the expected output does not exceed \code{maxAbsoluteDiviation}.
#' Otherwise the returned function will return false
#'
#' @param maxAbsoluteDiviation the maximal allowed absolute diviation from
#' the expected output
#'
#' @return absolute accuracy tester function for regression
#'
#' @export
accuracy_tester_regression_abs <- function(maxAbsoluteDiviation) {
  function(netOutput, expectedOutput) {
    absDelta <- abs(expectedOutput - netOutput)
    all(absDelta <= maxAbsoluteDiviation)
  }
}

