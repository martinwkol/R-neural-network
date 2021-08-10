

test_that("getLastXInfluence regression", {
  getLastXInfluence <- getLastXInfluenceL[["regression"]]

  set.seed(10)
  layersizes <- c(1, round(runif(10, min = 1, max = 100)))
  for (layersize in layersizes) {
    input <- round(runif(1, min = 1, max = 100))
    expectedOutput <- round(runif(1, min = 1, max = 100))
    nn <- NeuralNet$new(c(1, layersize, 1), category="regression")

    netOutput <- nn$calculate(input)$output
    lastWeights <- nn$weights[[length(nn$weights)]]
    lastXInfluence <- getLastXInfluence(expectedOutput,
                                        netOutput, lastWeights)

    expect_equal(length(lastXInfluence), dim(lastWeights)[2])
    for (i in seq(length(lastXInfluence))) {
      # test with element wise formula
      expect_equal(lastXInfluence[i],
                   -2 * (expectedOutput - netOutput) * lastWeights[1, i])
    }
  }
})

test_that("getLastXInfluence classification", {
  getLastXInfluence <- getLastXInfluenceL[["classification"]]

  set.seed(10)
  iterations <- 10
  layersizes_hidden <- c(1, round(runif(iterations, min = 1, max = 100)))
  layersizes_output <- c(1, round(runif(iterations, min = 1, max = 100)))
  for (i in seq(iterations + 1)) {
    layersize_hidden <- layersizes_hidden[i]
    layersize_output <- layersizes_output[i]

    input <- round(runif(1, min = 1, max = 100))
    expectedOutput <- round(runif(1, min = 1, max = layersize_output))
    nn <- NeuralNet$new(c(1, layersize_hidden, layersize_output),
                        category="classification")

    netCRes <- nn$calculate(input)
    netOutput <- netCRes$nodeValues[[length(netCRes$nodeValues)]]
    lastWeights <- nn$weights[[length(nn$weights)]]
    lastXInfluence <- getLastXInfluence(expectedOutput,
                                        netOutput, lastWeights)

    expect_equal(length(lastXInfluence), dim(lastWeights)[2])
    for (i in seq(length(lastXInfluence))) {
      # test with element wise formula
      sum_val <- 0
      for (k in seq(length(netOutput))) {
        sum_val <- sum_val + softmax(netOutput)[k] * lastWeights[k, i]
      }
      expected <- -(lastWeights[expectedOutput,i] - sum_val)
      expect_equal(lastXInfluence[i], expected)
    }
  }
})


test_that("getLastDelta", {
  getLastXInfluence <- getLastXInfluenceL[["classification"]]

  set.seed(10)
  iterations <- 10
  layersizes_hidden <- c(1, round(runif(iterations, min = 1, max = 100)))
  layersizes_output <- c(1, round(runif(iterations, min = 1, max = 100)))
  for (i in seq(iterations + 1)) {
    layersize_hidden <- layersizes_hidden[i]
    layersize_output <- layersizes_output[i]

    input <- round(runif(1, min = 1, max = 100))
    expectedOutput <- round(runif(1, min = 1, max = layersize_output))
    nn <- NeuralNet$new(c(1, layersize_hidden, layersize_output),
                        category="classification")

    netCRes <- nn$calculate(input)
    netOutput <- netCRes$nodeValues[[length(netCRes$nodeValues)]]
    secLastRawNodeVal <- netCRes$rawNodeValues[[length(netCRes$nodeValues) - 1]]
    lastWeights <- nn$weights[[length(nn$weights)]]
    lastXInfluence <- getLastXInfluence(expectedOutput,
                                        netOutput, lastWeights)
    lastDelta <- getLastDelta(lastXInfluence, secLastRawNodeVal,
                              nn$dActfct)

    expect_equal(length(lastDelta), layersize_hidden)
    expect_equal(lastDelta, lastXInfluence * nn$dActfct(secLastRawNodeVal))
  }
})


test_that("getPrevDelta", {
  getLastXInfluence <- getLastXInfluenceL[["classification"]]

  set.seed(10)
  iterations <- 10
  layersizes_hidden1 <- c(1, round(runif(iterations, min = 1, max = 100)))
  layersizes_hidden2 <- c(1, round(runif(iterations, min = 1, max = 100)))
  layersizes_output <- c(1, round(runif(iterations, min = 1, max = 100)))
  for (i in seq(iterations + 1)) {
    layersize_hidden1 <- layersizes_hidden1[i]
    layersize_hidden2 <- layersizes_hidden2[i]
    layersize_output <- layersizes_output[i]

    input <- round(runif(1, min = 1, max = 100))
    expectedOutput <- round(runif(1, min = 1, max = layersize_output))
    nn <- NeuralNet$new(c(1, layersize_hidden1, layersize_hidden2, layersize_output),
                        category="classification")

    netCRes <- nn$calculate(input)
    netOutput <- netCRes$nodeValues[[length(netCRes$nodeValues)]]
    secLastRawNodeVal <- netCRes$rawNodeValues[[length(netCRes$nodeValues) - 1]]
    trdLastRawNodeVal <- netCRes$rawNodeValues[[length(netCRes$nodeValues) - 2]]
    lastWeights <- nn$weights[[length(nn$weights)]]
    secLastWeights <- nn$weights[[length(nn$weights) - 1]]
    lastXInfluence <- getLastXInfluence(expectedOutput,
                                        netOutput, lastWeights)
    lastDelta <- getLastDelta(lastXInfluence, secLastRawNodeVal,
                              nn$dActfct)
    prevDelta <- getPrevDelta(lastDelta, trdLastRawNodeVal, secLastWeights,
                              nn$dActfct)

    expect_equal(length(prevDelta), layersize_hidden1)
    expect_equal(prevDelta, (t(secLastWeights) %*% lastDelta) *
                   nn$dActfct(trdLastRawNodeVal))
  }
})



test_that("getLastWeightsInfluence regression", {
  getLastWeightsInfluence <- getLastWeightsInfluenceL[["regression"]]

  set.seed(10)
  layersizes <- c(1, round(runif(10, min = 1, max = 100)))
  for (layersize in layersizes) {
    input <- round(runif(1, min = 1, max = 100))
    expectedOutput <- round(runif(1, min = 1, max = 100))
    nn <- NeuralNet$new(c(1, layersize, 1), category="regression")

    netCRes <- nn$calculate(input)
    netOutput <- netCRes$nodeValues[[length(netCRes$nodeValues)]]
    secLastNodeVal <- netCRes$nodeValues[[length(netCRes$nodeValues) - 1]]
    lastWeights <- nn$weights[[length(nn$weights)]]
    lastWeightsInfluence <- getLastWeightsInfluence(expectedOutput, netOutput,
                                                    secLastNodeVal)

    expect_equal(dim(lastWeightsInfluence), dim(lastWeights))
    expect_equal(dim(lastWeightsInfluence)[1], 1)
    for (i in seq(dim(lastWeightsInfluence)[2])) {
      # test with element wise formula
      expect_equal(lastWeightsInfluence[1,i],
                   -2 * (expectedOutput - netOutput) * secLastNodeVal[i])
    }
  }
})



test_that("getLastXInfluence classification", {
  getLastWeightsInfluence <- getLastWeightsInfluenceL[["classification"]]

  set.seed(10)
  iterations <- 10
  layersizes_hidden <- c(1, round(runif(iterations, min = 1, max = 10)))
  layersizes_output <- c(1, round(runif(iterations, min = 1, max = 10)))
  for (i in seq(iterations + 1)) {
    layersize_hidden <- layersizes_hidden[i]
    layersize_output <- layersizes_output[i]

    input <- round(runif(1, min = 1, max = 100))
    expectedOutput <- round(runif(1, min = 1, max = layersize_output))
    nn <- NeuralNet$new(c(1, layersize_hidden, layersize_output),
                        category="classification")

    netCRes <- nn$calculate(input)
    netOutput <- netCRes$nodeValues[[length(netCRes$nodeValues)]]
    secLastNodeVal <- netCRes$nodeValues[[length(netCRes$nodeValues) - 1]]
    lastWeights <- nn$weights[[length(nn$weights)]]
    lastWeightsInfluence <- getLastWeightsInfluence(expectedOutput, netOutput,
                                                    secLastNodeVal)

    expect_equal(dim(lastWeightsInfluence), dim(lastWeights))
    for (j in seq(dim(lastWeightsInfluence)[1])) {
      for (m in seq(dim(lastWeightsInfluence)[2])) {
        # test with element wise formula
        expOutputTerm <- 0
        if (j == expectedOutput) {
          expOutputTerm <- 1
        }
        netOutputTerm <- softmax(netOutput)[j]
        expected <- -(expOutputTerm - netOutputTerm) * secLastNodeVal[m]
        expect_equal(lastWeightsInfluence[j,m], expected)
      }
    }
  }
})

# Test for calculateBiasUpdate is skipped! It is just a
# multiplication




test_that("calculateWeightUpdate", {

  set.seed(10)
  iterations <- 10
  matrix_widths <- c(1, round(runif(iterations, min = 1, max = 10)))
  matrix_heights <- c(1, round(runif(iterations, min = 1, max = 10)))
  for (i in seq(iterations + 1)) {
    matrix_width <- matrix_widths[i]
    matrix_height <- matrix_heights[i]

    oldWeights <- matrix(runif(matrix_width * matrix_height, min = -2,
                                     max = 2), nrow = matrix_width)
    weightsInfluence <- matrix(runif(matrix_width * matrix_height, min = -1,
                                     max = 1), nrow = matrix_width)
    lambda <- runif(1, min = 0, max=0.01)
    learning_rate <- runif(1, min = 0, max=0.01)
    weightUpdate <- calculateWeightUpdate(oldWeights, weightsInfluence,
                                          learning_rate, lambda)

    expect_equal(dim(oldWeights), dim(weightUpdate))
    for (j in seq(dim(oldWeights)[1])) {
      for (m in seq(dim(oldWeights)[2])) {
        # test with element wise formula
        expected <- learning_rate * (weightsInfluence[j,m] + 2 * lambda * oldWeights[j,m])
        expect_equal(weightUpdate[j,m], expected)
      }
    }
  }
})
