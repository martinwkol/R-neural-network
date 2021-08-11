test_that("optim", {
  set.seed(10)
  nn <- NeuralNet$new(c(1, 1, 10), category = "classification")
  oldWeights <- rlang::duplicate(nn$weights)
  optimizer <- OptimizerSGD$new(0.1, 0)
  training_data <- list(list(input = 1, expectedOutput = 1))
  optimizer$optim(nn, training_data)

  for (i in seq_len(length(nn$weights))) {
    expect_false(any(nn$weights[[i]] == oldWeights[[i]]))
  }
})
