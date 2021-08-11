test_that("combineData", {
  inputs <- as.list(1:100)
  targets <- as.list(100:1)
  combined <- combineData(inputs, targets)
  expect_equal(length(inputs), length(combined))
  for (i in seq(length(combined))) {
    expect_equal(inputs[[i]], combined[[i]]$input)
    expect_equal(targets[[i]], combined[[i]]$expectedOutput)
  }
})

test_that("combineData different length", {
  inputs <- as.list(1:50)
  targets <- as.list(100:1)
  expect_error(combineData(inputs, targets))
})

test_that("Initialize Trainer no training / test data", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  training_data <- NULL
  test_data <- NULL

  trainer <- Trainer$new(nn, optimizer, training_data, test_data)

  expect_identical(nn, trainer$getNeuralnet())
  expect_identical(optimizer, trainer$getOptimizer())
  expect_identical(NULL, trainer$getTrainingData())
  expect_identical(NULL, trainer$getTestData())
})

test_that("Initialize Trainer classification", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  training_data <- combineData(as.list(1:10), as.list(10:1))
  test_data <- combineData(as.list(1:2), as.list(2:1))

  trainer <- Trainer$new(nn, optimizer, training_data, test_data)

  expect_identical(nn, trainer$getNeuralnet())
  expect_identical(optimizer, trainer$getOptimizer())
  expect_identical(training_data, trainer$getTrainingData())
  expect_identical(test_data, trainer$getTestData())
})

test_that("Initialize Trainer regression", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "regression")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  training_data <- combineData(as.list(1:10), as.list(10:1))
  test_data <- combineData(as.list(1:2), as.list(2:1))
  accuracy_tester <- accuracy_tester_regression_abs(0.5)

  trainer <- Trainer$new(nn, optimizer, training_data, test_data,
                         accuracy_tester)

  expect_identical(nn, trainer$getNeuralnet())
  expect_identical(optimizer, trainer$getOptimizer())
  expect_identical(training_data, trainer$getTrainingData())
  expect_identical(test_data, trainer$getTestData())
})

test_that("Initialize Trainer regression / no accuracy tester", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "regression")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  training_data <- combineData(as.list(1:10), as.list(10:1))
  test_data <- combineData(as.list(1:2), as.list(2:1))
  accuracy_tester <- accuracy_tester_regression_abs(0.5)

  expect_error(Trainer$new(nn, optimizer, training_data, test_data))
})

test_that("Initialize Trainer regression / no accuracy tester", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "regression")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  training_data <- combineData(as.list(1:10), as.list(10:1))
  test_data <- combineData(as.list(1:2), as.list(2:1))
  accuracy_tester <- accuracy_tester_regression_abs(0.5)

  expect_error(Trainer$new(nn, optimizer, training_data, test_data))
})

test_that("Initialize Trainer classification / no optimizer", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "regression")
  training_data <- combineData(as.list(1:10), as.list(10:1))
  test_data <- combineData(as.list(1:2), as.list(2:1))

  expect_error(Trainer$new(nn, training_data = training_data,
                           test_data = test_data))
})

test_that("Initialize Trainer regression / no neural network", {
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  training_data <- combineData(as.list(1:10), as.list(10:1))
  test_data <- combineData(as.list(1:2), as.list(2:1))
  accuracy_tester <- accuracy_tester_regression_abs(0.5)

  expect_error(Trainer$new(optimizer = optimizer, training_data = training_data,
                           test_data = test_data))
})

test_that("setNeuralnet", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  newNN <- NeuralNet$new(c(3, 10, 10, 1), category = "regression")
  trainer$setNeuralnet(newNN)
  expect_identical(newNN, trainer$getNeuralnet())
})

test_that("setNeuralnet null", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  expect_error(trainer$setNeuralnet(NULL))
})


test_that("setOptimizer", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  newOpt <- OptimizerNesterovAG$new(0.01, 0.2, 0.5)
  trainer$setOptimizer(newOpt)
  expect_identical(newOpt, trainer$getOptimizer())
})

test_that("setOptimizer null", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  expect_error(trainer$setOptimizer(NULL))
})


test_that("separateData", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  data <- combineData(as.list(1:20), as.list(20:1))
  trainer$separateData(data, test_percentage = 0.1)
  expect_equal(length(trainer$getTrainingData()), 18)
  expect_equal(length(trainer$getTestData()), 2)
  for (d in data) {
    inTraining <- any(sapply(trainer$getTrainingData(), function(td) {
      td$input == d$input && td$expectedOutput == d$expectedOutput
    }))
    inTest <- any(sapply(trainer$getTestData(), function(td) {
      td$input == d$input && td$expectedOutput == d$expectedOutput
    }))
    expect_true((inTraining && !inTest) || (!inTraining && inTest))
  }
})

test_that("separateData / data not a list", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  expect_error(trainer$separateData("data", test_percentage = 0.1))
})

test_that("separateData / Test percentage out of range", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  data <- combineData(as.list(1:20), as.list(20:1))
  expect_error(trainer$separateData(data, test_percentage = -0.1))
  expect_error(trainer$separateData(data, test_percentage = 5))
})

test_that("separateData / Wrong data format", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  expect_error(trainer$separateData(list(1, "L")))
  expect_error(trainer$separateData(list(1, list(input=1, expectedOutput=4))))
  expect_error(trainer$separateData(list(list(input = 5),list(expectedOutput = 5) )))
})

test_that("generateTrainingTest", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  trainer$generateTrainingTest(as.list(1:20), as.list(20:1), test_percentage = 0.1)
  expect_equal(length(trainer$getTrainingData()), 18)
  expect_equal(length(trainer$getTestData()), 2)
  for (input in 1:20) {
    d <- list(input = input, expectedOutput = 21 - input)
    inTraining <- any(sapply(trainer$getTrainingData(), function(td) {
      td$input == d$input && td$expectedOutput == d$expectedOutput
    }))
    inTest <- any(sapply(trainer$getTestData(), function(td) {
      td$input == d$input && td$expectedOutput == d$expectedOutput
    }))
    expect_true((inTraining && !inTest) || (!inTraining && inTest))
  }
})

test_that("generateTrainingTest / different length", {
  nn <- NeuralNet$new(c(1, 3, 1), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  expect_error(trainer$generateTrainingTest(as.list(1:20), as.list(10:1)))
  expect_error(trainer$generateTrainingTest(as.list(1:50), as.list(100:1)))
})

test_that("test classification", {
  nn <- NeuralNet$new(c(10, 10), category = "classification")
  nn$weights <- list(diag(10))
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  test_data_1 <- combineData(lapply(1:10, function(x) as.integer(x == 1:10)),
                             rep(1, 10))
  trainer$setTestData(test_data_1)
  expect_equal(trainer$test(), 0.1)

  test_data_2 <- combineData(lapply(1:10, function(x) as.integer(x == 1:10)),
                             1:10)
  trainer$setTestData(test_data_2)
  expect_equal(trainer$test(), 1)

  test_data_3 <- combineData(lapply(1:10, function(x) as.integer(x == 1:10)),
                             c(1:3, 14:20))
  trainer$setTestData(test_data_3)
  expect_equal(trainer$test(), 0.3)
})

test_that("test regression absolute accuracy tester", {
  nn <- NeuralNet$new(c(2, 1), category = "regression")
  nn$weights <- list(matrix(c(1, -1), ncol=2, byrow = T))
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  accuracy_tester_abs <- accuracy_tester_regression_abs(1)
  trainer <- Trainer$new(nn, optimizer, accuracy_tester = accuracy_tester_abs)

  test_data_1 <- combineData(lapply(1:10, function(x) c(x, 5)),
                             rep(0, 10))
  trainer$setTestData(test_data_1)
  expect_equal(trainer$test(), 0.3)

  test_data_2 <- combineData(lapply(11:20, function(x) c(x, 5)),
                             rep(10, 10))
  trainer$setTestData(test_data_2)
  expect_equal(trainer$test(), 0.3)

  test_data_3 <- combineData(lapply(11:20, function(x) c(x, 5)),
                             6:15)
  trainer$setTestData(test_data_3)
  expect_equal(trainer$test(), 1)
})

test_that("test regression relative accuracy tester", {
  nn <- NeuralNet$new(c(2, 1), category = "regression")
  nn$weights <- list(matrix(c(1, -1), ncol=2, byrow = T))
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  accuracy_tester_abs <- accuracy_tester_regression_rel(0.1)
  trainer <- Trainer$new(nn, optimizer, accuracy_tester = accuracy_tester_abs)

  test_data_1 <- combineData(lapply(1:10, function(x) c(x, 0)),
                             rep(10, 10))
  trainer$setTestData(test_data_1)
  expect_equal(trainer$test(), 0.2)

  test_data_2 <- combineData(lapply(6:15, function(x) c(x, 0)),
                             rep(10, 10))
  trainer$setTestData(test_data_2)
  expect_equal(trainer$test(), 0.3)

  test_data_3 <- combineData(lapply(6:15, function(x) c(x, 0)),
                             rep(16, 10))
  trainer$setTestData(test_data_3)
  expect_equal(trainer$test(), 0.1)
})

test_that("test / no test data", {
  nn <- NeuralNet$new(c(2, 1), category = "regression")
  nn$weights <- list(matrix(c(1, -1), ncol=2, byrow = T))
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  accuracy_tester_abs <- accuracy_tester_regression_rel(0.1)
  trainer <- Trainer$new(nn, optimizer, accuracy_tester = accuracy_tester_abs)

  expect_error(trainer$test())
})




test_that("train no early stopping", {
  optimCalls <- 0
  training_data_sum <- 0

  nn <- NeuralNet$new(c(1, 3, 2), category = "classification")
  optimizer <- R6::R6Class("TOpt", public = list(
    optim = function(nn, td) {
      optimCalls <<- optimCalls + 1
      training_data_sum <<- training_data_sum + length(td)
    },reset = function() {}
  ))$new()
  accuracy_tester_abs <- accuracy_tester_regression_rel(0.1)
  trainer <- Trainer$new(nn, optimizer, accuracy_tester = accuracy_tester_abs)
  trainer$generateTrainingTest(as.list(1:2000), as.list(rep(c(1, 2), 1000)), test_percentage = 0.1)

  expect_true(trainer$train(50, 200, use_early_stopping = F))
  expect_equal(optimCalls, 50)
  expect_equal(training_data_sum, 50 * 200)
})

test_that("train early stopping", {
  set.seed(10)

  optimCalls <- 0
  training_data_sum <- 0

  nn <- NeuralNet$new(c(1, 1), category = "regression")
  nn$weights <- list(diag(1))
  optimizer <- R6::R6Class("TOpt", public = list(
    optim = function(nn, td) {
      optimCalls <<- optimCalls + 1
      training_data_sum <<- training_data_sum + length(td)
    },reset = function() {}
  ))$new()
  accuracy_tester_abs <- accuracy_tester_regression_abs(0.1)
  trainer <- Trainer$new(nn, optimizer, accuracy_tester = accuracy_tester_abs)
  trainer$generateTrainingTest(as.list(1:2000), as.list(rep(1, 2000)), test_percentage = 0.1)
  trainer$setTestData(list(list(input = 1, expectedOutput = 1)))

  expect_true(trainer$train(50, 200, use_early_stopping = T, es_test_frequency = 50,
                es_test_size = 5, es_minimal_improvement = 0))
  expect_equal(optimCalls, 200)
  expect_equal(training_data_sum, 200 * 50)

  optimCalls <- 0
  training_data_sum <- 0

  expect_false(trainer$train(50, 200, use_early_stopping = T, es_test_frequency = 50,
                            es_test_size = 5, es_minimal_improvement = 1))
  expect_equal(optimCalls, 1)
  expect_equal(training_data_sum, 50)

  newOptimizer <- R6::R6Class("TOpt", public = list(
    optim = function(nn, td) {
      optimCalls <<- optimCalls + 1
      training_data_sum <<- training_data_sum + length(td)
      if (optimCalls >= 5) {
        # ruin the network
        nn$weights <- matrix(0)
      }
    },reset = function() {}
  ))$new()
  trainer$setOptimizer(newOptimizer)

  optimCalls <- 0
  training_data_sum <- 0

  expect_false(trainer$train(50, 200, use_early_stopping = T, es_test_frequency = 50,
                             es_test_size = 5, es_minimal_improvement = 0))
  expect_equal(optimCalls, 5)
  expect_equal(training_data_sum, 50 * 5)
})

test_that("train / no training data", {
  nn <- NeuralNet$new(c(1, 3, 2), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)

  expect_error(trainer$train(1))
})

test_that("train / negative epoch or training per epoch", {
  nn <- NeuralNet$new(c(1, 3, 2), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)
  trainer$generateTrainingTest(as.list(1:200), as.list(rep(c(1, 2), 100)),
                            test_percentage = 0.1)

  expect_error(trainer$train(-1))
  expect_error(trainer$train(5, -9))
})

test_that("train early stopping / no test data", {
  nn <- NeuralNet$new(c(1, 3, 2), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)
  trainer$generateTrainingTest(as.list(1:200), as.list(rep(c(1, 2), 100)), test_percentage = 0.1)
  trainer$setTestData(NULL)

  expect_error(trainer$train(2, use_early_stopping = T))
})

test_that("train early stopping / test frequency or test size 0 or less", {
  nn <- NeuralNet$new(c(1, 3, 2), category = "classification")
  optimizer <- OptimizerMomentum$new(0.0005, 0.00001, 0.9)
  trainer <- Trainer$new(nn, optimizer)
  trainer$generateTrainingTest(as.list(1:200), as.list(rep(c(1, 2), 100)), test_percentage = 0.1)

  expect_error(trainer$train(2, use_early_stopping = T, es_test_frequency = -1))
  expect_error(trainer$train(2, use_early_stopping = T, es_test_frequency = 0))
  expect_error(trainer$train(2, use_early_stopping = T, es_test_size = 0))
  expect_error(trainer$train(2, use_early_stopping = T, es_test_size = -10))
})

