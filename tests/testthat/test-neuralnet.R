#initialize method
test_that("Testing if the activation function is used correctly", {
  f <- function(x) tanh(x)
  nn <- NeuralNet$new(c(1,3,3,1), f, f, f)
  for(i in runif(10, 0, 100)) {
    expect_equal(f(i), nn$actfct(i))
  }
})

test_that("Testing if the derivative activation function is used correctly", {
  f <- function(x) tanh(x)
  nn <- NeuralNet$new(c(1,3,3,1), f, f, f)
  for(i in runif(10, 0, 100)) {
    expect_equal(f(i), nn$dActfct(i))
  }
})

test_that("Testing if the output activation function is used correctly", {
  f <- function(x) tanh(x)
  nn <- NeuralNet$new(c(1,3,3,1), f, f, f)
  for(i in runif(10, 0, 100)) {
    expect_equal(f(i), nn$outputfct(i))
  }
})

test_that("Testing if classifaction only gives integer values in output range", {
  nn <- NeuralNet$new(c(2, 3, 10, 100))
  for(i in runif(30, 0, 3)) {
    expect_true(nn$calculate(c(i, i))[3] %in% 1:100)
  }
})
