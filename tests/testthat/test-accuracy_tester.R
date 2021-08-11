test_that("accuracy_tester_classification", {
  accuracy_tester <- accuracy_tester_classification()
  for(i in seq(5)) {
    for(j in seq(5)) {
      expect_equal(accuracy_tester(i, j), i == j)
    }
  }
})

test_that("accuracy_tester_regression_rel", {
  for (maxAbsoluteDiviation in (1:10) / 20) {
    accuracy_tester <- accuracy_tester_regression_rel(maxAbsoluteDiviation)
    for(netOutput in seq(5) * 10) {
      for(expectedOutput in seq(5) * 10) {
        # Determine accuracy using a different algorithm
        minVal <- expectedOutput * (1 - maxAbsoluteDiviation)
        maxVal <- expectedOutput * (1 + maxAbsoluteDiviation)
        expected <- minVal <= netOutput && netOutput <= maxVal
        expect_equal(accuracy_tester(netOutput, expectedOutput),
                     expected)
      }
    }
  }
})

test_that("accuracy_tester_regression_abs", {
  for (maxAbsoluteDiviation in (1:10)) {
    accuracy_tester <- accuracy_tester_regression_abs(maxAbsoluteDiviation)
    for(netOutput in seq(5) * 3) {
      for(expectedOutput in seq(5) * 3) {
        # Determine accuracy using a different algorithm
        minVal <- expectedOutput - maxAbsoluteDiviation
        maxVal <- expectedOutput + maxAbsoluteDiviation
        expected <- minVal <= netOutput && netOutput <= maxVal
        expect_equal(accuracy_tester(netOutput, expectedOutput),
                     expected)
      }
    }
  }
})
