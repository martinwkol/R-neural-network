library(stringr)

devtools::load_all("./R/")
mnist_folder <- "./mnist/"
mnist <- MNIST$new(training_labels_fn = str_c(mnist_folder, "train-labels.idx1-ubyte"),
                   training_images_fn = str_c(mnist_folder, "train-images.idx3-ubyte"),
                   test_labels_fn = str_c(mnist_folder, "t10k-labels.idx1-ubyte"),
                   test_images_fn = str_c(mnist_folder, "t10k-images.idx3-ubyte"))

## MINST with classification
nn <- NeuralNet$new(c(784,  200, 10), activationfct="ReLU", category="classification")
optimizer <- OptimizerSGD$new(0.0005, 0)
trainer <- Trainer$new(nn, optimizer, NULL, NULL)
trainer$separateData(c(mnist$training_data, mnist$test_data))

trainer$train(1, use_early_stopping = T,
              es_test_frequency = 1000, es_test_size = 100,
              es_minimal_improvement = -0.05)
trainer$train(1, use_early_stopping = T,
              es_test_frequency = 1000, es_test_size = 100,
              es_minimal_improvement = -0.02)
trainer$train(1, 500)
trainer$train(1, 1000)
trainer$train(1, 100)

trainer$test(500)
trainer$swapWithBestNeuralnet()

trainer$setNeuralnet(nn)

## MINST with regression
nn <- NeuralNet$new(c(784,200, 1), activationfct="ReLU", category="regression")
optimizer <- OptimizerSGD$new(0.3, 0)
trainer <- Trainer$new(nn, optimizer, mnist$training_data, mnist$test_data,
                       measurement_regression_abs(0.5))

trainer$train(100)
trainer$train(500)
trainer$train(1000)

trainer$test(500)

bestnn <- trainer$getBestNeuralnet()
trainer$


# MINST with MomentumOpt
nn <- NeuralNet$new(c(784,  200, 10), activationfct="ReLU", category="classification")
optimizer <- OptimizerMomentum$new(0.3, 0, 0.9)
trainer <- Trainer$new(nn, optimizer, NULL, NULL)
trainer$separateData(c(mnist$training_data, mnist$test_data))

trainer$train(1, use_early_stopping = T,
              es_test_frequency = 1000, es_test_size = 100,
              es_minimal_improvement = -0.02)
trainer$train(1, use_early_stopping = T,
              es_test_frequency = 1000, es_test_size = 100,
              es_minimal_improvement = -0.02)
trainer$train(1, 500)
trainer$train(1, 1000)
trainer$train(1, 100)

trainer$test(500)




# MINST with MomentumNesterovAG
nn <- NeuralNet$new(c(784,  200, 10), activationfct="ReLU", category="classification")
optimizer <- OptimizerNesterovAG$new(0.0005, 0, 0.9)
trainer <- Trainer$new(nn, optimizer, NULL, NULL)
trainer$separateData(c(mnist$training_data, mnist$test_data))

trainer$train(1, 10000, use_early_stopping = T,
              es_test_frequency = 1000, es_test_size = 100,
              es_minimal_improvement = -0.02)
trainer$train(1, use_early_stopping = T,
              es_test_frequency = 1000, es_test_size = 100,
              es_minimal_improvement = -0.03)
trainer$train(1, 500)
trainer$train(1, 1000)
trainer$train(1, 100)

trainer$test(500)
