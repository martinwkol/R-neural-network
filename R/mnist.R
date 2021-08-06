MNIST <- R6::R6Class("MNIST",
private = list(
  load_labels = function(filename) {
    file_obj <- file(filename, "rb")
    nums <- readBin(file_obj, integer(), n = 2, size = 4,
                    endian = "big")
    number_of_items <- nums[2]

    labels <- readBin(file_obj, integer(),
                      n = number_of_items, size = 1,
                      endian = "big")
    close(file_obj)

    labels
  },

  load_images = function(filename) {
    file_obj <- file(filename, "rb")
    nums <- readBin(file_obj, integer(), n = 4, size = 4,
                    endian = "big")
    number_of_items <- nums[2]

    # not important
    # rows <- nums[3]
    # columns <- nums[4]

    pixels <- readBin(file_obj, integer(),
                      n = number_of_items, size = 1,
                      endian = "big")
    close(file_obj)

    pixels
  }
),

public = list(
  trainingData = list(),
  testData = list(),
  initialize = function(training_labels_fn, training_images_fn,
                        test_labels_fn, test_images_fn) {
    training_labels <- self$load_labels(training_labels_fn)
    training_images <- self$load_images(training_images_fn)
    test_labels <- self$load_labels(test_labels_fn)
    test_images <- self$load_images(test_images_fn)

    self$training_data <- mapply(\(label, image)
                                list(input=image,
                                     expectedOutput=label),
                                training_labels,
                                training_images,
                                SIMPLIFY = F)
    self$test_data <-     mapply(\(label, image)
                                list(input=image,
                                     expectedOutput=label),
                                test_labels,
                                test_images,
                                SIMPLIFY = F)
  }
))



