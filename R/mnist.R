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

    labels + 1
  },

  load_images = function(filename) {
    file_obj <- file(filename, "rb")
    nums <- readBin(file_obj, integer(), n = 4, size = 4,
                    endian = "big")
    number_of_items <- nums[2]

    # not important
    rows <- nums[3]
    columns <- nums[4]
    num_pixels <- rows * columns

    images <- list()
    for(i in seq(number_of_items)) {
      pixels <- readBin(file_obj, integer(), n = num_pixels,
                        size = 1, signed = F, endian = "big")
      images[[length(images) + 1]] <- pixels / 255
    }
    close(file_obj)

    images
  }
),

public = list(
  training_data = NULL,
  test_data = NULL,
  initialize = function(training_labels_fn, training_images_fn,
                        test_labels_fn, test_images_fn) {
    training_labels <- private$load_labels(training_labels_fn)
    training_images <- private$load_images(training_images_fn)
    test_labels <- private$load_labels(test_labels_fn)
    test_images <- private$load_images(test_images_fn)

    self$training_data <- combineData(training_images, training_labels)
    self$test_data <-     combineData(test_images, test_labels)
  }
))



