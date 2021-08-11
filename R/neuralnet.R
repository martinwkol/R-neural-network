#hopefully this doesn't need to be moved inside the class when this is a package
predefinedactivation <- list(
  ReLU = function(x) {max(0,x)},
  sigmoid = function(x) {1/(1 + exp(-x))},
  tanh = function(x) {tanh(x)}
)
predefinedactivationDriv <- list(
  ReLU = function(x) {as.integer(x > 0)},
  sigmoid = function(x) {predefinedactivation$sigmoid(x) * (1 - predefinedactivation$sigmoid(x))},
  tanh = function(x) {1/cosh(x)^2}
)

#' R6 Class Representing a Neural Network
#' @export
NeuralNet <- R6::R6Class("NeuralNet",
public = list(

  #' @field inputsize size of the input for the Neural Network (first layer of nodes).
  inputsize = NULL,

  #' @field weights list of matrices representing the weights of the connections
  #' between nodes.
  weights = list(),

  #' @field bias list of vectors representing the biases for nodes.
  bias = list(),

  #' @field nrhiddenlayers number of hidden layers of the Neural Network.
  nrhiddenlayers = NULL,

  #' @field actfct function to be used as activation function.
  actfct = NULL,

  #' @field dActfct function to be used as derivative of the activation function.
  dActfct = NULL,

  #' @field outputfct function to be used as activation function on the last layer.
  outputfct = NULL,

  #' @field category either classification or regression
  category = NULL,

  #' @description
  #' create a new R6 instance of a NeuralNetwork
  #'
  #' @param layers numeric vector with nodes per layer
  #' @param activationfct string matching predefined activation function or
  #' other function, see \code{Activation Functions}.
  #' @param dActivationfct NULL, when using predefined Activation functions, or
  #' the derivative of a custom Activation function, it is not checked for
  #' correcteness.
  #' @param outputfct string matching predefined activation function or
  #' other a function, see \code{Activation Functions}, used as activation
  #' function for the last layer of nodes
  #' @param category string matching valid category for Neural Network, either
  #' classification or regression.
  #'
  #' @details
  #' ## Activation Functions
  #' This Function is used on the Values of a Node after calculating Weights
  #' and Biases. The values for \code{activationfct} and \code{outputfct}
  #' can be either strings matching a predefined Function or a \code{function}.
  #'
  #' The predefined Functions are currently:
  #' * ReLu
  #' * sigmoid
  #' * tanh
  #'
  #' \code{outputfct} can alternatively be \code{NULL} if no activation
  #' function shoud be applied to the last layer.
  #'
  #' When using a custom function to be used, a derivative for this function must
  #' be given to be used while training the Neural Network. This derivative is
  #' not checked for correcteness.
  #'
  #' @examples
  #' # will generate a Neural Network with two input nodes, two output nodes,
  #' # and two hidden layers with four nodes each.
  #' nn.1 <- NeuralNet$new(c(2,4,4,2))
  #'
  #' # will generate a Neural Network using the predefined sigmoid function as
  #' # activation function for all layers (including the outputlayer)
  #' nn.2 <- NeuralNet$new(
  #'   c(2,4,4,2),
  #'   activationfct = "sigmoid",
  #'   outputfct = "sigmoid")
  initialize = function(layers, activationfct = "ReLU", dActivationfct = NULL,
                        outputfct = NULL, category="classification") {

    #choose activation function
    if(class(activationfct) == "character") {
      stopifnot("Specified activation function is not implemented." = !is.null(predefinedactivation[[activationfct]]))
      self$actfct <- predefinedactivation[[activationfct]]
      self$dActfct <- predefinedactivationDriv[[activationfct]]
      #print(self$actfct)
      #print(self$dActfct)
    } else if (class(activationfct) == "function") {
      stopifnot("Derivative of activation function is missing" = class(dActivationfct) == "function")
      self$actfct <- activationfct
      self$dActfct <- dActivationfct
    } else {
      stop("activationfct must be a character or function")
    }

    #choose activation function for outputlayer
    if(is.null(outputfct)) {
      self$outputfct <- function(x){x}
    } else if(class(outputfct) == "character") {
      stopifnot("Specified activation function for the output layer is not
                implemented." = !is.null(predefinedactivation[[outputfct]]))
      self$outputfct <- predefinedactivation[[outputfct]]
    } else if (class(outputfct) == "function") {
      self$outputfct <- outputfct
    } else {
      stop("outputfct must be a character, function or NULL")
    }

    self$inputsize <- layers[1]
    #nr of hidden layers, first is input, last is output
    self$nrhiddenlayers <- length(layers) - 2
    #save size of inputlayer, and remove from layersvector
    lastsize <- layers[1]
    layers <- layers[-1]
    for (lsize in layers) {
      #add the weights, TO DO: initialize weights with other values
      matrix_entries <- rnorm(lsize * lastsize, 0, lsize**-0.5)
      self$weights <- c(self$weights, list(matrix(matrix_entries, lsize, lastsize)))
      lastsize <- lsize
      #create bias TO DO: initialize bias
      self$bias <- c(self$bias, list(rep(0, lsize)))
    }
    #remove last bias because no bias for output layer
    self$bias <- self$bias[-length(layers)]

    if (!(category %in% c("classification", "regression")))
        stop("Unknown category")
    self$category <- category
  },
  #' @description
  #' will calculate an \code{output} for a given \code{input} using the existing
  #' weights and biases, as well as the existing activation and output functions.
  #'
  #' @param input numeric vector of values with length equal the number of nodes
  #' in the input layer.
  #'
  #' @return list with three numeric vectors, containing rawNodeValues,
  #' nodeValues and output.
  calculate = function(input) {
    stopifnot("input size doesn't fit inputlayer size" = length(input) == self$inputsize)

    rawNodeValues <- list()
    nodeValues <- list()
    output <- input

    rawNodeValues[[1]] <- input
    nodeValues[[1]] <- input

    for(i in seq_len(self$nrhiddenlayers)) {
      #weights
      output <- self$weights[[i]]%*%output
      #bias
      output <- output + self$bias[[i]]
      rawNodeValues[[length(rawNodeValues) + 1]] <- output
      #apply the activation function
      'if(!all(!is.nan(output))) {
        print(any(is.nan(self$weights[[i]])))
        print(any(is.nan(self$bias[[i]])))
        print(self$weights[[i]])
        print(self$bias[[i]])
        print("Before Actf")
        stop()
      }'

      output <- sapply(output, self$actfct)
      nodeValues[[length(nodeValues) + 1]] <- output

      'if(!all(!is.nan(output))) {
        print("After Actf")
        stop()
      }'
      #stopifnot(all(!is.nan(output)))
      #print(output)
    }

    output <- self$weights[[self$nrhiddenlayers + 1]] %*% output
    rawNodeValues[[length(rawNodeValues) + 1]] <- output
    output <- sapply(output, self$outputfct)
    nodeValues[[length(nodeValues) + 1]] <- output

    stopifnot(all(!is.nan(output)))

    if (self$category == "classification") {
      output <- which.max(output)
    }

    list(rawNodeValues = rawNodeValues,
         nodeValues = nodeValues,
         output = output)
  },

  #' @description
  #' plot an R6 NeuralNet using graphics and grDevices
  #'
  #' @param max.lwd numeric value representing the maximum linewidth, default is 5.
  #' @param standard.lwd logical value, should all lines should be normalized.
  #' @param col.fct a function determining the color for specific values.
  #' @param image.save logical value, should the image be saved as file.
  #' @param image.filename name of the outputfile, if \code{image.save = TRUE}
  #' filepath and filename used to save the image.
  #' @param image.type name of the filetype, if \code{image.save = TRUE} this
  #' filetype will be used to save the image.
  #'
  #'
  #' @details
  #' ## Color Functions
  #' \code{col.fct} will be given a vector with values between -1 and 1.
  #' They must return a vector of colors usable by the graphics package
  #' (see the \code{col} Argument of \code{?graphics::par}).
  #'
  #' If the Color Function is \code{NULL} all colors will be black.
  #'
  #' ## Saving Images
  #' When \code{image.save} is set to TRUE, \code{NeuralNet$plot} will attempt
  #' to save the shown graphic as file.
  #'
  #' The file will be saved using the \code{image.filename} Argument, this can
  #' be a path to a file. If the path does not exist \code{grDevices::device()}
  #' will give an error. If \code{image.filename} does not contain a valid file
  #' extension to be used with the type specified in \code{image.type}, an
  #' extension will be added.
  #'
  #' Valid filetypes are:
  #' * \code{wmf} and \code{emf} using \code{grDevices::win.metafile}
  #' * \code{png} using \code{grDevices::png}
  #' * \code{jpg} and \code{jpeg} using \code{grDevices::jpeg}
  #' * \code{bmp} using \code{grDevices::bmp}
  #' * \code{tif} and \code{tiff} using \code{grDevices::tiff}
  #' * \code{ps} and \code{eps} using \code{grDevices::postscript}
  #' * \code{pdf} using \code{grDevices::pdf}
  #' * \code{svg} using \code{grDevices::svg}
  #'
  #' @examples
  #' nn <- NeuralNet$new(c(2,4,4,2))
  #'
  #' # will plot the Neural Network
  #' nn$plot()
  #'
  #' # will plot the Neural Network with normalized linewidth
  #' nn$plot(standard.lwd = TRUE)
  #'
  #' # will plot the Neural Network with only two colors,
  #' # red for negative values, green for positive values
  #' nn$plot(col.fct = function(x) {
  #'   b <- (x < 0)
  #'   v <- vector(length = length(b))
  #'   v[b] <- "red"
  #'   v[!b] <- "green"
  #'
  #'   v
  #' })
  #'
  #' # saving a plot
  #'
  #' # file will be saved as 'file.png'
  #' nn$plot(image.save = TRUE,
  #'   image.filename = "file",
  #'   image.type = "png")
  #'
  #' # here '.jpeg' is not a valid extension for type 'png', '.png' will
  #' # be appendedto the filename, file will be saved as 'file.jpeg.png'
  #' nn$plot(image.save = TRUE,
  #'   image.filename = "file.jpeg",
  #'   image.type = "png")
  #'
  #' # file will be saved in directory 'test/', if 'test/' does not exist
  #' # grDevices will return an error
  #' nn$plot(image.save = TRUE,
  #'   image.filename = "test/file.jpg",
  #'   image.type = "jpg")
  #'
  plot = function(max.lwd = 5, standard.lwd = FALSE, col.fct = function(x) { grDevices::hcl(x * 60 + 60) },
                  image.save = FALSE, image.filename = NULL, image.type = NULL) {
    stopifnot("col.fct must be a function" = identical(class(col.fct), "function"))
    if(image.save){
      stopifnot("image.filename must not be NULL" = !is.null(image.filename))
      stopifnot("image.type must not be NULL" = !is.null(image.type))

      valid_image_types <- c("wmf", "emf", "png", "jpg", "jpeg", "bmp",
                             "tif", "tiff", "ps", "eps", "pdf", "svg")
      txt <- paste("image.type must be one of", valid_image_types)
      stopifnot(txt =
                  {image.type %in% valid_image_types})
    }


    # same as 'layers' of 'initialize()'
    p <- c(self$inputsize, sapply(self$weights, function(x) nrow(x)))

    # offsets
    x_diff <- 1 / (length(p) - 1)
    b_off <- 0.3 * x_diff

    # x coords of first layer
    x.1 <- rep(0, times = p[1])
    # y coords of nodes in first layer
    y.1 <- 1:p[1] / (p[1] + 1)
    # y coord of bias node is always fix
    y.b <- 0.9

    # empty arrays:
    y <- c() # y coordinates of points
    x <- c() # x coordinates of points
    l <- c() # labels of points

    # variables for segments() call
    x0 <- c()
    y0 <- c()
    x1 <- c()
    y1 <- c()
    lwd <- c()

    for(layer in 1:(length(p) - 1)){
      ## calculate positions

      # x coord of current layer
      ## gets set from the outside of the loop or
      ## from the last iteration of the loop

      # x coord of next layer
      x.2 <- rep(x.1[1] + x_diff, times = p[layer + 1])
      # x coord of bias
      x.b <- x.1[1] + b_off

      # y coords of nodes in current layer
      ## gets set from the outside of the loop or
      ## from the last iteration of the loop

      # y coords of nodes in next layer
      y.2 <- 1:p[layer+1] / (p[layer+1] + 1)

      # y coord of bias node is fix

      if(length(self$bias) < layer){
        x.1.all <- rep(x.1, each = p[layer + 1])
        y.1.all <- rep(y.1, each = p[layer + 1])
        x.2.all <- rep(x.2, times = p[layer])
        y.2.all <- rep(y.2, times = p[layer])
      } else {
        x.1.all <- rep(c(x.1, x.b), each = p[layer + 1])
        y.1.all <- rep(c(y.1, y.b), each = p[layer + 1])
        x.2.all <- rep(x.2, times = (p[layer] + 1))
        y.2.all <- rep(y.2, times = (p[layer] + 1))
      }

      stopifnot("something went wrong,x.1.all and y.1.all are of different
                lengths" = length(x.1.all) == length(y.1.all))
      stopifnot("something went wrong, x.1.all and x.2.all are of different
                lengths" = length(x.1.all) == length(x.2.all))
      stopifnot("something went wrong, y.1.all and y.2.all are of different
                lengths" = length(y.1.all) == length(y.2.all))

      x0 <- c(x0, x.1.all)
      y0 <- c(y0, y.1.all)
      x1 <- c(x1, x.2.all)
      y1 <- c(y1, y.2.all)

      # linewidths
      if (length(self$bias) < layer) {
        lwd <- c(lwd, as.vector(self$weights[[layer]]))
      } else {
        lwd <- c(lwd, as.vector(self$weights[[layer]]),  self$bias[[layer]])
      }


      #lwd <- c(lwd, rep(1, length.out = length(x.1.all)))

      # adding coordinates to arrays
      if (length(self$bias) < layer) {
        y <- c(y, y.1)
        x <- c(x, x.1)
        l <- c(l, paste(layer, ",", p[layer]:1, sep = ""))
      } else {
        y <- c(y, y.1, y.b)
        x <- c(x, x.1, x.b)
        l <- c(l, paste(layer, ",", p[layer]:1, sep = ""), paste("b ", layer, sep = ""))
      }

      # setting x.1 and y.1 coords for next iteration of loop
      y.1 <- y.2
      x.1 <- x.2
    }

    # adding coordinates of last layer to arrays
    y <- c(y, y.1)
    x <- c(x, x.1)
    l <- c(l, paste(layer, ",", p[length(p)]:1, sep = ""))

    # drawing lines
    if(standard.lwd) lwd <- (0.5 - 1 * (lwd < 0)) * (lwd != 0)
    else lwd <- (lwd / max(abs(lwd)))

    if(is.null(col.fct)){
      col <- "black"
    } else {
      col <- col.fct(lwd)
    }

    # initialize plot
    graphics::par(mar = c(2,2,2,2))
    graphics::plot(x = c(0,1), y = c(0,1), type="n", axes=FALSE, xlab = "", ylab = "")

    # drawing lines
    graphics::segments(x0 = x0, y0 = y0, x1 = x1, y1 = y1, lwd = (abs(lwd) * max.lwd), col = col)

    # drawing points
    graphics::points(x, y, pch = 21, cex = 3, bg = "white")

    #adding labels
    graphics::text(x, y, labels = l, pos = 3, offset = 1)

    # saving image
    if(image.save){

      # ensuring that file extension contained in the filename,
      # some of the grDevice functions can do this automaticaly, but not all
      if(identical(grep(paste(".", image.type, sep = ""), image.filename, ignore.case = TRUE), integer(0)))
        image.filename <- paste(image.filename, ".", image.type, sep = "")

      # selecting correct grDevices function for the filetype
      if(image.type %in% c("emf", "wmf"))
        image.fct <- grDevices::win.metafile
      else if(image.type %in% c("png"))
        image.fct <- grDevices::png
      else if(image.type %in% c("jpg", "jpeg"))
        image.fct <- grDevices::jpeg
      else if(image.type %in% c("bmp"))
        image.fct <- grDevices::bmp
      else if(image.type %in% c("tif", "tiff"))
        image.fct <- grDevices::tiff
      else if(image.type %in% c("ps", "eps"))
        image.fct <- grDevices::postscript
      else if(image.type %in% c("pdf"))
        image.fct <- grDevices::pdf
      else if(image.type %in% c("svg"))
        image.fct <- grDevices::svg

      # copying currently displayed plot and saving it using image.fct
      # with the name image.filename
      grDevices::dev.copy(image.fct, image.filename)
      grDevices::dev.off()
    }
    }
  )
)


myn <- NeuralNet$new(
  c(1,3,3,1),
  activationfct =  function(x) x,
  dActivationfct = function(x) 1)

myn$weights
myn$bias
myn$calculate(-3)


a <- matrix(0.5, nrow=3, ncol=1)
b <- matrix(0.5, nrow=3, ncol=3)
c <- matrix(0.5, nrow=1, ncol=3)
c%*%b%*%a%*%-3

a

x <- c(0,1,1)

a+x


# for testing NeuralNet$plot()
nn <- NeuralNet$new(c(2,4,4,4,2))

# setting weights to be random
nn$weights <- list(
  matrix(runif(8, min = -1), nrow = 4),
  matrix(runif(16, min = -1), nrow = 4),
  matrix(runif(16, min = -1), nrow = 4),
  matrix(runif(8, min = -1), nrow = 2)
)

# setting bias to be random
nn$bias <- list(
  runif(4, min = -1),
  runif(4, min = -1),
  runif(4, min = -1)
  )

nn$weights
nn$bias
nn$plot(image.save = TRUE, image.filename = ".test.png", image.type = "png")

### will not change line width
#nn$plot(standard.lwd = TRUE)

### will not change colors
#nn$plot(col.fct = NULL)

### will change neiter line width nor color
#nn$plot(standard.lwd = TRUE, col.fct = NULL)
