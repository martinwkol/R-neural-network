library(R6)

#hopefully this doesn't need to be moved inside the class when this is a package
predefinedactivation <- list(
  ReLU = function(x) {max(0,x)},
  sigmoid = function(x) {1/(1 + e^(-x))},
  tanh = function(x) {tanh(x)}
)

NeuralNet <- R6Class("NeuralNet",
                     public = list(
                       inputsize = NULL,
                       weights = list(),
                       bias = list(),
                       nrhiddenlayers = NULL,
                       actfct = NULL,
                       outputfct = NULL,
                       initialize = function(layers, activationfct = "ReLU", outputfct = NULL) {
                         #choose activation function
                         if(class(activationfct) == "character") {
                           stopifnot("Specified activation function is not implemented." = !is.null(predefinedactivation[[activationfct]]))
                           self$actfct <- predefinedactivation[[activationfct]]
                         } else if (class(activationfct) == "function") {
                           self$actfct <- activationfct
                         } else {
                           stop("activationfct must be a character or function")
                         }
                         #choose activation function for outputlayer
                         if(is.null(outputfct)) {
                           self$outputfct <- function(x){x}
                         } else if(class(outputfct) == "character") {
                           stopifnot("Specified activation function for the output layer is not implemented." = !is.null(predefinedactivation[[outputfct]]))
                           self$outputfct <- predefinedactivation[[outputfct]]
                         } else if (class(outputfct) == "function") {
                           self$actfct <- outputfct
                         } else {
                           stop("outputfct must be a character, function or NULL")
                         }
                         #
                         #
                         #
                         self$inputsize <- layers[1]
                         #nr of hidden layers, first is input, last is output
                         self$nrhiddenlayers <- length(layers) - 2
                         #save size of inputlayer, and remove from layersvector
                         lastsize <- layers[1]
                         layers <- layers[-1]
                         for (lsize in layers) {
                           #add the weights, TO DO: initialize weights with other values
                           self$weights <- c(self$weights, list(matrix(0.5, lsize, lastsize)))
                           lastsize <- lsize
                           #create bias TO DO: initialize bias
                           self$bias <- c(self$bias, list(rep(0, lsize)))
                         }
                         #remove last bias because no bias for output layer
                         self$bias <- self$bias[-length(layers)]
                       },
                       calculate = function(input) {
                         stopifnot("input size doesn't fit inputlayer size" = length(input) == self$inputsize)
                         #don't override input, it probably needs to be saved for some learning methods
                         output <- input
                         #for(w in self$weights[1:self$nrhiddenlayers]) {
                         for(i in 1:self$nrhiddenlayers) {
                           #weights
                           output <- self$weights[[i]]%*%output
                           #bias
                           output <- output + self$bias[[i]]
                           #apply the activation function
                           output <- sapply(output, self$actfct)
                         }
                         #calculate weigths to output layer and the output activation function if specified(no bias)
                         output <- self$weights[[self$nrhiddenlayers + 1]] %*% output
                         output <- sapply(output, self$outputfct)
                         output
                       },
                       plot = function() {
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


                           x.1.all <- rep(c(x.1, x.b), each = p[layer + 1])
                           y.1.all <- rep(c(y.1, y.b), each = p[layer + 1])
                           x.2.all <- rep(x.2, times = (p[layer] + 1))
                           y.2.all <- rep(y.2, times = (p[layer] + 1))

                           stopifnot("something went wrong,
                                     x.1.all and y.1.all are of different lengths"
                                     = length(x.1.all) == length(y.1.all))
                           stopifnot("something went wrong,
                                     x.1.all and x.2.all are of different lengths"
                                     = length(x.1.all) == length(x.2.all))
                           stopifnot("something went wrong,
                                     y.1.all and y.2.all are of different lengths"
                                     = length(y.1.all) == length(y.2.all))

                           x0 <- c(x0, x.1.all)
                           y0 <- c(y0, y.1.all)
                           x1 <- c(x1, x.2.all)
                           y1 <- c(y1, y.2.all)

                           #lwd <- c(lwd, rep(1, length.out = length(x.1.all)))

                           # adding coordinates to arrays
                           y <- c(y, y.1, y.b)
                           x <- c(x, x.1, x.b)
                           l <- c(l, paste(layer, ",", p[layer]:1), paste("b", layer))

                           # setting x.1 and y.1 coords for next iteration of loop
                           y.1 <- y.2
                           x.1 <- x.2
                         }

                         # adding coordinates of last layer to arrays
                         y <- c(y, y.1)
                         x <- c(x, x.1)
                         l <- c(l, paste(layer, ",", p[length(p)]:1))

                         # initialize plot
                         graphics::par(mar = c(2,2,2,2))
                         graphics::plot(x = c(0,1), y = c(0,1), type="n", axes=FALSE, xlab = "", ylab = "")

                         # drawing lines
                         segments(x0 = x0, y0 = y0, x1 = x1, y1 = y1)

                         # drawing points
                         graphics::points(x, y, pch = 21, cex = 3, bg = "white")
                         graphics::text(x, y, labels = l, pos = 3, offset = 1)
                       }
                     ))


myn <- NeuralNet$new(c(1,3,3,1), function(x){x})

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

