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

