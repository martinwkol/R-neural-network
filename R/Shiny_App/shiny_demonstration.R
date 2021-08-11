library(shiny)
library(stringr)
ui <- tagList(shinyjs::useShinyjs(),navbarPage("NeuralNet",
  tabPanel("Configure Neural Net",
    radioButtons("actfunction", "Activation Function", c("ReLU"="ReLU", "Tangens Hyperbolicus" = "tanh", "Sigmoid" = "sigmoid")),
    tags$b("Hidden Layers"),
    p("A comma seperated list with each value denoting the number of neurons for that layer (This won't be tested!!)"),
    textInput("layers", NULL, value="100, 200"),
    tags$b("Pressing this button will delete previous Network and Training!!"),
    actionButton("newNet", "Create new Network")
  ),
  tabPanel("Train and Visualize", id = "trainTab",
  sidebarLayout(
  sidebarPanel(
    h3("Optimizer Configuration"),
    radioButtons("optalg", "Optimazation Algorithm", c("Stochastik Gradient Ascent"="sga", "Momentum"="mom", "Nesterov Accelarated Gradient" = "nesag")),
    textInput("learningRate", "Learning Rate", "0.0005"),
    textInput("regrate", "Regularization Rate", "0.0"),
    HTML("<b>Momentum</b>"),
    p("Only used for Momentum and Nesterov"),
    textInput("momentum", NULL, "0.9"),
    hr(style="border-top: 1px solid #000000;"),
    h3("Trainer Configuration"),
    textInput("epochs", "Number of Epochs", value=1),
    sliderInput("dataPerEpoch", "Data Points per Epoch", min = 500, max = 60000, value = 30000, step = 500),
    hr(style="border-top: 1px solid #505050;"),
    actionButton("trainNet", "Train!")
  ),
  mainPanel(
    tabsetPanel(
      tabPanel("Net Visualization",
        #
        #For plot height
        tags$head(tags$script('
                                var dimension = [0, 0];
                                $(document).on("shiny:connected", function(e) {
                                    dimension[0] = window.innerWidth;
                                    dimension[1] = window.innerHeight;
                                    Shiny.onInputChange("dimension", dimension);
                                });
                                $(window).resize(function(e) {
                                    dimension[0] = window.innerWidth;
                                    dimension[1] = window.innerHeight;
                                    Shiny.onInputChange("dimension", dimension);
                                });
                            ')),
        #
        #
        plotOutput("visplot")
      ),
      tabPanel("Learning Curve",
        plotOutput("curvePlot"))
      ,
      tabPanel("Draw Yourself",
        HTML("<b>The Network predicted:</b>"),
        textOutput("drawPrediction"),
        actionButton("resetDraw", "reset!"),
        plotOutput("drawPlot",
                  hover=hoverOpts(id="drawHower", delay=100, delayType="throttle", clip=TRUE, nullOutside = TRUE),
                  click="drawClick")
      )
    )
  )
  )
  )
))

server <- function(input, output, session) {
  shinyjs::disable("trainNet")
  session$userData$train_data <- mnist$training_data
  session$userData$test_data <- mnist$test_data

  #Draw Plot
  drawVals = reactiveValues(x=NULL, y=NULL)
  drawing = reactiveVal(FALSE)

  observeEvent(input$resetDraw, {
    drawVals$x <- NULL
    drawVals$y <- NULL
  })

  observeEvent(input$drawClick, {
    drawing(!drawing())
    if(!drawing()) {
      drawVals$x <- c(drawVals$x, NA)
      drawVals$y <- c(drawVals$y, NA)
      #Calc
      x <- drawVals$x
      y <- drawVals$y
      x <- x[!is.na(x)]
      y <- y[!is.na(y)]
      x <- round(x)
      y <- round(y)

      flat <- NULL
      for(j in 28:1) {
        for(i in 1:28) {
          ind <- which(x == i)
          flat <- c(flat, any(y[ind] == j))
        }
      }
      flat <- as.integer(flat)
      for(i in 1:length(flat)) {
        if(i > 28 && flat[i] == 1) {
          flat[i-28] <- max(flat[i-28], 0.5)
        }
        if(i <= 756 && flat[i] == 1) {
          flat[i+28] <- max(flat[i+28], 0.5)
        }
        if(i > 1 && flat[i] == 1) {
          flat[i-1] <- max(flat[i-1], 0.5)
        }
        if(i < 784 && flat[i] == 1) {
          flat[i+1] <- max(flat[i+1], 0.5)
        }
      }
      output$drawPrediction <- renderText({session$userData$nn$calculate(flat)[[3]] - 1})
    }
  })

  observeEvent(input$drawHower, {
    if(drawing()) {
      drawVals$x <- c(drawVals$x, input$drawHower$x)
      drawVals$y <- c(drawVals$y, input$drawHower$y)
    }
  })

  output$drawPlot = renderPlot({
    plot(x=drawVals$x, y=drawVals$y, xlim=c(0,28), ylim=c(0,28), type="l", lwd=14, xlab="", ylab="", xaxt="n", yaxt="n", mar=c(2,2,2,2)+0.1)
  }, width=560, height=560)

  #New Network Creation
  observeEvent(input$newNet, {
    actfct <- input$actfunction
    layers <- input$layers
    layers <- str_replace_all(layers, " ", "")
    layers <- str_split(layers, ",")[[1]]
    layers <- as.integer(layers)
    layers <- c(784, layers, 10)
    session$userData$nn <- NeuralNet$new(layers, actfct)

    output$visplot <- renderPlot({
      session$userData$nn$plot(labels=FALSE)
    }, height=input$dimension[2] - 120)


    output$curvePlot <- renderPlot({
      plot(x=NULL, y=NULL, xlim=c(0,60000), ylim=c(0,1), xlab="Number of Datapoints trained", ylab="Accuracy")
    }, height=input$dimension[2] - 120)
    session$userData$x <- NULL
    session$userData$y <- NULL
    session$userData$lastx <- 0


    shinyjs::enable("trainNet")
  })

  #Train network
  observeEvent(input$trainNet,{
    shinyjs::disable("trainNet")
    nn <- session$userData$nn
    learningRate <- as.numeric(input$learningRate)
    regrate <- as.numeric(input$regrate)
    momentum <- as.numeric(input$momentum)
    optimizer <- NULL
    if(input$optalg == "sga") {
      optimizer <- OptimizerSGD$new(learningRate, regrate)
    } else if(input$optalg == "mom") {
      optimizer <- OptimizerMomentum$new(learningRate, regrate, momentum)
    } else {
      optimizer <- OptimizerNesterovAG$new(learningRate, regrate, momentum)
    }
    trainer <- Trainer$new(nn, optimizer, session$userData$train_data, session$userData$test_data)

    epochs <- input$epochs
    dataPerEpoch <- input$dataPerEpoch
    if(session$userData$lastx == 0) {
      session$userData$x <- 0
      session$userData$y <- trainer$test(1000)
    }
    for(i in 1:(dataPerEpoch/500)) {
      trainer$train(1, 500)
      trainer$swapWithBestNeuralnet()
      session$userData$lastx <- session$userData$lastx + 500
      session$userData$x <- c(session$userData$x, session$userData$lastx)
      session$userData$y <- c(session$userData$y, trainer$test(1000))
    }
    output$curvePlot <- renderPlot({
      plot(x=session$userData$x, y=session$userData$y, type="l", xlim=c(0,max(60000, session$userData$lastx)), ylim=c(0,1), xlab="Number of Datapoints trained", ylab="Accuracy")
      abline(h = max(session$userData$y), col="red", lty="dashed")
    }, height=input$dimension[2] - 120)
    session$userData$nn <- trainer$getNeuralnet()


    output$visplot <- renderPlot({
      session$userData$nn$plot(labels=FALSE)
    }, height=input$dimension[2] - 120)

    shinyjs::enable("trainNet")
  })
}


mnist_folder <- "./R/Shiny_App/mnist/"
mnist <- MNIST$new(training_labels_fn = str_c(mnist_folder, "train-labels.idx1-ubyte"),
                   training_images_fn = str_c(mnist_folder, "train-images.idx3-ubyte"),
                   test_labels_fn = str_c(mnist_folder, "t10k-labels.idx1-ubyte"),
                   test_images_fn = str_c(mnist_folder, "t10k-images.idx3-ubyte"))


app <- shinyApp(ui, server)

app


