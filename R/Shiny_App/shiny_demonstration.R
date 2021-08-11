library(shiny)
library(stringr)
ui <- tagList(shinyjs::useShinyjs(),navbarPage("NeuralNet",
  tabPanel("Configure Neural Net",
    fluidRow(
      column(6,
        radioButtons("actfunction", "Activation Function", c("ReLU"="ReLU", "Tangens Hyperbolicus" = "tanh", "Sigmoid" = "sigmoid")),
      ),
      column(6,
        radioButtons("outputactfunction", "Activation Function for Output", c("None"= "none","ReLU"="ReLU", "Tangens Hyperbolicus" = "tanh", "Sigmoid" = "sigmoid")),
      ),
    ),
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
        plotOutput("visplot")
      ),
      tabPanel("Learning Curve",
        plotOutput("curvePlot")
      ),
      tabPanel("Draw Yourself")
    )
  )
  )
  )
))

server <- function(input, output, session) {
  shinyjs::disable("trainNet")
  session$userData$train_data <- mnist$training_data
  session$userData$test_data <- mnist$test_data

  #New Network Creation
  observeEvent(input$newNet, {
    actfct <- input$actfunction
    outputfct <- NULL
    if(input$outputactfunction != "none") {
      outputfct <- input$outputactfunction
    }
    layers <- input$layers
    layers <- str_replace_all(layers, " ", "")
    layers <- str_split(layers, ",")[[1]]
    layers <- as.integer(layers)
    layers <- c(784, layers, 10)
    session$userData$nn <- NeuralNet$new(layers, actfct, outputfct)

    output$visplot <- renderPlot({
      session$userData$nn$plot()
    })
    shinyjs::enable("trainNet")
  })

  #Train network
  observeEvent(input$trainNet,{
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
    for(i in 1:(dataPerEpoch/500)) {
      print(i)
    }
  })
}


mnist_folder <- "./R/Shiny_App/mnist/"
#mnist <- MNIST$new(training_labels_fn = str_c(mnist_folder, "train-labels.idx1-ubyte"),
#                   training_images_fn = str_c(mnist_folder, "train-images.idx3-ubyte"),
#                   test_labels_fn = str_c(mnist_folder, "t10k-labels.idx1-ubyte"),
#                   test_images_fn = str_c(mnist_folder, "t10k-images.idx3-ubyte"))


app <- shinyApp(ui, server)

app

plot(sinpi)
