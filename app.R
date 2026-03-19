library(shiny)
library(keras3)
library(magick)

if (!reticulate::py_available()) {
  keras3::install_keras()
}

# --- 1. SETUP & MODEL LOADING ---

# Load the model
model <- load_model("models/fruits_360_cnn.h5")

# Load labels from the text file we generated
# This ensures we have exactly 253 classes in alphabetical order
if (!file.exists("models/labels.txt")) {
  stop("labels.txt not found! Please generate it from your Training folder first.")
}
fruit_labels <- readLines("models/labels.txt")

# Safety check
if (length(fruit_labels) != 253) {
  warning("Label count (", length(fruit_labels), ") does not match model output (253)!")
}

# --- 2. USER INTERFACE ---

ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "flatly"),
  titlePanel("Fruits-360 Classifier (Keras 3)"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("upload", "Upload Fruit Image", accept = c('image/png', 'image/jpeg')),
      actionButton("predict", "Identify Fruit", class = "btn-primary", style = "width: 100%"),
      hr(),
      helpText("Note: This model expects 100x100 RGB images.")
    ),
    
    mainPanel(
      column(12, align = "center",
             imageOutput("selected_img", height = "auto"),
             br(),
             wellPanel(
               h3(textOutput("prediction_text")),
               h4(textOutput("confidence_text"))
             )
      ),
      hr(),
      h4("Top 5 Probabilities"),
      tableOutput("prob_table")
    )
  )
)

# --- 3. SERVER LOGIC ---

server <- function(input, output) {
  
  # Display the uploaded image
  output$selected_img <- renderImage({
    req(input$upload)
    list(src = input$upload$datapath, width = 300, height = 300, alt = "Uploaded Image")
  }, deleteFile = FALSE)
  
  # Perform prediction
  observeEvent(input$predict, {
    req(input$upload)
    
    # Preprocessing
    # 1. Load image and resize to 100x100 (matching your Conv2D input)
    img <- image_load(input$upload$datapath, target_size = c(100, 100))
    img_array <- image_to_array(img)
    
    # 2. Reshape to (1, 100, 100, 3) and normalize
    img_array <- array_reshape(img_array, c(1, 100, 100, 3))
   img_array <- img_array / 255 
    
    # 3. Predict
    preds <- model %>% predict(img_array)
    
    # Process Results
    top_idx <- which.max(preds)
    top_fruit <- fruit_labels[top_idx]
    confidence <- round(max(preds) * 100, 2)
    
    # Output: Main prediction text
    output$prediction_text <- renderText({
      paste0("Result: ", top_fruit)
    })
    
    output$confidence_text <- renderText({
      paste0("Confidence: ", confidence, "%")
    })
    
    # Output: Probability Table (Top 5)
    output$prob_table <- renderTable({
      results_df <- data.frame(
        Fruit = fruit_labels,
        Probability = as.vector(preds)
      )
      # Sort by probability and take top 5
      results_df <- results_df[order(-results_df$Probability), ]
      head(results_df, 5)
    }, digits = 4)
  })
}

# --- 4. RUN APP ---
shinyApp(ui = ui, server = server)
