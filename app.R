library(shiny)
library(bslib)
library(glue)
library(keras)
library(tensorflow)
library(magrittr)
library(reticulate)

# --- CONFIGURATION ---
base_path <- "nuage"
img_folder <- base_path
model_file <- file.path(base_path, "modele_nuages.h5")
indices_file <- file.path(base_path, "indices_classes.rds")

# --- DESIGN IMMERSIF (CSS) ---
custom_css <- "
  body {
    background: linear-gradient(180deg, #e3f2fd 0%, #90caf9 100%);
    min-height: 100vh;
    font-family: 'Inter', sans-serif;
  }
  
  .navbar {
    background-color: rgba(15, 23, 42, 0.9) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding: 0.8rem 2rem;
  }

  .navbar-brand {
    color: #ffffff !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 3px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
  }

  .nav-link {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 500;
    margin-left: 15px;
    transition: all 0.3s ease;
  }
  
  .nav-link:hover { color: #4fc3f7 !important; }
  .nav-link.active { 
    color: #ffffff !important; 
    border-bottom: 2px solid #4fc3f7; 
  }

  .card {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    margin-bottom: 20px;
  }
  
  .card-header {
    background: transparent !important;
    font-weight: 700;
    color: #1e293b;
    text-transform: uppercase;
    font-size: 0.85rem;
  }

  .gallery-img {
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
  }
  
  .gallery-img:hover {
    transform: scale(1.05);
    box-shadow: 0 15px 30px rgba(0,0,0,0.2);
  }
"

# --- DONNÉES MÉTIER ---
cloud_data <- list(
  "Haute Altitude (> 6000m)" = list(
    "cirrus" = "Fins, légers, plumes. Beau temps.",
    "cirrostratus" = "Voile blanc. Halo possible.",
    "cirrocumulus" = "Petits flocons blancs en rangées."
  ),
  "Moyenne Altitude (2000-6000m)" = list(
    "altostratus" = "Couche grise. Pluie continue.",
    "altocumulus" = "Masses blanches ondulées. Orage possible."
  ),
  "Basse Altitude (< 2000m)" = list(
    "stratus" = "Brouillard d'altitude. Bruine.",
    "stratocumulus" = "Grosses masses sombres.",
    "nimbostratus" = "Épais et sombre. Pluie durable."
  ),
  "Développement Vertical" = list(
    "cumulus" = "Blancs et gonflés. Beau temps.",
    "cumulonimbus" = "Enclume géante. Orages et grêle."
  )
)

# --- UI ---
ui <- page_navbar(
  theme = bs_theme(version = 5, bootswatch = "flatly", primary = "#0288d1") %>% 
    bs_add_rules(custom_css),
  title = "NuageLab Explorer Pro",
  
  nav_panel("Découvrir",
            layout_sidebar(
              sidebar = sidebar(
                title = "Observation",
                selectInput("etage", "Étage atmosphérique :", choices = names(cloud_data)),
                uiOutput("ui_select_cloud"),
                hr(),
                card(card_header("Description"), textOutput("cloud_desc"))
              ),
              uiOutput("gallery_display")
            )
  ),
  
  nav_panel("Classification IA",
            layout_column_wrap(
              width = 1/2,
              card(
                card_header("Analyse de spécimen"),
                fileInput("upload", "Capture de ciel", accept = c("image/png", "image/jpeg")),
                actionButton("classify_btn", "Lancer l'analyse", class = "btn-primary w-100"),
                hr(),
                uiOutput("ui_action_move") 
              ),
              card(
                card_header("Résultat"),
                verbatimTextOutput("ia_result"),
                uiOutput("preview_upload")
              )
            )
  ),
  
  nav_panel("Entraînement",
            card(
              card_header("Apprentissage"),
              p("Dossier source :"),
              code(img_folder),
              hr(),
              actionButton("train_btn", "Mettre à jour l'IA", class = "btn-dark"),
              verbatimTextOutput("train_log")
            )
  )
)

# --- SERVER ---
server <- function(input, output, session) {
  
  addResourcePath("photos_nuages", img_folder)
  prediction_res <- reactiveVal(NULL)
  
  output$ui_select_cloud <- renderUI({
    req(input$etage)
    selectInput("cloud_type", "Type de nuage :", choices = names(cloud_data[[input$etage]]))
  })
  
  output$cloud_desc <- renderText({
    req(input$cloud_type)
    cloud_data[[input$etage]][[input$cloud_type]]
  })
  
  output$gallery_display <- renderUI({
    req(input$cloud_type)
    folder_path <- file.path(img_folder, input$cloud_type)
    img_files <- list.files(folder_path, pattern = "\\.(jpg|jpeg|png)$", ignore.case = TRUE)
    
    if(length(img_files) == 0) return(p("Aucune image disponible.", style="padding:20px;"))
    
    tags$div(
      style = "display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; padding: 20px;",
      lapply(img_files, function(f) {
        tags$img(src = file.path("photos_nuages", input$cloud_type, f), class = "gallery-img",
                 style = "width: 100%; height: 250px; object-fit: cover;")
      })
    )
  })
  
  # --- LOGIQUE IA (PREDICTION) ---
  observeEvent(input$classify_btn, {
    req(input$upload)
    if(!file.exists(model_file)) {
      showNotification("IA non entraînée.", type = "error")
      return()
    }
    
    withProgress(message = 'Calculs...', {
      tryCatch({
        model <- load_model_hdf5(model_file)
        class_indices <- readRDS(indices_file)
        
        img <- tensorflow::tf$keras$utils$load_img(input$upload$datapath, target_size = c(224L, 224L))
        x <- tensorflow::tf$keras$utils$img_to_array(img) / 255
        x <- array_reshape(x, c(1, 224, 224, 3))
        
        # Appel natif $predict
        preds <- model$predict(x)
        idx <- which.max(preds)
        nom_nuage <- names(class_indices)[which(class_indices == (idx - 1))]
        
        prediction_res(nom_nuage)
        output$ia_result <- renderPrint({
          cat("TYPE DÉTECTÉ :", toupper(nom_nuage), "\n")
          cat("CONFIANCE    :", round(preds[idx] * 100, 1), "%")
        })
      }, error = function(e) { showNotification(e$message, type = "error") })
    })
  })
  
  output$preview_upload <- renderUI({
    req(input$upload)
    tags$img(src = session$fileURL(input$upload$datapath), style = "width: 100%; border-radius: 15px; margin-top: 10px;")
  })
  
  output$ui_action_move <- renderUI({
    req(prediction_res())
    actionButton("move_btn", paste("Classer en", prediction_res()), class = "btn-success w-100")
  })
  
  observeEvent(input$move_btn, {
    target_dir <- file.path(img_folder, prediction_res())
    if(!dir.exists(target_dir)) dir.create(target_dir)
    file.copy(input$upload$datapath, file.path(target_dir, input$upload$name))
    showNotification("Image classée.", type = "message")
    prediction_res(NULL)
  })
  
  # --- LOGIQUE ENTRAÎNEMENT (API NATIVE $) ---
  observeEvent(input$train_btn, {
    withProgress(message = 'Apprentissage...', value = 0, {
      tryCatch({
        k_clear_session()
        
        train_gen <- flow_images_from_directory(
          img_folder, target_size = c(224, 224), 
          batch_size = as.integer(4), class_mode = "categorical"
        )
        
        num_classes <- length(train_gen$class_indices)
        if(num_classes < 2) stop("Besoin d'au moins 2 catégories.")
        
        base_model <- application_mobilenet_v2(weights = "imagenet", include_top = FALSE, input_shape = c(224, 224, 3))
        freeze_weights(base_model)
        
        model <- keras_model_sequential()
        model$add(base_model)
        model$add(layer_global_average_pooling_2d())
        model$add(layer_dense(units = 512, activation = "relu"))
        model$add(layer_dense(units = num_classes, activation = "softmax"))
        
        # APPELS NATIFS $ POUR ÉVITER LES ERREURS DE MÉTHODES
        model$compile(
          optimizer = "adam",
          loss = "categorical_crossentropy",
          metrics = list("accuracy")
        )
        
        model$fit(
          train_gen, 
          epochs = as.integer(5), 
          steps_per_epoch = as.integer(max(1, train_gen$n / 4))
        )
        
        model$save(model_file)
        saveRDS(train_gen$class_indices, indices_file)
        
        output$train_log <- renderPrint({ cat("Succès : IA à jour.") })
      }, error = function(e) { output$train_log <- renderPrint({ cat("Erreur :", e$message) }) })
    })
  })
}

shinyApp(ui, server)