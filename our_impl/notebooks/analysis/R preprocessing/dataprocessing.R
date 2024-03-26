# Library -------------------------------------------------------------
library(data.table)
library(dplyr)

# Path ----------------------------------------------------------------

file_data_1 <- "data/SKU data store 234212.Rda"
file_data_2 <- "data/SKU data store 236117.Rda"
file_data_3 <- "data/SKU data store 649405.Rda"
file_data_4 <- "data/SKU data store 657979.Rda"

makeVlist <- function(dt) {
  labels <- sapply(dt, function(x) attr(x, "label"))
  tibble(name = names(labels),
         label = labels)
}



# Load Data ----------------------------------------------------------

time <- as.data.table(fread("../../assets/time.csv"))
time_id <- as.data.table(time$`IRI Week`)

# Get variable names
fn_get_var <- function(file_data) {
  
  load(file_data)
  milk_data <- as.data.table(SKU.data$milk)
  milk_data <- cbind(milk_data, time_id[1:nrow(milk_data), ])
  
  # Other categories (if needed)
  # beer_data <- as.data.table(SKU.data$beer)
  # beer_data <- cbind(beer_data, time_id[1:nrow(beer_data), ])
  # 
  # mayo_data <- as.data.table(SKU.data$mayo)
  # mayo_data <- cbind(mayo_data, time_id[1:nrow(mayo_data), ])
  # 
  # yogurt_data <- as.data.table(SKU.data$yogurt)
  # yogurt_data <- cbind(yogurt_data, time_id[1:nrow(yogurt_data), ])
  
  # Rename column names
  varname <- as.data.table(makeVlist(milk_data))
  varname[, label := as.character(label)]
  
  return(varname)
}

varname_1 <- fn_get_var(file_data_1)
varname_2 <- fn_get_var(file_data_2)
varname_3 <- fn_get_var(file_data_3)
varname_4 <- fn_get_var(file_data_4)

fwrite(varname_1, "data/234212 milk var.csv")
fwrite(varname_2, "data/236117 milk var.csv")
fwrite(varname_3, "data/649405 milk var.csv")
fwrite(varname_4, "data/657979 milk var.csv")

# Modify column names to standardise melting
varname_new_1 <- fread("data/234212 milk var new.csv")
varname_new_2 <- fread("data/236117 milk var new.csv")
varname_new_3 <- fread("data/649405 milk var new.csv")
varname_new_4 <- fread("data/657979 milk var new.csv")


# Convert wide to long format
fn_melt <- function(file_data, varname_new) {
  load(file_data)
  
  milk_data <- as.data.table(SKU.data$milk)
  milk_data <- cbind(milk_data, time_id[1:nrow(milk_data), ])
  
  # Other categories (if needed)
  # beer_data <- as.data.table(SKU.data$beer)
  # beer_data <- cbind(beer_data, time_id[1:nrow(beer_data), ])
  # 
  # mayo_data <- as.data.table(SKU.data$mayo)
  # mayo_data <- cbind(mayo_data, time_id[1:nrow(mayo_data), ])
  # 
  # yogurt_data <- as.data.table(SKU.data$yogurt)
  # yogurt_data <- cbind(yogurt_data, time_id[1:nrow(yogurt_data), ])
  
  # Rename for melting
  setnames(milk_data, varname_new$name, varname_new$new_name)
  
  data_long <- melt(
    milk_data,
    measure.vars = patterns(
      "^Sales_",
      "^Discount_",
      "^Price_",
      "^Display1_",
      "^Display2_",
      "^Feature1_",
      "^Feature2_",
      "^Feature3_",
      "^Feature4_"
    ),
    value.name = c(
      "Sales",
      "Discount",
      "Price",
      "Display1",
      "Display2",
      "Feature1",
      "Feature2",
      "Feature3",
      "Feature4"
    ),
    id.vars = c("Time_ID")
  )
  return(data_long)
}

# Get mapping of SKU
fn_sku <- function(data_path, string) {
  load(data_path)
  
  milk_data <- as.data.table(SKU.data$milk)
  milk_data <- cbind(milk_data, time_id[1:nrow(milk_data), ])
  
  # Other categories (if needed)
  # beer_data <- as.data.table(SKU.data$beer)
  # beer_data <- cbind(beer_data, time_id[1:nrow(beer_data), ])
  # 
  # mayo_data <- as.data.table(SKU.data$mayo)
  # mayo_data <- cbind(mayo_data, time_id[1:nrow(mayo_data), ])
  # 
  # yogurt_data <- as.data.table(SKU.data$yogurt)
  # yogurt_data <- cbind(yogurt_data, time_id[1:nrow(yogurt_data), ])
  
  sku_list <- grep(eval(string), names(milk_data), value = TRUE)
  sku_list <- gsub(eval(string), "", sku_list)
  sku_map <- data.frame(SKU = sku_list)
  sku_map$variable <- 1:nrow(sku_map)
  sku_map$variable <- as.factor(sku_map$variable)
  
  return(as.data.table(sku_map))
}


dt_1 <- fn_melt(file_data_1, varname_new_1)
dt_2 <- fn_melt(file_data_2, varname_new_2)
dt_3 <- fn_melt(file_data_3, varname_new_3)
dt_4 <- fn_melt(file_data_4, varname_new_4)

sku_map_dt_1 <- fn_sku(file_data_1, 'milk_SS')
sku_map_dt_2 <- fn_sku(file_data_2, '-UNITS')
sku_map_dt_3 <- fn_sku(file_data_3, '-UNITS')
sku_map_dt_4 <- fn_sku(file_data_4, '-UNITS')

sku_map_dt_2$SKU <- gsub('236117_', "", sku_map_dt_2$SKU)
sku_map_dt_3$SKU <- gsub('649405_', "", sku_map_dt_3$SKU)
sku_map_dt_4$SKU <- gsub('657979_', "", sku_map_dt_4$SKU)

dt_1 <- merge(dt_1, sku_map_dt_1, key = 'variable')
dt_2 <- merge(dt_2, sku_map_dt_2, key = 'variable')
dt_3 <- merge(dt_3, sku_map_dt_3, key = 'variable')
dt_4 <- merge(dt_4, sku_map_dt_4, key = 'variable')

dt_1[, Store_ID := 234212]
dt_2[, Store_ID := 236117]
dt_3[, Store_ID := 649405]
dt_4[, Store_ID := 657979]


# Combine all files ---------------------------------------------------------
combined_dt <- rbind(dt_1, dt_2, dt_3, dt_4)

# Add year
time_year <- time[, c('IRI Week', 'Year')]
combined_dt <- merge(combined_dt, time_year, all.x = TRUE, by.x = 'Time_ID', by.y = 'IRI Week')

# Remove variable
combined_dt$variable <- NULL

fwrite(combined_dt, "../../assets/combined_milk_final.csv")
