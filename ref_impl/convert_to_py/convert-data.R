# load corresponding .Rda file into memory first then run conversion one by one

#store.id=234212
#store.id=236117
#store.id=649405
store.id=657979

if (!file.exists("data/raw")) {
  dir.create("data/raw", recursive = TRUE)
}

# Iterate over each category in SKU.data
for (i in seq_along(SKU.data)) {
  category <- names(SKU.data)[i]
  category_data <- SKU.data[[i]]
  
  # Generate file name based on store ID and category
  file_name <- paste0("data/raw/SKU_data_store_", store.id, "_", category, ".csv")
  
  write.csv(category_data, file = file_name, row.names = FALSE)
}