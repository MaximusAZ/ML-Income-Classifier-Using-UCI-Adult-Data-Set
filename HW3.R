##################################################
# ECON 418-518 Homework 3
# Danny Watkins
# The University of Arizona
# maximus@arizona.edu 
# 30 November 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# install packages
install.packages("caret")
install.packages("glmnet")
install.packages("randomForest")

library(readr)
ECON_418_518_HW3_Data <- read_csv("ECON_418-518_HW3_Data.csv")

# Set seed for reproducibility
set.seed(418518)

# Use the already loaded dataset
data <- ECON_418_518_HW3_Data

# (i)
# Drop the specified columns
columns_to_drop <- c("fnlwgt", "occupation", "relationship", "capital-gain", "capital-loss", "educational-num")
data <- data[, !(names(data) %in% columns_to_drop)]

# (ii) Data cleaning
# First convert and dashes to underscores for column headers bc R can't read dashes for names
names(data) <- gsub("-", "_", names(data))
# (a) Convert "income" column to binary indicator
data$income <- ifelse(data$income == ">50K", 1, 0)

# (b) Convert "race" column to binary indicator
data$race <- ifelse(data$race == "White", 1, 0)

# (c) Convert "gender" column to binary indicator
data$gender <- ifelse(data$gender == "Male", 1, 0)

# (d) Convert "workclass" column to binary indicator
data$workclass <- ifelse(data$workclass == "Private", 1, 0)

# (e) Convert "native_country" column to binary indicator
data$native_country <- ifelse(data$native_country == "United-States", 1, 0)

# (f) Convert "marital_status" column to binary indicator
data$marital_status <- ifelse(data$marital_status == "Married-civ-spouse", 1, 0)

# (g) Convert "education" column to binary indicator
data$education <- ifelse(data$education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)

# (h) Create an "age_sq" variable
data$age_sq <- data$age^2

# (i) Standardize "age", "age_sq", and "hours_per_week" variables
standardize <- function(x) {
  (x - mean(x)) / sd(x)
}

data$age <- standardize(data$age)
data$age_sq <- standardize(data$age_sq)
data$hours_per_week <- standardize(data$hours_per_week)


# (iii) Data Analysis
# (a) Proportion of individuals with income greater than $50,000
proportion_income_gt_50k <- mean(data$income == 1)
cat("\nProportion of individuals with income greater than $50,000:\n", proportion_income_gt_50k, "\n")

# (b) Proportion of individuals in the private sector
proportion_private_sector <- mean(data$workclass == 1)
cat("\nProportion of individuals in the private sector:\n", proportion_private_sector, "\n")

# (c) Proportion of married individuals
proportion_married <- mean(data$marital_status == 1)
cat("\nProportion of married individuals:\n", proportion_married, "\n")

# (d) Proportion of females
proportion_females <- mean(data$gender == 0)
cat("\nProportion of females:\n", proportion_females, "\n")

# (e) Total number of NAs in the dataset
# I believe "?" means "NA" so i'm replacing each "?" with an "NA"
data[data == "?"] <- NA
total_nas <- sum(is.na(data))
cat("\nTotal number of NAs in the dataset:\n", total_nas, "\n")

# (f) Convert the "income" variable to a factor data type
data$income <- as.factor(data$income)
cat("\n'Income' variable converted to a factor data type.\n")
print(summary(data$income))



# (iv) Splitting the data into training and testing sets
# (a) Find the last training set observation
set.seed(418518)  # Ensure reproducibility for consistent splits
train_size <- floor(nrow(data) * 0.70)
cat("Last training set observation index:\n", train_size, "\n")

# (b) Create the training data set
training_data <- data[1:train_size, ]

# (c) Create the testing data set
testing_data <- data[(train_size + 1):nrow(data), ]



# (v) ML
# Load required packages
library(caret)
library(glmnet)

# Define the grid for lambda
lambda_grid <- 10^seq(5, -2, length = 50)

# Prepare data
x_train <- model.matrix(income ~ ., data = training_data)[, -1]  # Remove intercept
y_train <- training_data$income

# Define the training control
train_control <- trainControl(method = "cv", number = 10)

# Train the lasso regression model
lasso_model <- train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid),
  trControl = train_control
)

# Output best lambda
cat("Best lambda for Lasso Regression:", lasso_model$bestTune$lambda, "\n")
cat("Lasso Model Classification Accuracy:", max(lasso_model$results$Accuracy), "\n")

coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)

# Subset training data with non-zero coefficient variables
significant_variables <- c("age", "education", "marital_status", "hours_per_week")
reduced_x_train <- x_train[, significant_variables, drop = FALSE]

# Train reduced Lasso model
lasso_reduced_model <- train(
  x = reduced_x_train,
  y = y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid),
  trControl = train_control
)

# Train reduced Ridge model
ridge_reduced_model <- train(
  x = reduced_x_train,
  y = y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid),
  trControl = train_control
)

# Extract classification accuracies
lasso_reduced_accuracy <- max(lasso_reduced_model$results$Accuracy)
ridge_reduced_accuracy <- max(ridge_reduced_model$results$Accuracy)

cat("Best Reduced Lasso Accuracy:", lasso_reduced_accuracy, "\n")
cat("Best Reduced Ridge Accuracy:", ridge_reduced_accuracy, "\n")



# (vi) Random Forest Model Estimation
library(randomForest)

# Define the parameter grid
rf_grid <- expand.grid(mtry = c(2, 5, 9))  # Number of random features to try

# Set up train control
train_control <- trainControl(method = "cv", number = 5)

# Function to train random forest models with a given number of trees
train_rf_model <- function(ntree) {
  train(
    income ~ .,  # Formula with all explanatory variables
    data = training_data,
    method = "rf",
    tuneGrid = rf_grid,
    trControl = train_control,
    ntree = ntree
  )
}

# Train random forest models
cat("Training Random Forest with 100 trees...\n")
rf_model_100 <- train_rf_model(ntree = 100)

cat("Training Random Forest with 200 trees...\n")
rf_model_200 <- train_rf_model(ntree = 200)

cat("Training Random Forest with 300 trees...\n")
rf_model_300 <- train_rf_model(ntree = 300)

# Output results
cat("\nBest model for 100 trees:\n")
print(rf_model_100$bestTune)
cat("\nBest model for 200 trees:\n")
print(rf_model_200$bestTune)
cat("\nBest model for 300 trees:\n")
print(rf_model_300$bestTune)

# Extract highest accuracy
cat("\nHighest classification accuracy for 100 trees:\n", max(rf_model_100$results$Accuracy), "\n")
cat("\nHighest classification accuracy for 200 trees:\n", max(rf_model_200$results$Accuracy), "\n")
cat("\nHighest classification accuracy for 300 trees:\n", max(rf_model_300$results$Accuracy), "\n")

# Compare best random forest model accuracy with lasso/ridge model
best_rf_accuracy <- max(rf_model_300$results$Accuracy)  
cat("\nBest Random Forest Model Accuracy (300 trees):", best_rf_accuracy, "\n")

# Assuming lasso_model was trained in Part (v)
cat("Best Lasso Model Accuracy:", max(lasso_model$results$Accuracy), "\n")

# (e) Create a confusion matrix
cat("\nGenerating Confusion Matrix for the best Random Forest model...\n")
best_rf_model <- rf_model_300  
predictions <- predict(best_rf_model, training_data)

# Generate the confusion matrix
conf_matrix <- confusionMatrix(predictions, training_data$income)
cat("\nConfusion Matrix:\n")
print(conf_matrix)

# Extract false positives and false negatives
false_positives <- conf_matrix$table[2, 1]  # Predicted 1, actual 0
false_negatives <- conf_matrix$table[1, 2]  # Predicted 0, actual 1

cat("\nNumber of False Positives:", false_positives, "\n")
cat("Number of False Negatives:", false_negatives, "\n")


# Use the Random Forest model with 300 trees as the best model
best_model <- rf_model_300

# Make predictions on the testing set
testing_predictions <- predict(best_model, testing_data)

# Calculate classification accuracy on the testing set
testing_accuracy <- mean(testing_predictions == testing_data$income)
cat("\nClassification accuracy on the testing set:\n", testing_accuracy, "\n")

# Generate a confusion matrix for further evaluation
conf_matrix_testing <- confusionMatrix(testing_predictions, testing_data$income)
cat("\nConfusion Matrix on the testing set:\n")
print(conf_matrix_testing)

