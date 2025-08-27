# MovieLens Rating Prediction Script
# Author: Yap Kah Yong
# Purpose: Predict movie ratings and output RMSE on final_holdout_test
# Model: Regularized baseline (global mean + user bias + item bias)
# This script follows all project rules:
# - final_holdout_test is NOT used for training or model selection
# - Only used ONCE at the end to compute RMSE

# Set seed for reproducibility
set.seed(5)

# Install required packages if not already installed
if (!require(tidyverse)) {
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
}
if (!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
}

# Load libraries
library(tidyverse)
library(caret)

##########################################################
# Step 1: Download and Load MovieLens 10M Dataset
##########################################################

# Set longer timeout for download
options(timeout = 120)

# File paths
dl <- "ml-10M100K.zip"
ratings_file <- "ml-10M100K/ratings.dat"
movies_file <- "ml-10M100K/movies.dat"

# Download dataset if not already present
if (!file.exists(dl)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl, 
                method = "curl",
                extra = "-k" )
}
# Extract required files
if (!file.exists(ratings_file)) {
  unzip(dl, ratings_file)
}
if (!file.exists(movies_file)) {
  unzip(dl, movies_file)
}

# Read ratings and movies data
ratings <- read_delim(ratings_file, delim = "::",
                      col_names = c("userId", "movieId", "rating", "timestamp"),
                      show_col_types = FALSE)
movies <- read_delim(movies_file, delim = "::",
                     col_names = c("movieId", "title", "genres"),
                     show_col_types = FALSE)

# Convert data types
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Join ratings with movie info
movielens <- left_join(ratings, movies, by = "movieId")

##########################################################
# Step 2: Create edx and final_holdout_test sets
##########################################################

# Set seed for reproducible split
set.seed(5)
test_index <- createDataPartition(movielens$rating, p = 0.1, list = FALSE)

# Split into edx (90%) and temp (10%)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Ensure users and movies in final_holdout_test also appear in edx
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add back any rows that were removed (to keep all data in edx)
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Clean up intermediate objects
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Step 3: Train Final Model on Full edx
##########################################################

# Global average rating
mu <- mean(edx$rating)

# Regularization parameter (tuned separately)
lambda <- 20

# User bias: b_u = sum(rating - mu) / (lambda + n_ratings_by_user)
user_bias <- edx %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu) / (lambda + n()), .groups = 'drop')

# Movie bias: b_i = sum(rating - mu) / (lambda + n_ratings_for_movie)
movie_bias <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu) / (lambda + n()), .groups = 'drop')

##########################################################
# Step 4: Predict on final_holdout_test
##########################################################

# Join user and movie biases to final_holdout_test
predictions <- final_holdout_test %>%
  left_join(user_bias, by = "userId") %>%
  left_join(movie_bias, by = "movieId") %>%
  mutate(
    # Handle cold-start: users/movies not in edx get zero bias
    b_u = ifelse(is.na(b_u), 0, b_u),
    b_i = ifelse(is.na(b_i), 0, b_i),
    # Predict rating
    pred = mu + b_u + b_i,
    # Clamp prediction to valid range [0.5, 5]
    pred = pmin(pmax(pred, 0.5), 5)
  )

# Calculate RMSE between predicted and actual ratings
rmse_final <- RMSE(predictions$pred, predictions$rating)

# Output final RMSE (this will be used for grading)
cat("Final Holdout Test RMSE:", round(rmse_final, 5), "\n")
