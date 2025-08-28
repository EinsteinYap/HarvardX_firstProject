# MovieLens Rating Prediction Script
# Author: Yap Kah Yong
# Date: Dynamically set via Sys.Date()
# Purpose: Predict ratings and output RMSE on final_holdout_test

# Set seed for reproducibility
set.seed(8)

# Install required packages if not already installed
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")

# Load libraries
library(tidyverse)
library(caret)

##########################################################
# Step 1: Load and Split Data
##########################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
options(timeout = 120)
dl <- "ml-10M100K.zip"
ratings_file <- "ml-10M100K/ratings.dat"
movies_file <- "ml-10M100K/movies.dat"

# Download if not exists
if (!file.exists(dl)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl, 
                method = "curl", extra = "-k")
}

if (!file.exists(ratings_file)) unzip(dl, ratings_file)
if (!file.exists(movies_file)) unzip(dl, movies_file)

# Read data
ratings <- read_delim(ratings_file, delim = "::", 
                      col_names = c("userId", "movieId", "rating", "timestamp"),
                      show_col_types = FALSE)
movies <- read_delim(movies_file, delim = "::", 
                     col_names = c("movieId", "title", "genres"),
                     show_col_types = FALSE)

# Convert types
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Join
movielens <- left_join(ratings, movies, by = "movieId")

# Create edx and final_holdout_test
test_index <- createDataPartition(movielens$rating, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Clean up
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Step 2: Split edx for Tuning
##########################################################

train_idx <- createDataPartition(edx$rating, p = 0.8, list = FALSE)
train <- edx[train_idx, ]
valid <- edx[-train_idx, ]

##########################################################
# Step 3: Tune Lambda on Validation Set
##########################################################

lambdas <- c(50, 100, 200, 300)
results <- data.frame(lambda = numeric(), rmse = numeric())

for (lambda in lambdas) {
  mu <- mean(train$rating)
  
  user_bias <- train %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu) / (lambda + n()), .groups = 'drop')
  
  movie_bias <- train %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (lambda + n()), .groups = 'drop')
  
  pred_valid <- valid %>%
    left_join(user_bias, by = "userId") %>%
    left_join(movie_bias, by = "movieId") %>%
    mutate(
      b_u = ifelse(is.na(b_u), 0, b_u),
      b_i = ifelse(is.na(b_i), 0, b_i),
      pred = mu + b_u + b_i,
      pred = pmin(pmax(pred, 0.5), 5)
    )
  
  rmse <- RMSE(pred_valid$pred, pred_valid$rating)
  results <- add_row(results, lambda = lambda, rmse = rmse)
}

best_lambda <- results$lambda[which.min(results$rmse)]
cat("Best lambda:", best_lambda, "\n")
cat("Best rmse:", results$rmse[which.min(results$rmse)], "\n")

##########################################################
# Step 4: Final Model with Time Trend
##########################################################

mu_final <- mean(edx$rating, na.rm = TRUE)
cat("Global mean (mu_final):", round(mu_final, 3), "\n")
cat("Using best_lambda:", best_lambda, "\n")

# User and movie biases
user_bias_final <- edx %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu_final) / (best_lambda + n()), .groups = 'drop')

movie_bias_final <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu_final) / (best_lambda + n()), .groups = 'drop')

# Time trend with user-specific centering
min_time <- min(edx$timestamp, na.rm = TRUE)

edx <- edx %>%
  group_by(userId) %>%
  mutate(
    user_mean_rating = mean(rating),
    time_days = (timestamp - min_time) / (60*60*24),
    user_mean_time = mean(time_days)
  ) %>%
  ungroup()

user_time <- edx %>%
  group_by(userId) %>%
  summarise(
    n = n(),
    time_var = var(time_days),
    beta_t_raw = ifelse(n > 1 & time_var > 1,
                        cov(time_days - user_mean_time, rating - user_mean_rating,
                            use = "complete.obs") / var(time_days - user_mean_time),
                        0),
    beta_t = pmin(pmax(beta_t_raw, -0.001), 0.001),
    user_mean_time = mean(user_mean_time),
    .groups = 'drop'
  ) %>%
  select(userId, beta_t, user_mean_time)

# Final prediction
predictions <- final_holdout_test %>%
  select(userId, movieId, rating, timestamp) %>%
  left_join(user_bias_final, by = "userId") %>%
  left_join(movie_bias_final, by = "movieId") %>%
  left_join(user_time, by = "userId") %>%
  mutate(
    b_u = coalesce(b_u, 0),
    b_i = coalesce(b_i, 0),
    beta_t = coalesce(beta_t, 0),
    user_mean_time = coalesce(user_mean_time, mean(edx$time_days)),
    time_days = (timestamp - min_time) / (60*60*24),
    pred_raw = mu_final + b_u + b_i + beta_t * (time_days - user_mean_time),
    pred = pmin(pmax(pred_raw, 0.5), 5)
  )

# Diagnostics
cat("Raw prediction range:", range(predictions$pred_raw), "\n")
cat("Clamped to 0.5:", sum(predictions$pred == 0.5), "\n")
cat("Clamped to 5.0:", sum(predictions$pred == 5.0), "\n")
cat("Prediction mean:", round(mean(predictions$pred), 3), "\n")

# Final RMSE
rmse_final <- RMSE(predictions$pred, predictions$rating)
cat("Final Holdout Test RMSE:", round(rmse_final, 5), "\n")
