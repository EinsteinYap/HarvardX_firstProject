# MovieLens Rating Prediction Script
# Author: Yap Kah Yong
# Date: Dynamically set via Sys.Date()
# Purpose: Predict movie ratings and output RMSE on final_holdout_test
# This script follows all project rules:
# - final_holdout_test is NOT used for training or model selection
# - Only used ONCE at the end to compute RMSE

# Set seed for reproducibility
set.seed(8)

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
# Step 1: Load and Preprocess MovieLens 10M Dataset
##########################################################

# Dataset info: https://grouplens.org/datasets/movielens/10m/
options(timeout = 120)
dl <- "ml-10M100K.zip"

# Download if not exists, I have to add Method as Curl, else unable download from insecure link provided by edx
if (!file.exists(dl)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl,
                method = "curl", extra = "-k")
}

# Define file paths
ratings_file <- "ml-10M100K/ratings.dat"
movies_file <- "ml-10M100K/movies.dat"

# Extract files if not already extracted
if (!file.exists(ratings_file)) unzip(dl, ratings_file)
if (!file.exists(movies_file)) unzip(dl, movies_file)

# Read ratings data
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Read movies data
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Join ratings and movies
movielens <- left_join(ratings, movies, by = "movieId")

# Create edx (90%) and final_holdout_test (10%)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Ensure overlap: only keep users/movies in both edx and temp
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add back non-overlapping rows to edx
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Clean up intermediate objects
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Step 2: Split edx into Train (80%) and Valid (20%)
##########################################################

train_idx <- createDataPartition(edx$rating, p = 0.8, list = FALSE)
train <- edx[train_idx, ]
valid <- edx[-train_idx, ]

cat("Training set size:", nrow(train), "\n")
cat("Validation set size:", nrow(valid), "\n")

##########################################################
# Step 3: Tune Regularization Parameter (lambda)
##########################################################

# Test different lambda values (higher = more regularization)
lambdas <- c(50, 100, 200, 300)
results <- data.frame(lambda = numeric(), rmse = numeric())

for (lambda in lambdas) {
  mu <- mean(train$rating)  # Global mean from training set
  
  # Compute regularized user bias
  user_bias <- train %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu) / (lambda + n()), .groups = 'drop')
  
  # Compute regularized movie bias
  movie_bias <- train %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (lambda + n()), .groups = 'drop')
  
  # Predict on validation set
  pred_valid <- valid %>%
    left_join(user_bias, by = "userId") %>%
    left_join(movie_bias, by = "movieId") %>%
    mutate(
      b_u = ifelse(is.na(b_u), 0, b_u),
      b_i = ifelse(is.na(b_i), 0, b_i),
      pred = mu + b_u + b_i,
      pred = pmin(pmax(pred, 0.5), 5)  # Clamp to valid range
    )
  
  rmse <- RMSE(pred_valid$pred, pred_valid$rating)
  results <- add_row(results, lambda = lambda, rmse = rmse)
}

# Select best lambda (lowest validation RMSE)
best_lambda <- results$lambda[which.min(results$rmse)]
cat("Best lambda:", best_lambda, "\n")
cat("Best validation RMSE:", round(results$rmse[which.min(results$rmse)], 5), "\n")

# Plot lambda vs RMSE
plot(results$lambda, results$rmse,
     type = "b", pch = 19, col = "blue",
     xlab = "Regularization Parameter (lambda)",
     ylab = "Validation RMSE",
     main = "Hyperparameter Tuning: lambda vs. RMSE")
grid()

##########################################################
# Step 4: Final Model with Time Trend
##########################################################

# Global average rating
mu_final <- mean(edx$rating, na.rm = TRUE)
cat("Global mean (mu_final):", round(mu_final, 3), "\n")
cat("Using best_lambda:", best_lambda, "\n")

# Final user and movie biases
user_bias_final <- edx %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu_final) / (best_lambda + n()), .groups = 'drop')

movie_bias_final <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu_final) / (best_lambda + n()), .groups = 'drop')

# Add time trend with user-specific centering
min_time <- min(edx$timestamp, na.rm = TRUE)

edx <- edx %>%
  group_by(userId) %>%
  mutate(
    user_mean_rating = mean(rating),
    time_days = (timestamp - min_time) / (60*60*24),        # seconds → days
    user_mean_time = mean(time_days)                        # user's average time
  ) %>%
  ungroup()

# Fit user-specific time trend (rating drift over time)
user_time <- edx %>%
  group_by(userId) %>%
  summarise(
    n = n(),
    time_var = var(time_days),
    # Only fit if user has variation
    beta_t_raw = ifelse(n > 1 & time_var > 1,
                        cov(time_days - user_mean_time, rating - user_mean_rating, use = "complete.obs") / var(time_days - user_mean_time),
                        0),
    # Clip to prevent explosion
    beta_t = pmin(pmax(beta_t_raw, -0.001), 0.001),
    user_mean_time = mean(user_mean_time),
    .groups = 'drop'
  ) %>%
  select(userId, beta_t, user_mean_time)

# Final prediction on final_holdout_test
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

##########################################################
# Step 5: Visualizations
##########################################################

# --- Distribution of User Biases ---
hist(user_bias_final$b_u, breaks = 50, col = "skyblue", border = "white",
     main = "Distribution of User Biases (b_u)",
     xlab = "User Bias (deviation from global mean)")
abline(v = 0, col = "red", lwd = 2)

# --- Distribution of Movie Biases ---
hist(movie_bias_final$b_i, breaks = 50, col = "lightgreen", border = "white",
     main = "Distribution of Movie Biases (b_i)",
     xlab = "Movie Bias (deviation from global mean)")
abline(v = 0, col = "red", lwd = 2)

# --- Boxplot: Predictions by Actual Rating ---
boxplot(pred ~ rating, data = predictions,
        main = "Distribution of Predictions by Actual Rating",
        xlab = "Actual Rating", ylab = "Predicted Rating",
        col = "lightblue", boxwex = 0.5)
abline(h = seq(0.5, 5.0, by = 0.5), col = "gray", lty = 2)
grid()

# Insight: Model performs best for ratings between 3.0 and 4.5

cat("Final Holdout Test RMSE:", round(rmse_final, 5), "\n")