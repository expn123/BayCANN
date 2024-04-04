# load libraries ========
library(keras)
library(rstan)
library(reshape2)
library(tidyverse)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# load baycann functions =======
source("baycann_functions.R")

# ==================
# Input parameters
n_iter <- 10000
n_hidden_nodes <- 100
n_hidden_layers <- 2 
n_epochs <- 10000
verbose <- 0
n_batch_size <- 2000
n_chains <- 4
set.seed(1234)

# load the training and test datasets for the simulations =========
cal_res_MA <- readRDS("data/cal_res_MA/cal_res.RDS")
index_train <- sample(x = 1:nrow(cal_res_MA), size = nrow(cal_res_MA)*0.8, replace = FALSE)
samp_train <- cal_res_MA[index_train, 32:50]
samp_test  <- cal_res_MA[-index_train, 32:50]
out_train  <- cal_res_MA[index_train, 3:30]
out_test  <- cal_res_MA[-index_train, 3:30]


prepared_data <- prepare_data(xtrain = samp_train,
                              ytrain = out_train,
                              xtest  = samp_test,
                              ytest  = out_test)

list2env(prepared_data, envir = .GlobalEnv)


# load the targets and their se ==========


# ============== TensorFlow Keras ANN Section ========================

model <- keras_model_sequential() 
mdl_string <- paste("model %>% layer_dense(units = n_hidden_nodes, activation = 'tanh', input_shape = n_inputs) %>%", 
                    paste(rep(x = "layer_dense(units = n_hidden_nodes, activation = 'tanh') %>%", 
                              n_hidden_layers), collapse = " "), 
                    "layer_dense(units = n_outputs)")
eval(parse(text = mdl_string))
summary(model)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam'
)
keras.time <- proc.time()
history <- model %>% fit(
  xtrain_scaled, ytrain_scaled,
  epochs = n_epochs, 
  batch_size = n_batch_size, 
  validation_data = list(xtest_scaled, ytest_scaled), 
  verbose = verbose
)
proc.time() - keras.time #keras ann fitting time

png(filename='output/ann_convergence.png')
plot(history)
dev.off()

weights <- get_weights(model) #get ANN weights
pred <- model %>% predict(xtest_scaled)
ytest_scaled_pred <- data.frame(pred)
colnames(ytest_scaled_pred) <- y_names
head(ytest_scaled_pred)

ann_valid <- rbind(data.frame(sim = 1:n_test, ytest_scaled, type = "model"), 
                   data.frame(sim = 1:n_test, ytest_scaled_pred, type = "pred"))
ann_valid_transpose <- ann_valid %>% 
  pivot_longer(cols = -c(sim, type)) %>% 
  pivot_wider(id_cols = c(sim, name), names_from = type, values_from = value)
ggplot(data = ann_valid_transpose, aes(x = model, y = pred)) + 
  geom_point(alpha = 0.5, color = "tomato") + 
  facet_wrap(~name, ncol = 6) + 
  xlab("Model outputs (scaled)") + 
  ylab("ANN predictions (scaled)") + 
  coord_equal() + 
  theme_bw()

ggsave(filename = "figs/fig4_ann_validation_vs_observed.pdf", width = 8.5, height = 11)
ggsave(filename = "figs/fig4_ann_validation_vs_observed.png", width = 8.5, height = 11)
ggsave(filename = "figs/fig4_ann_validation_vs_observed.jpg", width = 8.5, height = 11)


