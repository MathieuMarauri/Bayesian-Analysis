
# Bayesian optimisation framework (https://bearloga.github.io/bayesopt-tutorial-r/)

# Set-up --------------------------------------------------------------------------------------

library("ggplot2") # data visualisation
library("GPfit") # Gaussian Process fit
library("patchwork") # arrange multiple plots
library("tensorflow") # tensorflow API to use Adam Optimizer
source("src/helpers.R")

# Set default ggplot theme
theme_set(
  theme_light(
  base_size = 20
  ) +
  theme(
    text = element_text(family = "Gibson", colour = "gray10"),
    panel.border = element_blank(),
    axis.line = element_line(colour = "gray50", size = .5),
    axis.ticks = element_blank(),
    strip.background = element_rect(colour = "gray50", fill = "transparent", size = .7),
    strip.text.x = element_text(colour = "gray10"),
    legend.key.size = unit(2, "cm")
  )
)


# Objective function  -------------------------------------------------------------------------

# Define several objective functions that will be used as example. The interval on which
# optimisation is done is [0,1].

# First function: objective is minimisation
f <- function(x) (6 * x - 2)^2 * sin(12 * x - 4)

# Brute force minimum
x <- seq(0, 1, length.out = 1000000)
f_x <- f(x)
arg_min <- x[which.min(f_x)]
min <- min(f_x)

# Visualise function and minimum
ggplot(
  data = data.frame(x = x, y = f(x)),
  mapping = aes(x, y)
) +
  geom_line() +
  geom_point(
    mapping = aes(x = arg_min, y = min),
    color = "red",
    size = 3
  ) +
  labs(
    y = "f(x)",
    title = "Objective function and its minimum"
  )


# Baseline using tensorflow -------------------------------------------------------------------

# Compute the argmin of the function using tensorflow Adam optimizer.

# Define the tf session
sess <- tf$Session()

# Define a trainable variable
x <- tf$Variable(0.0, trainable = TRUE)

# Define the objective function using tensorflow sinus
f <- function(x) (6 * x - 2)^2 * tf$sin(12 * x - 4)

# Train the Adam optimizer on f
adam <- tf$train$AdamOptimizer(learning_rate = 0.3)
f_x <- f(x)
opt <- adam$minimize(f_x, var_list = x)

# Store the value of x and f(x) at each iteration
sess$run(tf$global_variables_initializer())
n_iter <- 30
arg_min <- vector(mode = "numeric", length = n_iter)
min <- vector(mode = "numeric", length = n_iter)
for (i in 1:n_iter) {
  sess$run(opt)
  arg_min[i] <- sess$run(x)
  min[i] <- sess$run(f_x)
}
arg_min[n_iter]

# Visualise the optimizer results
x <- seq(0, 1, length.out = 1000)
f <- function(x) (6 * x - 2)^2 * sin(12 * x - 4)
plots <- vector(mode = "list", length = length(arg_min))
for (i in seq_len(length(arg_min))) {
  plots[[i]] <- ggplot(
    data = data.frame(x = x, y = f(x)),
    mapping = aes(x = x, y = y)
  ) +
    geom_line() +
    geom_point(
      data = data.frame(x = arg_min[1:(i - 1)], y = min[1:(i - 1)]),
      mapping = aes(x = x, y = y),
      color = "grey",
      size = 2
    ) +
    geom_point(
      data = data.frame(x = arg_min[i], y = min[i]),
      mapping = aes(x = x, y = y),
      color = "red",
      size = 3
    ) +
    labs(
      y = "f(x)",
      title = glue::glue("Iteration {i}")
    )
}
createGif(plots, "output/adam.gif", 20)

# Clean the environment
rm(adam, f_x, opt, plots, sess, arg_min, min, i, n_iter, x)


# Gaussian process ----------------------------------------------------------------------------

# A gaussian process is fitted to the data and will be used latter as a surrogate model in the
# different acquisition functions.

x <- seq(0, 1, length.out = 1000)
f <- function(x) (6 * x - 2)^2 * sin(12 * x - 4)

### Define the gaussian process

# Build the GP model with 4 initial points
init_eval <- data.frame(x = seq(0, 1, length.out = 4), y = f(seq(0, 1, length.out = 4)))
fit <- GP_fit(
  X = init_eval[, "x"],
  Y = init_eval[, "y"],
  corr = list(type = "exponential", power = 1.95)
)

# Predict value at 100 points with the GP prior
x_new <- seq(0, 1, length.out = 100)
pred <- predict.GP(fit, xnew = data.frame(x = x_new))
mu <- pred$Y_hat
sigma <- sqrt(pred$MSE)

# Visualise the prior GP
ggplot(
  data = data.frame(x = x_new, y = mu, se_lower = mu - sigma, se_upper = mu + sigma)
) +
  geom_line(
    mapping = aes(x = x, y = y),
    linetype = "dashed"
  ) +
  geom_ribbon(
    mapping = aes(x = x, ymin = se_lower, ymax = se_upper),
    alpha = .7
  ) +
  geom_point(
    data = init_eval,
    mapping = aes(x = x, y = y)
  ) +
  geom_line(
    data = data.frame(x = x, y = f(x)),
    mapping = aes(x = x, y = y)
  ) +
  labs(
    y = "f(x)",
    title = "Prior GP distribution",
    subtitle = "The function has been evaluated at 4 initial points."
  )


# Probability of improvement ------------------------------------------------------------------

# Build the acquisition function using probability of improvement. It compute the probability that a
# new value leads to a value at least as low as the previous best one. The cumulative normal
# distribution is used.

# Define the best value so far
y_best <- min(init_eval$y)

# Compute the probability of improvement for all points (points with previous estimation have poi 0)
probability_improvement <- purrr::map2_dbl(mu, sigma, function(m, s) {
  if (s == 0) return(0)
  else {
    poi <- pnorm((y_best - m) / s) # if the objective is to maximise then use 1 - poi
    return(poi)
  }
})

# Next point: arg max of the PI
x_next <- x_new[which.max(probability_improvement)]

# Visualise the poi with its best point
ggplot(
  data = data.frame(x = x_new, y = probability_improvement),
  mapping = aes(x = x, y = y)
) +
  geom_line() +
  geom_point(
    mapping = aes(x = x_next, y = max(probability_improvement)),
    color = "red",
    size = 3
  ) +
  labs(
    y = "Probability of improvement",
    title = "Acquisition function to select the next best point"
  )

# Fonction value at next point
y_next <- f(x_next)

# Optimisation of the objective function
iter_max <- 10
iter <- 1
evaluations <- data.frame(x = seq(0, 1, length.out = 4), y = f(seq(0, 1, length.out = 4)))
y_best <- min(evaluations$y)
pi_values <- vector(mode = "list", length = iter_max)
gp_values <- vector(mode = "list", length = iter_max)
while (iter <= iter_max) {
  print(iter)
  # Fit a GP to the data
  set.seed(12)
  fit <- GP_fit(
    X = evaluations[, "x"],
    Y = evaluations[, "y"],
    corr = list(type = "exponential", power = 1.95)
  )
  # Update the GP with new data
  x_new <- seq(0, 1, length.out = 100)
  pred <- predict.GP(fit, xnew = data.frame(x = x_new))
  mu <- pred$Y_hat
  sigma <- sqrt(pred$MSE)
  gp_values[[iter]] <- data.frame(x = x_new, y = mu, se_lower = mu - sigma, se_upper = mu + sigma)
  # Find the next point to evaluate
  probability_improvement <- purrr::map2_dbl(mu, sigma, function(m, s) {
    if (s == 0) return(0)
    else {
      poi <- pnorm((y_best - m) / s) # if the objective is to maximise then use 1 - poi
      return(poi)
    }
  })
  pi_values[[iter]] <- probability_improvement
  x_next <- x_new[which.max(probability_improvement)]
  y_next <- f(x_next)
  # Update the loop variables
  iter <- iter + 1
  # Update the evaluations and the best value
  evaluations <- rbind(evaluations, c("x" = x_next, "y" = y_next))
  y_best <- min(evaluations$y)
}

# Visualise the acquisition function and the posterior distribution at each step
plots <- vector(mode = "list", length = iter_max)
for (i in 1:iter_max) {
  pi <- ggplot(
    data = data.frame(x = x_new, y = pi_values[[i]]),
    mapping = aes(x = x, y = y)
  ) +
    geom_line() +
    geom_point(
      data = data.frame(x = x_new[which.max(pi_values[[i]])], y = max(pi_values[[i]])),
      mapping = aes(x = x, y = y),
      color = "red",
      size = 3
    ) +
    labs(
      y = "PI(x)",
      title = "Acquisition function"
    )
  gp <- ggplot(
    data = gp_values[[i]]
  ) +
    geom_line(
      mapping = aes(x = x, y = y),
      linetype = "dashed"
    ) +
    geom_ribbon(
      mapping = aes(x = x, ymin = se_lower, ymax = se_upper),
      alpha = .7
    ) +
    geom_point(
      data = evaluations[1:(i + 3),],
      mapping = aes(x = x, y = y)
    ) +
    geom_point(
      data = evaluations[i + 4,],
      mapping = aes(x = x, y = y),
      color = "red"
    ) +
    geom_line(
      data = data.frame(x = x, y = f(x)),
      mapping = aes(x = x, y = y)
    ) +
    labs(
      y = "f(x)",
      title = "Posterior GP distribution"
    )
  plots[[i]] <- pi + gp
}
createGif(plots, "output/bayes_pi.git", 20)

# Clean the envirnonment
rm(evaluations, fit, gp, gp_values, pi, pi_values, plots, pred, f_x, i, iter,
   iter_max, mu, probability_improvement, sigma, x_new, x_next, y_next, y_best, x)

# Expected improvement ------------------------------------------------------------------------

# Build the acquisition function using probability of improvement. It compute the probability that a
# new value leads to a value at least as low as the previous best one. The cumulative normal
# distribution is used.

# Build the GP process
x <- seq(0, 1, length.out = 1000)
f <- function(x) (6 * x - 2)^2 * sin(12 * x - 4)
init_eval <- data.frame(x = seq(0, 1, length.out = 4), y = f(seq(0, 1, length.out = 4)))
fit <- GP_fit(
  X = init_eval[, "x"],
  Y = init_eval[, "y"],
  corr = list(type = "exponential", power = 1.95)
)
x_new <- seq(0, 1, length.out = 100)
pred <- predict.GP(fit, xnew = data.frame(x = x_new))
mu <- pred$Y_hat
sigma <- sqrt(pred$MSE)

# Define the best value so far
y_best <- min(init_eval$y)

# Compute the probability of improvement for all points (points with previous estimation have poi 0)
expected_improvement <- purrr::map2_dbl(mu, sigma, function(m, s) {
  if (s == 0) return(0)
  gamma <- (y_best - m) / s
  phi <- pnorm(gamma)
  return(s * (gamma * phi + dnorm(gamma)))
})

# Next point: arg max of the PI
x_next <- x_new[which.max(expected_improvement)]

# Visualise the poi with its best point
ggplot(
  data = data.frame(x = x_new, y = expected_improvement),
  mapping = aes(x = x, y = y)
) +
  geom_line() +
  geom_point(
    mapping = aes(x = x_next, y = max(expected_improvement)),
    color = "red",
    size = 3
  ) +
  labs(
    y = "Expected improvement",
    title = "Acquisition function to select the next best point"
  )

# Fonction value at next point
y_next <- f(x_next)

# Optimisation of the objective function
iter_max <- 10
iter <- 1
evaluations <- data.frame(x = seq(0, 1, length.out = 4), y = f(seq(0, 1, length.out = 4)))
y_best <- min(evaluations$y)
pi_values <- vector(mode = "list", length = iter_max)
gp_values <- vector(mode = "list", length = iter_max)
while (iter <= iter_max) {
  print(iter)
  # Fit a GP to the data
  set.seed(12)
  fit <- GP_fit(
    X = evaluations[, "x"],
    Y = evaluations[, "y"],
    corr = list(type = "exponential", power = 1.95)
  )
  # Update the GP with new data
  x_new <- seq(0, 1, length.out = 100)
  pred <- predict.GP(fit, xnew = data.frame(x = x_new))
  mu <- pred$Y_hat
  sigma <- sqrt(pred$MSE)
  gp_values[[iter]] <- data.frame(x = x_new, y = mu, se_lower = mu - sigma, se_upper = mu + sigma)
  # Find the next point to evaluate
  probability_improvement <- purrr::map2_dbl(mu, sigma, function(m, s) {
    if (s == 0) return(0)
    gamma <- (y_best - m) / s
    phi <- pnorm(gamma)
    return(s * (gamma * phi + dnorm(gamma)))
  })
  pi_values[[iter]] <- probability_improvement
  x_next <- x_new[which.max(probability_improvement)]
  y_next <- f(x_next)
  # Update the loop variables
  iter <- iter + 1
  # Update the evaluations and the best value
  evaluations <- rbind(evaluations, c("x" = x_next, "y" = y_next))
  y_best <- min(evaluations$y)
}

# Visualise the acquisition function and the posterior distribution at each step
plots <- vector(mode = "list", length = iter_max)
for (i in 1:iter_max) {
  pi <- ggplot(
    data = data.frame(x = x_new, y = pi_values[[i]]),
    mapping = aes(x = x, y = y)
  ) +
    geom_line() +
    geom_point(
      data = data.frame(x = x_new[which.max(pi_values[[i]])], y = max(pi_values[[i]])),
      mapping = aes(x = x, y = y),
      color = "red",
      size = 3
    ) +
    labs(
      y = "EI(x)",
      title = "Acquisition function"
    )
  gp <- ggplot(
    data = gp_values[[i]]
  ) +
    geom_line(
      mapping = aes(x = x, y = y),
      linetype = "dashed"
    ) +
    geom_ribbon(
      mapping = aes(x = x, ymin = se_lower, ymax = se_upper),
      alpha = .7
    ) +
    geom_point(
      data = evaluations[1:(i + 3),],
      mapping = aes(x = x, y = y)
    ) +
    geom_point(
      data = evaluations[i + 4,],
      mapping = aes(x = x, y = y),
      color = "red"
    ) +
    geom_line(
      data = data.frame(x = x, y = f(x)),
      mapping = aes(x = x, y = y)
    ) +
    labs(
      y = "f(x)",
      title = "Posterior GP distribution"
    )
  plots[[i]] <- pi + gp
}
createGif(plots, "output/bayes_ei.git", 20)



