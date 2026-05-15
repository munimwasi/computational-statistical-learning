library(survival)
x <- Surv(c(1.2, 2, 1.5, 0.2), event = c(1, 0, 0, 1))
summary(survfit(x ~ 1))

library(tidyverse)

table_data <- tribble(
  ~Y, ~D, ~X, 
  26.5, 1, 0.1,
  37.2, 1, 11,
  57.3, 1, -0.3,
  90.8, 0, 2.8,
  20.2, 0, 1.8,
  89.8, 0, 0.4
)
x <- Surv(table_data$Y, table_data$D)
summary(survfit(x ~ 1))

x <- split(Surv(table_data$Y, table_data$D), table_data$X < 2)
plot(NULL, xlim = c(0, 100), ylim = c(0, 1), ylab = "Survival Probability")
lines(survfit(x[[1]] ~ 1), conf.int=FALSE, col = 2)
lines(survfit(x[[2]] ~ 1), conf.int=FALSE, col = 3)
legend("bottomleft", c(">= 2", "<2"), col = 2:3, lty = 1)