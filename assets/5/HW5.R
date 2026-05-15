# a. Plot the observations.


library(ggplot2)
d <- data.frame(
  x1 = c(1, 1, 0, 5, 6, 4),
  x2 = c(4, 3, 4, 1, 2, 0)
)
ggplot(d, aes(x = x1, y = x2)) + geom_point()


# b. Randomly assign a cluster label to each observation. You can use the

set.seed(42)
d$cluster <- sample(c(1, 2), size = nrow(d), replace = TRUE)

#> c. Compute the centroid for each cluster.

centroids <- sapply(c(1,2), function(i) colMeans(d[d$cluster == i, 1:2]))

# d. Assign each observation to the centroid to which it is closest, in terms of

dist <- sapply(1:2, function(i) {
    sqrt((d$x1 - centroids[1, i])^2 + (d$x2 - centroids[2, i])^2)
})
d$cluster <- apply(dist, 1, which.min)

# e. Repeat (c) and (d) until the answers obtained stop changing.

centroids <- sapply(c(1,2), function(i) colMeans(d[d$cluster == i, 1:2]))
dist <- sapply(1:2, function(i) {
    sqrt((d$x1 - centroids[1, i])^2 + (d$x2 - centroids[2, i])^2)
})
d$cluster <- apply(dist, 1, which.min)

# f. In your plot from (a), color the observations according to the cluster
ggplot(d, aes(x = x1, y = x2, color = factor(cluster))) + geom_point()

