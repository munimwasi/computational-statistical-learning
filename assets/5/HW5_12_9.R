#a. Using hierarchical clustering with complete linkage and Euclidean distance,

set.seed(42)
hc <- hclust(dist(USArrests), method = "complete")

# b. Cut the dendrogram at a height that results in three distinct clusters.
ct <- cutree(hc, 3)
sapply(1:3, function(i) names(ct)[ct == i])

# c. Hierarchically cluster the states using complete linkage and Euclidean
hc2 <- hclust(dist(scale(USArrests)), method = "complete")

# d. What effect does scaling the variables have on the hierarchical clustering
ct <- cutree(hc, 3)
sapply(1:3, function(i) names(ct)[ct == i])

