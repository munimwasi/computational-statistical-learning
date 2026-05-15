#  > a. Load in the data using `read.csv()`. You will need to select `header = F`.

data <- read.csv("Ch12Ex13.csv", header = FALSE)
colnames(data) <- c(paste0("H", 1:20), paste0("D", 1:20))

#> b. Apply hierarchical clustering to the samples using correlation-based
hc.complete <- hclust(as.dist(1 - cor(data)), method = "complete")
plot(hc.complete)

hc.complete <- hclust(as.dist(1 - cor(data)), method = "average")
plot(hc.complete)

hc.complete <- hclust(as.dist(1 - cor(data)), method = "single")
plot(hc.complete)

#> c. Your collaborator wants to know which genes differ the most across the two
class <- factor(rep(c("Healthy", "Diseased"), each = 20))
pvals <- p.adjust(apply(data, 1, function(v) t.test(v ~ class)$p.value))
which(pvals < 0.05)
