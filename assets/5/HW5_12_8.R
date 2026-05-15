pr <- prcomp(USArrests, scale = TRUE)
pr$sdev^2 / sum(pr$sdev^2)
colSums(pr$x^2) / sum(colSums(scale(USArrests)^2))
