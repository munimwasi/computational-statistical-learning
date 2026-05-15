rm(list = ls())

set.seed(1)
n <- 20
m <- 100
X <- matrix(rnorm(n * m), ncol = m)

#  > a. Conduct a one-sample $t$-test for each fund manager, and plot a histogram
pvals <- apply(X, 2, function(p) t.test(p)$p.value)
hist(pvals, main = NULL)

#> b. If we control Type I error for each null hypothesis at level $\alpha =
sum(pvals < 0.05)

#> c. If we control the FWER at level 0.05, then how many null hypotheses do we
sum(pvals < 0.05 / length(pvals))

#> d. If we control the FDR at level 0.05, then how many null hypotheses do we
sum(p.adjust(pvals, "fdr") < 0.05)

#> e. Now suppose we “cherry-pick” the 10 fund managers who perform the best in
best <- order(apply(X, 2, sum), decreasing = TRUE)[1:10]
sum(pvals[best] < 0.05 / 10)
sum(p.adjust(pvals[best], "fdr") < 0.05)

#> f. Explain why the analysis in (e) is misleading.

