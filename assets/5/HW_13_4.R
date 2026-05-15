rm(list = ls())
pvals <- c(0.0011, 0.031, 0.017, 0.32, 0.11, 0.90, 0.07, 0.006, 0.004, 0.0009)
names(pvals) <- paste0("H", sprintf("%02d", 1:10))

#> a. Suppose that we wish to control the Type I error for each null hypothesis
names(which(pvals < 0.05))

#> b. Now suppose that we wish to control the FWER at level $\alpha = 0.05$.
names(which(pvals < 0.05 / 10))

#> c. Now suppose that we wish to control the FDR at level $q = 0.05$. Which null
names(which(p.adjust(pvals, "fdr") < 0.05))
#> d. Now suppose that we wish to control the FDR at level $q = 0.2$. Which null
names(which(p.adjust(pvals, "fdr") < 0.2))

#> e. Of the null hypotheses rejected at FDR level $q = 0.2$, approximately how
#> many are false positives? Justify your answer.

#We expect 20% (in this case 2 out of the 8) rejections to be false (false
#                                                                    positives).

