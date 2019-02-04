# Multiclass AUC - Hand and Till (2001)

# An example of generatic multiclass AUC values based on the scores and probabilities of the vanilla random forest (vRF) classifier.
# We assume that the output .RData objects of the 5 x 5 cross validation scheme are stored within a the subdirectory ./CV and are named probsCVfold.1.0, ... 1.1 ... etc.
# Each probsCVfold.i.0.RData file contains the matrices of scores and probs

# Generate empty lists for scores and probabilities 
probl <- scorel <- list()

# Cycle through (outer)folds and assign the respective scores and probs objects to the list 
for(i in 1:length(nfolds)){
  load(paste0("./CV/probsCVfold.",i,".",0,".RData"))
  probl[[i]] <- probs
  scorel[[i]] <- scores
}

# Collapse the lists into a large matrix
scores <- do.call(rbind, scorel)    ## rows: patients ; columns: classes (identical to levels of y see line 26, y)  values scores  # dim(scores) 2801   91
probs <- do.call(rbind, probl)      ## rows: patients ; columns: classes (identical to levels of y see line 26, y)  values probs   # dim(probs)  2801   91

# Match rownames of scores and probs to the original betas data.frame so the order of the cases is the same as in the outcome factor (y=anno$V5) ; columns stay unchanged (same as the order of levels of y)
scores <- scores[match(rownames(betas), rownames(scores)), ]
probs <- probs[match(rownames(betas), rownames(probs)), ]

# Outcome factor (vector of length 2801 i.e. the number of patients in the cohort)
y <- load("y.RData")
levels(y) # 91 possible diagnostic classes

library(HandTill2001) # A very lean package implementing merely M given by Hand and Till (2001)

# Evaluate multiclass AUC - Hand and Till (2001)

# AUC of (raw/uncalibrated) predictor scores
AUCscores <- HandTill2001::auc(multcap(response = as.factor(y), predicted = scores))

# AUC of calibrated (post-processed) probabilities
AUCprobs <-  HandTill2001::auc(multcap(response = as.factor(y), predicted = probs))
