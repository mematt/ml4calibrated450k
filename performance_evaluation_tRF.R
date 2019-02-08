# Source scripts of performance metrics, import packages and define subfunctions

# Source
# Subfunction for Brier score (BS)
source("brier.R")
# Subfunction for multiclass log loss (LL)
source("mlogloss.R")

# Subfunction for misclassification error (ME)
subfunc_misclassification_rate <- function(y.true.class, y.predicted){
  error_misclass <- sum(y.true.class != y.predicted)/length(y.true.class)
  return(error_misclass)
}

# Subfunction for multiclass AUC & ROC by Hand & Till 2001
library(HandTill2001) 
# Note: sum of row scores/probabilities must be scaled to 1 
subfunc_multiclass_AUC_HandTill2001 <- function(y.true.class, y.pred.matrix.rowsum.scaled1){
  auc_multiclass <- HandTill2001::auc(multcap(response = as.factor(y.true.class), 
                                              predicted = y.pred.matrix.rowsum.scaled1))
  return(auc_multiclass)
}


# Performance evaluator function (all metrics integrated)
performance_evaluator <- function(load.path.w.name. = "./tRF/MR-calibrated/probsCVfold.brier.", 
                                  name.of.obj.to.load = NULL, # as.character # defaults to "probs"  
                                  nfolds.. = NULL, 
                                  betas.. = NULL,         # defaults to BTMD obects
                                  y.. = NULL, # defaults to BTMD obects
                                  scale.rowsum.to.1 = T,
                                  reorder.columns = F, 
                                  reorder.rows = T, 
                                  misc.err = T, 
                                  multi.auc.HandTill2001 = T, 
                                  brier = T, 
                                  mlogLoss = T,
                                  verbose = T){
  # Start @ 
  if(verbose) message("\nStart performance evaluation @ ", Sys.time())
  
  # 1. Checks
  # Check whether nfolds.. is available
  if(is.null(nfolds..) & exists("nfolds")){nfolds.. <- get("nfolds", envir = .GlobalEnv)} else {
    stop("Please provide a fold structure for nested resampling. For instance, load the nfolds.RData")}
  # Check whether "betas" data.frame is available
  if(is.null(betas..) & exists("betas")){betas.. <- get("betas", envir = .GlobalEnv)} else {
    stop("Please provide the reference cohort methylation data frame (betas) for further analyses. For instance, load the betas1000.RData or the MNPbetas10Kvar.RData")}
  # Check whether y true outcome is available
  if(is.null(y..) & exists("y")){y.. <- get("y", envir = .GlobalEnv)} else {
    stop("Please provide the true class labels vector (y) for further analyses. For instance, load the y.RData or the MNPbetas10Kvar.RData")}
  # Check whether provided betas.. and y.. has the same dimension
  if(nrow(betas..) == length(y..)){cat("Checked: betas.. and y.. have the same corresponding dimensions.")} else{
    stop("Error: betas.. and y.. have different dimensions.")}
  # Set default "probs" for name.of.obj.to.load
  if(is.null(name.of.obj.to.load)) name.of.obj.to.load <- "probs"
  
  # Load objects with scores/probes
  probs.l <- list()
  for(i in 1:length(nfolds..)){
    # Create loading environment for safe loading the RData files of outerfolds
    env2load.outer <- environment()
    # Load
    if(verbose) message("Load the test set from the ", i, ".0 outer fold @ ", Sys.time())
    load(paste0(load.path.w.name., i, ".", 0, ".RData"), envir = env2load.outer)                                        
    probs.loaded <- get(as.character(name.of.obj.to.load), envir = env2load.outer)
    probs.l[[i]] <-  probs.loaded
  }
  
  # Patch the score & prob  list together as a matrix
  if(verbose) message("Combine loaded outer fold test sets  1.0 ... ", length(nfolds..), ".0")
  probs <- do.call(rbind, probs.l)
  if(verbose) message("Dimensions of the combined matrix: ", nrow(probs), " x ", ncol(probs))
  
  # Visual inspection of row and column levels (e.g. SVM mixes the order of both)
  #print(rownames(probs)[1:5])
  #print(rownames(betas..)[1:5])
  if(verbose) message("\nAre predicted class labels (colnames) identically ordered as levels of the true outcome (y..): ", identical(colnames(probs), levels(y..)))
  if(reorder.columns){
    message("Match colnames of probability output to levels(y) ... ")
    probs <- probs[ , match(levels(y..), colnames(probs))]
  }
  if(verbose) message("Check whether colnames are identically ordered after re-matching: ", identical(colnames(probs), levels(y..)))
  
  # Reorder cases/patients as in betas.. 
  if(reorder.rows){
    message("\nReorder sample (i.e. row) names as in betas & y (true outcome)")   
    probs <- probs[match(rownames(betas..), rownames(probs)), ]
  }
  message("Check whether rownames identical:", identical(rownames(probs), rownames(betas..)))
  
  # Rowsum check
  rowsum.p <- apply(probs, 1, sum)
  if(verbose) message("\nRow sum of first 10 cases :", paste(as.matrix(rowsum.p[1:10]), collapse = " ; "))
  # Show scores / probs ranges:
  p.range <- range(probs)
  if(verbose) message("The range of raw and/or calibrated object", " '", as.character(name.of.obj.to.load), "' ", 
                      "is : ", "\n", p.range[[1]], " - ", p.range[[2]])
  
  # Scale rowsum to 1 - default
  if(scale.rowsum.to.1){
    if(verbose) message("Scaling row sums to 1 (by default = TRUE): ")
    probs.rowsum1 <- t(apply(probs, 1, function(x) x/sum(x)))
  } else{probs.rowsum1 <-  probs}
  
  # Predicted class column with highest prob for each row (patient)
  y.p.rowsum1 <- colnames(probs.rowsum1)[apply(probs.rowsum1,1, which.max)] 
  y.l <- list(y.p.rowsum1)
  # Misclassification Error
  if(misc.err){
    err.misc.l <- lapply(y.l, subfunc_misclassification_rate, y.true.class = y..) 
    err.misc <- unlist(err.misc.l)
    message("\nMisclassification Error: ", err.misc)
  }
  # AUC HandTIll2001
  if(multi.auc.HandTill2001){
    results.sc.p.rowsum1.l <- list(probs.rowsum1)
    message("Calculating multiclass AUC (Hand&Till 2001) ... ", Sys.time())
    auc.HT2001.l <- lapply(results.sc.p.rowsum1.l, subfunc_multiclass_AUC_HandTill2001, y.true.class = y..)
    auc.HT2001 <- unlist(auc.HT2001.l)
    message("Multiclass AUC (Hand&Till 2001): ", auc.HT2001)
  } else{
    message("Multiclass AUC (Hand&Till 2001) is set to FALSE =>", 
            "Calculating RAW probs (row sum != 1). auc.HT2001 is set to NA.")
    auc.HT2001 <- NA
  }
  # Brier 
  if(brier){
    brierp.rowsum1 <- brier(scores = probs.rowsum1, y = y..)  # class(brierp.rowsum1)   # numeric
    message("Brier score (BS): ", brierp.rowsum1)
  }
  # mlogloss
  if(mlogLoss){
    loglp.rowsum1 <- mlogloss(scores = probs.rowsum1, y = y..) # class(loglp.rowsum1)  # numeric
    message("Multiclass log loss (LL): ", loglp.rowsum1)
  }
  # End of run
  message("Finished @ ", Sys.time())
  
  # Results
  res <- list(misc.error = err.misc, 
              auc.HandTill = auc.HT2001, 
              brier = brierp.rowsum1, 
              mlogloss = loglp.rowsum1)
  return(res)
} # end of func         

# # Function call:
# # Timing < 1 min  
# performance_evaluator(name.of.obj.to.load = "scores")
# performance_evaluator() # defaults to the "probs" object
# 
# # Or with lapply()
# l.obj.names <- list("scores", "probs")
# l.perf.eval <- lapply(seq_along(l.obj.names), function(i){
#   performance_evaluator(name.of.obj.to.load = l.obj.names[[i]])
# })
