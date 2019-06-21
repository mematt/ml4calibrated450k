#--------------------------------------------------------------------
# ml4calibrated450k - tuned RF (tRF) - Utility/Subfunctions 
#
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-24 
#--------------------------------------------------------------------


## Load required libraries, data objects 

# Check, install | load recquired packages ---------------------------------------------------------------------------------------------------------------------------
if (!requireNamespace("caret", quietly = TRUE)) { 
  install.packages("caret", dependencies = T)
  library(caret) 
} else {library(caret)}

if (!requireNamespace("randomForest", quietly = TRUE)) { 
  install.packages("randomForest")
  library(randomForest) } else {library(randomForest)}

if (!requireNamespace("doMC", quietly = TRUE)) { 
  install.packages("doMC")
  library(doMC) } else {library(doMC)}


# Source evaluation metrics --------------------------------------------------------------------------------------------------------------------------- 
source("evaluation_metrics.R") 
# Brier score, ME, log loss # needed for `p` variable selection tuning
# mAUC (Hand & Till 2001)


## Utility functions ---------------------------------------------------------------------------------------------------------------------------

### Subfunctions 

############################################
### 0. Nested CV scheduler / stopper     ###
############################################

subfunc_nestedcv_scheduler <- function(K, K.start, K.stop, k.start, k.stop, n.cv.folds, n.cv.inner.folds){
  
  # Correctly stop nested CV
  if(K > K.start && K < K.stop) {
    k.start <- 0
    n.cv.inner.folds <- n.cv.folds 
    
  } else {
    if(K == K.start) {
      if(K.start != K.stop) { # && k.start != 0){
        n.cv.inner.folds <- n.cv.folds # k inner goes to .5
      } else { # K == K.start == K.stop && k.start == 0  
        n.cv.inner.folds <- k.stop
      }
    } else { # K == K.stop
      if(k.start != 0) k.start <- 0
      n.cv.inner.folds <- k.stop
    } 
  }
  res <- list(k.start = k.start, 
              n.cv.inner.folds = n.cv.inner.folds
  )
  return(res)
}

############################################
### 1. Parallelized randomForest wrapper ###
############################################

rfp <- function(xx, ..., ntree = ntree, mc = mc, seed = NULL){
  if(!is.null(seed)) set.seed(seed, "L'Ecuyer") # help("mclapply") # reproducibility
  rfwrap <- function(ntree, xx, ...) randomForest::randomForest(x = xx, ntree = ntree, norm.votes = FALSE, ...)
  rfpar <- mclapply(rep(ceiling(ntree / mc), mc), mc.cores = mc, rfwrap, xx = xx, ...) 
  do.call(randomForest::combine, rfpar) # > ?randomForest::combine => combines two or more trees/ensembles of trees (rF objects) into one # get the subset of trees from each core and patch them back together
}

###########################################################################################
### 2. Custom random forest function within the caret package to allow for ntree tuning ###
###########################################################################################

# Custom function within the caret package to tune ntree, mtry and node.size paramters of randomForest() 
# This function uses the parllelized/wrapped rfp() function from above

# CustomRF for caret
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)

customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), 
                                  class = rep("numeric", 3), 
                                  label = c("mtry", "ntree", "nodesize"))

customRF$grid <- function(x, y, len = NULL, search = "grid"){}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...){
  rfp(xx = x, y = y, 
      mc = 4L,            # set multi-core fixed to 4 # can be changed according to hardware # AWS64 optimized # for 8-12 thread laptops use 2L
      mtry = param$mtry,  # tuneable parameter
      ntree=param$ntree,  # tuneable parameter
      nodesize = param$nodesize, #tuneable parameter
      strata = y, 
      sampsize = rep(min(table(y)), length(table(y))), # downsampling to the minority class 
      importance = TRUE, 
      replace = FALSE,  
      ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL) predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL) predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[, 1]), ]
customRF$levels <- function(x) x$classes


###################################################
###       FINE TUNE RF hyperparameters          ###
###       using customRF() and caret            ###
################################################### 

# This function is used within the train_tunedRF.R script and the trainRF_caret_custom_tuner() function

subfunc_rf_caret_tuner_customRF <- function(y.subfunc, betas.subfunc, 
                                            mtry.min = NULL, mtry.max = NULL, length.mtry = 6, 
                                            ntrees.min = 1000, ntrees.max = 2000, ntree.by = 500,
                                            use.default.nodesize.1 = T,
                                            nodesize.proc = c(0.01, 0.05, 0.1),
                                            n.cv = 5, n.rep = 1,
                                            seed = 1234, allowParall = T,
                                            print.res = T){
  # Train control
  control <- trainControl(method = "repeatedcv", number = n.cv, repeats = n.rep, 
                          classProbs = F, summaryFunction = multiClassSummary, 
                          allowParallel = allowParall)
  
  # Tune grid
  if(is.null(mtry.min)){mtry.min <- floor(sqrt(ncol(betas.subfunc)))*0.5}
  if(is.null(mtry.max)){mtry.max <- floor(sqrt(ncol(betas.subfunc)))} # defaults to sqrt(10000) = 100 
  
  # Limit grid size by using only the default nodesize for classification = 1 
  # (this performs the best and was chosen for BTMD for all scenarios and tuning measures)
  if(use.default.nodesize.1) {
    message("<NOTE>: `use.default.nodesize.1` = ", use.default.nodesize.1, 
            " thus `nodesize.proc` values are ignored and only the default value (randomForest) for classification = 1 will be used.")
    tunegrid <- expand.grid(.mtry = floor(seq(mtry.min, mtry.max, length.out = length.mtry)),
                            .ntree = seq(ntrees.min, ntrees.max, ntree.by), 
                            .nodesize = 1)
    
  } else {
    message("Tune terminal .nodesize for defaults classification = 1, regression = 5, and as assigned in `nodesize.proc`=[", nodesize.proc, "] values.")
    value_nodesizes <- nodesize.proc * nrow(betas.subfunc)
    tunegrid <- expand.grid(.mtry = floor(seq(mtry.min, mtry.max, length.out = length.mtry)),
                            .ntree = seq(ntrees.min, ntrees.max, ntree.by), 
                            .nodesize = floor(c(1, 5, value_nodesizes))
    )
  }
  # Train - 5x CV (n.cv) for tuning parameters with single repeat (n.rep)
  set.seed(seed, kind = "L'Ecuyer-CMRG")
  rf.tuned.caret.customRF <- train(x = betas.subfunc, y = y.subfunc, method = customRF, metric = "Accuracy", tuneGrid = tunegrid, trControl = control)
  if(print.res){print(rf.tuned.caret.customRF)}
  return(rf.tuned.caret.customRF)
}
