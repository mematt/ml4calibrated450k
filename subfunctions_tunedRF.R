########################################
### SUBFUNCTIONS for train_tunedRF.R ###
########################################

library(caret)
library(randomForest)

############################################
### 1. Parallelized randomForest wrapper ###
############################################

rfp <- function(xx, ..., ntree = ntree, mc = mc, seed = NULL) 
{
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
      mc = 4L,             # set multi-core fixed to 4 # can be changed according to hardware
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
###                                             ###
###################################################

# This function is used within the train_tunedRF.R script and the trainRF_caret_custom_tuner() function

subfunc_rf_caret_tuner_customRF <- function(y.subfunc, betas.subfunc, 
                                            mtry.min = NULL, mtry.max = NULL, length.mtry = 6, 
                                            ntrees.min = 1000, ntrees.max = 2000, ntree.by = 500,
                                            nodesize.proc = c(0.01, 0.05, 0.1),
                                            n.cv = 5, n.rep = 1,
                                            seed = 1234, allowParall = T,
                                            print.res = T){
  # Train control
  control <- trainControl(method = "repeatedcv", number = n.cv, repeats = n.rep, classProbs = F, summaryFunction = multiClassSummary, allowParallel = allowParall)
  
  # Tune grid
  if(is.null(mtry.min)){mtry.min <- floor(sqrt(ncol(betas.subfunc)))*0.5}
  if(is.null(mtry.max)){mtry.max <- floor(sqrt(ncol(betas.subfunc)))} # defaults to sqrt(10000) = 100 
  
  tunegrid <- expand.grid(.mtry = floor(seq(mtry.min, mtry.max, length.out = length.mtry)),
                          .ntree = seq(ntrees.min, ntrees.max, ntree.by), 
                          .nodesize = floor(c(1, 5, nrow(betas.subfunc)*nodesize.proc[[1]], nrow(betas.subfunc)*nodesize.proc[[2]], nrow(betas.subfunc)*nodesize.proc[[3]]))
  )
  # Train - 5x CV (n.cv) for tuning parameters with single repeat (n.rep)
  set.seed(seed, kind = "L'Ecuyer-CMRG")
  rf.tuned.caret.customRF <- train(x = betas.subfunc, y = y.subfunc, method = customRF, metric = "Accuracy", tuneGrid = tunegrid, trControl = control)
  if(print.res){print(rf.tuned.caret.customRF)}
  return(rf.tuned.caret.customRF)
}

# Sys.time() # [1] "2019-01-29 10:08:48 CET"
# subfunc.tRF.testrun <- subfunc_rf_caret_tuner_customRF(y = y[nfolds[[1]][[1]][[1]]$train], betas = betas100[nfolds[[1]][[1]][[1]]$train, ], seed = 1234, ntrees.min = 250, ntrees.max = 500, ntree.by = 250, length.mtry = 2, nodesize.proc = c(0.1, 0.05, 0.1))
# Sys.time() # [1] "2019-01-29 10:10:13 CET"
