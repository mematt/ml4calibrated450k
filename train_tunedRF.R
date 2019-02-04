
# Source required R scripts
source("subfunctions_tunedRF.R")
source("misclassification error.R")
#source("mAUC.R")
source("brier.R")
source("mlogloss.R")
# Import packages
library(caret)
library(randomForest)



############################################################
###     RF HYPER PARAMETER TUNING                        ###
###         using CARET customRF() FUNCTION              ###
###                                                      ###
###     trainRF_caret_custom_tuner()                     ###
############################################################


trainRF_caret_custom_tuner <- function(y., betas.,
                                       cores = 4L, # some variation of parallel backend needs to be defined
                                       mtry.min = NULL, mtry.max = NULL, length.mtry = 6,
                                       ntrees.min = 1000, ntrees.max = 2000, ntree.by = 500,
                                       nodesize.proc = c(0.01, 0.05, 0.1),
                                       n.cv = 5, n.rep = 1,
                                       p = c(100, 500, 1000, 10000),  # number of values in p should not be more than the number of available cores/threads in the system
                                       p.tuning.brier = T, p.tuning.miscl.err = T, p.tuning.mlogloss = T,
                                       seed, 
                                       allowParallel = T){
  
  # Info block about settings
  set.seed(seed, kind = "L'Ecuyer-CMRG") 
  message("seed: ", seed)
  message("cores: ", cores)
  
  message("\nRun 1.: Random Forest - Hyper parameter tuning with the following number of parameters : ", 
          "\nlength.mtry = ", length.mtry, 
          "; \nntree = ", paste(seq(ntrees.min, ntrees.max, by = ntree.by), sep = ";", collapse = ", "), 
          "; \nlength.nodesize = ", length(c(1, 5, nodesize.proc)),  # default classification =1, def. regression = 5, Kruppa et al. = 10%
          "; \nTotal GRID size: ", length.mtry*length(seq(ntrees.min, ntrees.max, by = ntree.by))*length(c(1, 5, nodesize.proc)), 
          "; \nUsing custom function in {caret} with repeated CV (n.cv = ", n.cv, "; n.rep = ", n.rep, ") ... ", Sys.time())
  
  # Run 1 - Tune RF hyperparameters using grid search (tuning measure = "Accuracy")
  t0 <- system.time(
    # Uses the customRF() function using grid search of the caret training/tuning framework
    rf.hypar.tuner <- subfunc_rf_caret_tuner_customRF(y.subfunc = y., betas.subfunc = betas., 
                                                      mtry.min = mtry.min, mtry.max = mtry.max, length.mtry = length.mtry, 
                                                      ntrees.min = ntrees.min, ntrees.max = ntrees.max, ntree.by = ntree.by,
                                                      nodesize.proc = nodesize.proc,
                                                      n.cv = n.cv, n.rep = n.rep,
                                                      seed = seed, 
                                                      allowParall = allowParallel,
                                                      print.res = T)
    
    
  )
  # Output hyperparameter results in the console
  message("Hyperparameter Tuning Results:", 
          "\n bestTune mtry: ", rf.hypar.tuner$bestTune$mtry, 
          "\n bestTune ntree: ", rf.hypar.tuner$bestTune$ntree, 
          "\n bestTune nodesize: ", rf.hypar.tuner$bestTune$nodesize)
  
  # Run 2 - Fit RF with best tuning parameters for variable selection
  message("\nVariable selection using random Forest with tuned hyperparameters ... ", Sys.time())
  message("Multi-core (available/assigned nr. of threads): ", cores)
  t1 <- system.time(
    rf.varsel <- rfp(xx = betas., y = y.,
                     mc = cores,
                     mtry = rf.hypar.tuner$bestTune$mtry,
                     ntree = rf.hypar.tuner$bestTune$ntree,
                     nodesize = rf.hypar.tuner$bestTune$nodesize,
                     strata = y.,
                     sampsize = rep(min(table(y.)),length(table(y.))), # downsampling to the minority class # smallest class in betas = 8 cases | 4-6 cases
                     importance = TRUE,                              # permutation based mean deacrease in accuracy 
                     replace = FALSE)                                # without replacement see Strobl et al. BMC Bioinf 2007
  )                                                                  # http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25
  
  
  # Extract permutation based importance measure
  message("Variable importance and subsetting of methylation data (betas) using p = ", paste(p, collapse = "; "), " ... ", Sys.time())
  imp.perm <- importance(rf.varsel, type = 1)
  
  # Variable selection ("p" most important variables are selected)
  p.l <- as.list(p)
  message("Reordering betas using multi-core: ", length(p.l))
  or <- order(imp.perm, decreasing = T)
  # Reorder - using mutli-core
  betasy.l <- mclapply(seq_along(p.l), function(i){
    p.l.i <- p.l[[i]]
    betas.[ , or[1:p.l.i]]
  }, mc.preschedule = T, mc.set.seed = T, mc.cores = length(p.l))
  
  
  message("Prediction using 'p' subset of most important variables ... ", Sys.time())
  message("p: ", paste(p, collapse = "; "))  
  message("Multi-core: ", length(betasy.l))
  
  
  set.seed(seed+1, kind = "L'Ecuyer-CMRG")
  t2 <- system.time(
    rf.pred.l <- mclapply(seq_along(betasy.l), function(i){
      randomForest(x = betasy.l[[i]], y = y.,
                   mtry = rf.hypar.tuner$bestTune$mtry,
                   ntree = rf.hypar.tuner$bestTune$ntree,
                   nodesize = rf.hypar.tuner$bestTune$nodesize,
                   strata = y.,
                   sampsize = rep(min(table(y.)), length(table(y.))), # downsampling to the minority class # smallest class in betas = 8 cases | 4-6 cases
                   proximity = TRUE,
                   oob.prox = TRUE,
                   importance = TRUE, # permutation based mean deacrease in accuracy 
                   keep.inbag = TRUE,
                   do.trace = FALSE,
                   seed = seed)
    }, mc.preschedule = T, mc.set.seed = T, mc.cores = length(betasy.l)) # number of cores/threads == number of p variables
  )
  # str(rf.pred.l) # list of 2 # rf objects
  message("\nRun time of training classifier: ")
  print(t2)
  
  # Refit rf.pred.l models on the training data
  score.pred.l <- mclapply(seq_along(rf.pred.l), function(i){
    predict(object = rf.pred.l[[i]], 
            newdata = betas.[ , rownames(rf.pred.l[[i]]$importance)],
            type = "prob")
  },  mc.preschedule = T, mc.set.seed = T, mc.cores = length(rf.pred.l))
  
  # Performance Evaluators for "p" number of predictors tuning
  # Brier score (BS)
  if(p.tuning.brier){
    brier.p.l <- lapply(seq_along(score.pred.l), function(i){
      brier(scores = score.pred.l[[i]], y = y.)
    })
    message("Brier scores: ", paste(brier.p.l, collapse = "; "))
  }
  # Misclassification Error (ME)
  if(p.tuning.miscl.err){
    err.p.l <- lapply(seq_along(score.pred.l), function(i){
      sum(colnames(score.pred.l[[i]])[apply(score.pred.l[[i]], 1, which.max)] != y.) / length(y.)
    })
    message("Misclassif. error on training set (p most important variable): ", paste(err.p.l, collapse = "; "))
  }
  # Multiclass logloss (mLL)
  if(p.tuning.mlogloss){
    mlogloss.p.l <- lapply(seq_along(score.pred.l), function(i){
      mlogloss(scores = score.pred.l[[i]], y = y.)
    })
    message("Multiclass logloss: ", paste(mlogloss.p.l, collapse = "; "))
  }
  # Output optimal "p" number of features (CpG probes) respective to Brier score, log loss and miscl. error
  message("Optimal number of predictor variables:  p(brier) = ", p[[which.min(brier.p.l)]])
  message("Optimal number of predictor variables:  p(mlogl) = ", p[[which.min(mlogloss.p.l)]])
  message("Optimal number of predictor variables:  p(miscl.err) = ", p[[which.min(err.p.l)]])
  
  # Best RF model object  
  message("\nAll p(brier), p(mlogl) and p(miscl.err) models are going to be written into the return object ... ", Sys.time())
  rf.pred.best.brier <- rf.pred.l[[which.min(brier.p.l)]]
  rf.pred.best.err <- rf.pred.l[[which.min(err.p.l)]]
  if(p[[which.min(brier.p.l)]] == p[[which.min(mlogloss.p.l)]]){rf.pred.best.mlogl <- rf.pred.best.brier} # if brier and mlogloss are the same 
  else{rf.pred.best.mlogl <- rf.pred.l[[which.min(mlogloss.p.l)]]}
  
  # Results
  res <- list(rf.pred.best.brier, rf.pred.best.err, rf.pred.best.mlogl, 
              imp.perm, rf.pred.l,
              score.pred.l, brier.p.l, err.p.l, mlogloss.p.l,
              t0, t1, t2)
  return(res)
}

# Test run call
# Sys.time() # [1] "2019-01-29 10:22:36 CET"
# test.run.train.tRF.caret <- trainRF_caret_custom_tuner(y = y[nfolds[[1]][[1]][[1]]$train], betas = betas100[nfolds[[1]][[1]][[1]]$train, ],
#                                                  cores = cores, p = c(50, 100), seed = seed+1, allowParallel = T)
# Sys.time() #[1] "2019-01-29 10:44:35 CET"
