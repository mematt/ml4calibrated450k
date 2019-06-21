#--------------------------------------------------------------------
# ml4calibrated450k - XGBOOST - Train function 
#
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-28
#--------------------------------------------------------------------


## 2. Training & tuning using the caret framework ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# The combination of best performing hyperparameter settings and default settings of `xgboost` parameters were investigated within a 3x-fold CV using the framework of the `caret` package. 
# 
# 1. The grid below is searched for highest accuracy. These parameters achieved the lowest misclassification error during prototyping on the large search grid in **Table. 2.**.  
# 
# `nrounds` = 100 (default) # xgboost converges very fast <80 iterations, however in some cases probably >100 would have yielded better results.
# `max_depth` = 6 (default) # proved to be the optimal setting
# `eta` = c(0.1, 0.3),
# `gamma` = c(0, 0.01)
# `colsample_bytree` = c(0.01, 0.02, 0.05 , 0.2)
# `min_child_weight` = 1 (default) 
# `subsample` = 1 (default)
# 
# GRID size: 1 x 2 (eta) x 2 (gamma) x 4 (colsamp) = 16  # optimized for => 16 x 4 hyperthread => 64vCPUs or larger (e.g. c5n.18xlarge 72vCPU) AWS EC2 instances. 
# 
# 2. The best cross-validated model settings (found by `caret`) are passed to the `xgboost::xgb.train()` function. 
#    Thereby exploiting the `xgb.DMatrix` structure and concurrent train-test `watchlist` functionality. 
#    In this step the nrouns = 100 is searched for optimal `n_iterations`. 
# 
#    The model object with best n_iterations (2-3Mb) is saved separately for later use.
# 
# 3. The final object is then refitted on the *K.k-fold training set* and also on the *test set*. All generated model/output objects are saved. 


### Define training & tuning function ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

trainXGBOOST_caret_tuner <- function(y, 
                                     train.K.k.mtx,            # for caret::train() use dense matrix train.K.k.mtx
                                     K., k.,                   # foldIDs of 5x5 nested CV
                                     dtrain., watchlist.,      # caret crashes if dtrain used within train(); these are for xgb.train() 
                                     n.CV = 3, n.rep = 1,      # caret CV: 3x CV takes 7-8hs for CV.1.1! 
                                     seed., 
                                     allow.parallel = T,
                                     mc.cores = 4L, # xgb.nthread (openmp hyperthreading in Linux),
                                     max_depth. = 6, 
                                     eta. = c(0.1, 0.3), 
                                     gamma. = c(0, 0.01),
                                     colsample_bytree. = c(0.01, 0.02, 0.05, 0.2), 
                                     minchwght = 1,
                                     subsample. = 1,
                                     nrounds. = 100,           # use default to limit computational burden 
                                     early_stopping_rounds. = 50,
                                     objective. = "multi:softprob", 
                                     eval_metric. = "merror",  # eval_metric = "mlogloss" # both can be provided at the same time/run; but then xgb.train uses mlogloss instead of merror;
                                                               # merror yielded consistently better results than mlogloss
                                     save.xgb.model = T, 
                                     out.path.model.folder = "xgboost-train-best-model-object", 
                                     save.xgb.model.name = "xgboost.model.train.caret"){   
  
  ## 1. 
  set.seed(seed., kind = "L'Ecuyer-CMRG") 
  message("seed: ", seed.)
  message("n: ", nrow(dtrain.))  # n_patients # same as nrow(train.K.k.mtx)
  
  # Security check for nested multicores/threads
  colsamp.l <- as.list(colsample_bytree.) 
  #if(length(colsamp.l)*xgb.nthread > detectCores()){stop("Planned number of multicores & threads is more than physically available.")}
  
  # CARET - Grid
  xgbGrid <- expand.grid(nrounds = nrounds.,      # optimal nrounds with d=6 from prototyping 107! or smaller 62, 81, 90, 102
                         max_depth = max_depth.,  # 6 was better than 2, 3, 4, 5, 8, 10
                         eta = eta.,              # 0.1 <- was better than 0.05 0.3 
                         gamma = gamma.,          # 0 was better than 2, 1, 0.1, 0.05, 0.001
                         colsample_bytree = colsample_bytree.,  
                         min_child_weight = minchwght,
                         subsample = subsample.)
  
  message("\nTuning grid size: ", nrow(xgbGrid))
  
  # CARET - trControl
  xgbTrControl <- trainControl(
    method = "repeatedcv",
    number = n.CV,
    repeats = n.rep,
    verboseIter = TRUE,
    returnData = FALSE,
    classProbs = F, # T, # make.names() error by caret
    summaryFunction = multiClassSummary,
    allowParallel = allow.parallel    # it crashes less and ca. 5% faster if caret handels parallelization # otherwise nthread = detectCores()/nrow(xgbGrid)
  )
  
  # CARET - train 
  Sys.time()
  set.seed(seed = seed., kind = "L'Ecuyer-CMRG")
  t1 <- system.time(
    xgb.Train.caret.res <- train(x = train.K.k.mtx,  # caret::source code => converts dense matrix internally to xgb.DMatrix (faster iteration)
                                 y = y,              # using R conform outcome/labels make.names(y) => classProbs = T should be possible, however doesnt work well - crashes!
                                 method = "xgbTree",
                                 trControl = xgbTrControl,
                                 tuneGrid = xgbGrid
                                 #nthread = xgb.nthread 
                                 # (as above) let caret set nthread based on nthread = detectCores()/nrow(xgbGrid);
    )
  )
  message("\nRuntime of caret grid search:")
  print(t1)
  message("\nResults of caret grid search:")
  print(xgb.Train.caret.res); # str(xgb.Train.caret.res)
  
  # Rerun best model => Switch to xgboost - xgb.train
  param.xgb.train <- list(max_depth = xgb.Train.caret.res$bestTune$max_depth,
                          eta = xgb.Train.caret.res$bestTune$eta, 
                          gamma = xgb.Train.caret.res$bestTune$gamma,
                          colsample_bytree = xgb.Train.caret.res$bestTune$colsample_bytree,      
                          min_child_weight = xgb.Train.caret.res$bestTune$min_child_weight, # controls the hessian
                          subsample = xgb.Train.caret.res$bestTune$subsample,
                          num_class = length(levels(y)), # 91 
                          objective = "multi:softprob", 
                          eval_metric = "merror",
                          nthread = mc.cores)   #allow all cores! 
  
  # Crossvalidate nrounds - takes too long -> commented out
  #xgbcv.merr. <- xgb.cv(params = param.xgb.train, data = dtrain, nrounds = 110, nfold = 5, stratified = T, prediction = T, showsd = T, verbose = T, early_stopping_rounds = 50, maximize = F)
  
  # Create output directory - for saving xgb.train model objects for eventual later loading/use
  folder.path <- file.path(getwd(), out.path.model.folder)
  dir.create(folder.path, recursive = T, showWarnings = F)
  # Saving path and object name
  save.xgb.model.train.name <- file.path(folder.path, paste(save.xgb.model.name, "train.autosave", K., k., sep = "."))
  
  # Re-Run xgb.train to find optimal niter/nrounds & concurrently check test error => get Model object
  message("Re-Run xgb.train to find optimal niter/nrounds & concurrently check test error & save model object @ ", Sys.time())
  xgb.Train.model.caret.best <-  xgb.train(params = param.xgb.train, 
                                           data = dtrain.,
                                           nrounds = xgb.Train.caret.res$bestTune$nrounds, 
                                           watchlist = watchlist., 
                                           verbose = 1, print_every_n = 1L, # verbose = 2 is pretty crazy but very insightful - trees are mostly pruned back to depth 2-3
                                           early_stopping_rounds = early_stopping_rounds., 
                                           maximize = F, 
                                           save_period = 0,   # at the end
                                           save_name = save.xgb.model.train.name, 
                                           xgb_model = NULL)                # a previously built model to continue the training from; Raw or xgb.Booster;
  message("\nBest {caret} CV model - performance using `xgb.train()`: ")
  print(xgb.Train.model.caret.best, verbose = T)
  
  message("\n(Re)Fitting tuned model with optimal nrounds/ntreelimit on training data ... ", Sys.time())
  scores.pred.xgboost.vec.train <- predict(object = xgb.Train.model.caret.best, 
                                           newdata = dtrain.,
                                           ntreelimit = xgb.Train.model.caret.best$best_iteration, 
                                           outputmargin = F)
  
  # Generate a matrix from the vector of probabilities
  scores.xgboost.tuned.train <- matrix(scores.pred.xgboost.vec.train, nrow = nrow(dtrain.), 
                                       ncol = length(levels(y)), byrow = T)
  # Reassign row & colnames
  rownames(scores.xgboost.tuned.train) <- rownames(train.K.k.mtx)
  colnames(scores.xgboost.tuned.train) <- levels(y)
  
  # Results
  res <- list(xgb.Train.model.caret.best,    # best model object used in predict
              scores.xgboost.tuned.train,    # predicted raw scores of the training data
              xgb.Train.caret.res,           # caret CV optimized obj. with best model     
              t1) 
  
  return(res)
}