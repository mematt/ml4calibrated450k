#--------------------------------------------------------------------------------
# ml4calibrated450k - Support Vector Machines (SVM) -  Train function  
# 
#                   - Linear Kernel SVM (e1071)
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#-------------------------------------------------------------------------------


### Training & tuning function - integrating the utility/subfuctions from `subfunctions_SVM_e1071.R`

train_SVM_e1071_LK <- function(y, betas.Train, 
                               seed, 
                               mc.cores, 
                               nfolds = 5, 
                               C.base = 10, C.min = -3, C.max = 3, 
                               scale.internally.by.e1071.svm = T,
                               mod.type = "C-classification"){  
  
  ## 1. Crossvalidate SVM/LiblineaR - Cost parameter for optimal
  set.seed(seed, kind = "L'Ecuyer-CMRG") 
  message("seed: ", seed)
  message("n: ", nrow(betas.Train))  # n_patients
  
  message("\nTuning SVM (e1071) linear kernel: hyperparameter C (cost) ... ", Sys.time())
  t1 <- system.time(
    cvfit.svm.e1071.linear.C.tuner <- subfunc_svm_e1071_linear_train_tuner_mc(data.xTrain = betas.Train, 
                                                                              target.yTrain = y,
                                                                              mod.type = mod.type, 
                                                                              kernel. = "linear", 
                                                                              scale. = scale.internally.by.e1071.svm,
                                                                              C.base = C.base, 
                                                                              C.min = C.min, C.max = C.max,
                                                                              n.CV = 5, 
                                                                              verbose = T, 
                                                                              seed = seed,
                                                                              parallel = T, 
                                                                              mc.cores = mc.cores)
  )
  
  # Extract optimal C or smallest C with highest accuracy 
  C.tuned.cv <-  subfunc_svm_e1071_linear_C_selector(results.cvfit.e1071.linear.C.tuner = cvfit.svm.e1071.linear.C.tuner, 
                                                     C.base = C.base, C.min = C.min, C.max = C.max, n.CV = nfolds, verbose = T)
  # C.tuned.cv = list of 3: $C.selected $mtx.accuracies.nCV $mtx.accuracies.mean
  # Provide message with value
  message(paste0("Optimal cost (C) parameter: ", C.tuned.cv$C.selected))
  
  # Refit models on s.xTrain L2R_LR (type0 - ca. 15 min/refit) and optionally only for classes Crammer & Singer (type4 - it takes just ca. +35s)
  message("\n(Re)Fitting optimal/tuned model on training data ... ", Sys.time())
  t2 <-  system.time(
    modfit.svm.linear.train <- subfunc_svm_e1071_linear_modfit_train(C.tuned = C.tuned.cv$C.selected, 
                                                                     data.xTrain = betas.Train, 
                                                                     target.yTrain = y, 
                                                                     results.cvfit.e1071.linear.C.tuner = cvfit.svm.e1071.linear.C.tuner, 
                                                                     C.selector.accuracy.mean = C.tuned.cv$mtx.accuracies.mean, 
                                                                     use.fitted = T)
    # uses predict supposed to scale data.xTrain / betas.Train automatically
    
  )
  # CAVE conames order is not the same as in levels(y) !!!
  pred.scores.trainfit.svm.lin1 <- attr(modfit.svm.linear.train$trainfit.svm.lin1, "probabilities") 
  
  # Results
  res <- list(modfit.svm.linear.train$svm.e1071.model.object,      # svm linear model object used in predict/fitted
              modfit.svm.linear.train$trainfit.svm.lin1,           # fitting train with predict.svm()/predict()
              pred.scores.trainfit.svm.lin1,                       # CAVE: colnames order is not the same as in levels(y) !!!
              modfit.svm.linear.train$trainfit.svm.lin2,           # fitting train with fitted() - as in the examples of svm{e1071} help
              cvfit.svm.e1071.linear.C.tuner,                      # list of 7: model objects of svm based on Cost tuner grid 10^(-3:3)
              C.tuned.cv,                                          # list of 3: $C.selected $mtx.accuracies.nCV $mtx.accuracies.mean 
              t1, t2) 
  return(res)
}
