#---------------------------------------------------------------------------
# ml4calibrated450k - Support Vector Machines (SVM) - Train function 
# 
#                   - Linear Kernel SVM (LiblineaR)
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#--------------------------------------------------------------------------


### Training & tuning function - integrating the utility/subfuctions from `subfunctions_SVM_LiblineaR.R`

train_SVM_LiblineaR <- function(y, 
                                s.betas.Train, 
                                seed, 
                                n.CV = 5, 
                                multicore = T, 
                                mc.cores, 
                                C.base = 10, C.min = -3, C.max = 3,
                                mod.type = 0, # defaults to type0 L2R_LR
                                type4.CramSing = T, # Crammer & Singer SVC is also fitted 
                                verbose = T){  
  
  ## 1. Crossvalidate SVM/LiblineaR - Cost parameter for optimal - L2R_LR (and Crammer & Singer models) 
  set.seed(seed, kind = "L'Ecuyer-CMRG") 
  message("seed: ", seed)
  message("n: ", nrow(s.betas.Train))
  if(multicore == T && mc.cores > 1) message("Parrallel computing with multicore: ", mc.cores)
  else message("Parralel computing (multicore = F) and/or mc.cores = 1 => Please revise (e.g. define backend)")
  
  model.names.4multiclass.LiblineaR <-c("0 – L2-regularized logistic regression (primal)", 
                                        "1 – L2-regularized L2-loss support vector classification (dual)", 
                                        "2 – L2-regularized L2-loss support vector classification (primal)", 
                                        "3 – L2-regularized L1-loss support vector classification (dual)", 
                                        "4 – support vector classification by Crammer and Singer", 
                                        "5 – L1-regularized L2-loss support vector classification", 
                                        "6 – L1-regularized logistic regression", 
                                        "7 – L2-regularized logistic regression (dual)")
  
  message("\nTuning linear kernel SVM hyperparameter C (cost) for `", 
          model.names.4multiclass.LiblineaR[[mod.type+1]], "`  ... ", Sys.time())
  if(type4.CramSing) message("\nAlso tuning for `", 
                             model.names.4multiclass.LiblineaR[[5]], "`  ... ", Sys.time())
  t1 <- system.time(
    cvfit.svm.liblinear.tuning <- subfunc_svm_liblinear_train_tuner_mc(data.scaled.xTrain = s.betas.Train, 
                                                                       target.yTrain = y, 
                                                                       mod.type = mod.type,  
                                                                       C.base = C.base, C.min = C.min, C.max = C.max, 
                                                                       bias = 1,
                                                                       n.CV = n.CV, 
                                                                       verbose = verbose,
                                                                       seed = seed,
                                                                       parallel = multicore, 
                                                                       mc.cores = mc.cores)
  )
  
  # Extract optimal C (i.e. smallest C with highest accuracy)
  C.tuned.cv <-  subfunc_svm_liblinear_C_selector(results.cvfit.liblineaR.C.tuner = cvfit.svm.liblinear.tuning, 
                                                  C.base = C.base, C.min = C.min, C.max = C.max, 
                                                  n.CV = nfolds)
  
  # Give message with values
  message(paste0("\nOptimal cost (C) parameter: ", C.tuned.cv))
  
  # Refit models on s.xTrain: 
  # - L2R_LR (type0 - ca. 15 min/refit) - gives prob output
  # - CS Crammer & Singer only for class output (ca. +35s)
  message("\nFitting optimal/tuned model on training data ... ", Sys.time())
  t2 <-  system.time(
    modfit.liblinear.train <- subfunc_svm_liblinear_refit_train(C.tuned = C.tuned.cv, 
                                                                data.scaled.xTrain = s.betas.Train, 
                                                                target.yTrain = y,
                                                                mod.type = mod.type, 
                                                                type4.CramSing = type4.CramSing)  
    # default is T for Cram&Sing # our prototyping showed that CS SVC is for all C value better/more accurate than type0
  )                                                                         
  
  # Results
  res <- list(modfit.liblinear.train$m.fit.mod.type, 
              modfit.liblinear.train$m.fit.ty4.cs, # if CS = F => NULL object
              cvfit.svm.liblinear.tuning, 
              C.tuned.cv, 
              t1, t2) 
  return(res)
}