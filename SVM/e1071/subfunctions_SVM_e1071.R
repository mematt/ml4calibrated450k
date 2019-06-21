#--------------------------------------------------------------------------------
# ml4calibrated450k - Support Vector Machines (SVM) -  Utility/subfunctions  
# 
#                   - Linear Kernel SVM (e1071)
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#-------------------------------------------------------------------------------

# Check, install|load recquired packages -----------------------------------------------------------------------------------------------
if (!requireNamespace("caret", quietly = TRUE)) { 
  install.packages("caret", dependencies = T)
  library(caret) 
} else {library(caret)}

if (!requireNamespace("doMC", quietly = TRUE)) {
  install.packages("doMC")
  library(doMC) } else {library(doMC)}

if (!requireNamespace("e1071", quietly = TRUE)) { 
  install.packages("e1071")
  library(e1071) } else {library(e1071)}

# if (!requireNamespace("LiblineaR", quietly = TRUE)) {
#   install.packages("LiblineaR")
#   library(LiblineaR) } else {library(LiblineaR)}


# Utility functions ---------------------------------------------------------------------------------------------------------------------------

## Nested CV scheduler / stopper ---------------------------------------------------------------------------------------------------------------------------
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

## Cost tuner subfunction ---------------------------------------------------------------------------------------------------------------------------
subfunc_svm_e1071_linear_train_tuner_mc <- function(data.xTrain, target.yTrain, 
                                                    mod.type = "C-classification", 
                                                    kernel. = "linear", 
                                                    scale. = T,
                                                    C.base = 10, C.min = -3, C.max = 3, 
                                                    n.CV = 5, verbose = T, 
                                                    seed = 1234, 
                                                    parallel = T, 
                                                    mc.cores = 4L){ 
  
  # Cost C grid + give feedback and Sys.time                                     
  Cost.l <- as.list(C.base^(C.min:C.max))
  message("\nCost (C) = ", paste(simplify2array(Cost.l), sep = " ", collapse = " ; "), 
          " ; \nNr. of iterations: ", length(Cost.l),
          "\nStart at ", Sys.time())
  # Predefine empty list for results 
  cvfit.e1071.linear.C.tuner <- list() 
  
  # Parallel 
  if(parallel){
    cvfit.e1071.linear.C.tuner <- mclapply(seq_along(Cost.l), function(i){   # uses only lenght(Cost.l) workers
      set.seed(seed+1, kind ="L'Ecuyer-CMRG")
      svm(x = data.xTrain, y = target.yTrain, scale = scale., type = mod.type,
          kernel = kernel., 
          cost = Cost.l[[i]], 
          cross = n.CV, 
          probability = T, 
          fitted = T) # gamma: default 1/n.features; tolerance: default: 0.001) 
    }, mc.preschedule = T, mc.set.seed = T, mc.cores = mc.cores)
    print(Sys.time()) 
    return(cvfit.e1071.linear.C.tuner)
    
    # Sequential                                       
  } else {  
    cvfit.e1071.linear.C.tuner <- lapply(seq_along(Cost.l), function(i){
      cat("\nTuning e1071 with linear kernel function C = ", Cost.l[[i]], " @ ", Sys.time())   
      set.seed(seed+1, kind ="L'Ecuyer-CMRG")
      svm(x = data.xTrain, y = target.yTrain, scale = scale., type = mod.type, 
          kernel = kernel., cost = Cost.l[[i]], cross = n.CV, probability = T, fitted = T)
    })
    print(Sys.time())
    return(cvfit.e1071.linear.C.tuner)
  }
}    

## Cost selector subfunction ---------------------------------------------------------------------------------------------------------------------------
subfunc_svm_e1071_linear_C_selector <- function(results.cvfit.e1071.linear.C.tuner, 
                                                C.base = 10, C.min = -3, C.max = 3, 
                                                n.CV = 5, verbose = T){
  
  Costs.l <- as.list(C.base^(C.min:C.max))
  # Print simplified version of each crossvalidated fold accuracy for eventual manual selection
  res.cvfit.svm.accuracies.nCV <- sapply(seq_along(C.base^(C.min:C.max)), function(i){
    simplify2array(results.cvfit.e1071.linear.C.tuner[[i]]$accuracies)})
  colnames(res.cvfit.svm.accuracies.nCV) <- paste0("Cost_", Costs.l)
  rownames(res.cvfit.svm.accuracies.nCV) <- paste0("nCV", seq(1, n.CV, 1))
  # Print matrix of all CV accuracies
  if(verbose){
    message("\nMatrix of all CV accuracies:")
    print(res.cvfit.svm.accuracies.nCV)
  }
  
  # Average accuracy 
  res.cvfit.svm.accuracies.mean <- sapply(seq_along(C.base^(C.min:C.max)), function(i){
    simplify2array(results.cvfit.e1071.linear.C.tuner[[i]]$tot.accuracy)})
  names(res.cvfit.svm.accuracies.mean) <- paste0("Cost_", Costs.l)
  # Same as: res.cvfit.svm.accuracies.mean <- apply(res.cvfit.svm.accuracies.nCV, 2, mean)
  # Print list of average CV accuracies/ $tot.accuracy
  if(verbose){
    message("\nMean CV accuracies:")
    print(res.cvfit.svm.accuracies.mean)
  }
  
  # Selection
  # Chooses the smallest C with highest 5-fold cross-validated accuracy among possible choices 
  # => if C is large enough anyway (see Appendix-N.5.) doesnt make a difference 
  # => saves also computation time if C is smaller # => error-margin/Nr of supp.vecs.
  C.selected <- Costs.l[[which.max(res.cvfit.svm.accuracies.mean)]] 
  message("\nCost parameter with highest ", n.CV, "-fold CV accuracy : C = ", C.selected, " ; ", 
          "\n Note: If more than one maximal accuracy exists, C returns the smallest cost parameter with highest accuracy.", 
          "\n Once C is large than a certain value, the obtained models have similar performances", 
          " (for theoretical proof see Theorem 3 of Keerthi and Lin, 2003)")
  res <- list(C.selected = C.selected, 
              mtx.accuracies.nCV = res.cvfit.svm.accuracies.nCV, 
              mtx.accuracies.mean = res.cvfit.svm.accuracies.mean)
  return(res)
  
  # Literature:
  # Important # A practical guide to LIBLINEAR - Fan, Chang, Hsiesh, Wang and Lin 2008 
  # <https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf>
  # Appendix-N.5. Parameter Selection:
  # 1. Solvers in LIBLINEAR are not very sensitive to C. Once C is large than a certain value, 
  # the obtained models have similar performances. 
  # Theoretical proof: Theorem 3 of Keerthi and Lin (2003)
}

## Re-fit training data subfunction ---------------------------------------------------------------------------------------------------------------------------

subfunc_svm_e1071_linear_modfit_train <- function(C.tuned, 
                                                  data.xTrain, target.yTrain,
                                                  results.cvfit.e1071.linear.C.tuner, 
                                                  C.selector.accuracy.mean, 
                                                  use.fitted = T){  #res.svm.C.tuner.l
  
  message("\n\nRe-fitting training data ... ", Sys.time())
  #Costs.l <- as.list(C.base^(C.min:C.max))
  i <- which.max(C.selector.accuracy.mean)
  
  # Ver.1 - Use predict function to refit training data
  # Note If the training set was scaled by svm (done by default), 
  # the new data is scaled accordingly using scale and center of the training data.
  modfit.train.svm.lin.pred <- predict(object = results.cvfit.e1071.linear.C.tuner[[i]], 
                                       newdata =  data.xTrain,
                                       decision.values = T,   
                                       # decision values of all binary classif. in multiclass setting are returned.
                                       probability = T)
  
  # Ver.2 - Use fitted() - see ??svm - Examples
  if(use.fitted){modfit.train.svm.lin.fitted <- fitted(results.cvfit.e1071.linear.C.tuner[[i]])}
  #message("\nBoth predict() & fitted() are ready @ ", Sys.time())
  message("\nPrediction is ready @ ", Sys.time())
  
  # Output
  res <- list(svm.e1071.model.object = results.cvfit.e1071.linear.C.tuner[[i]], 
              trainfit.svm.lin1 = modfit.train.svm.lin.pred, 
              trainfit.svm.lin2 = modfit.train.svm.lin.fitted) # => output file is rel. large / lot of large mtx.
  return(res)
}