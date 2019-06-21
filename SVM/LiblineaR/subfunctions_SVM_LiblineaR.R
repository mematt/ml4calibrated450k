#---------------------------------------------------------------------------
# ml4calibrated450k - Support Vector Machines (SVM) - Utility/Subfunctions 
# 
#                   - Linear Kernel SVM (LiblineaR)
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#--------------------------------------------------------------------------


# Check, install|load recquired packages -----------------------------------------------------------------------------------------------
if (!requireNamespace("doMC", quietly = TRUE)) {
  install.packages("doMC")
  library(doMC) } else {library(doMC)}

if (!requireNamespace("e1071", quietly = TRUE)) { 
  install.packages("e1071")
  library(e1071) } else {library(e1071)}

if (!requireNamespace("LiblineaR", quietly = TRUE)) {
  install.packages("LiblineaR")
  library(LiblineaR) } else {library(LiblineaR)}

## Utility functions ---------------------------------------------------------------------------------------------------------------------------

### Subfunctions 

##########################################
###  Nested CV scheduler / stopper     ###
##########################################

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

#####################################################################
###             SUBFUNCTION - LIBLINEAR - PREDICTOR               ###
### FOR VERSION 1 - DOUBLE FOREACH - Both MODEL TYPE & COST TUNER ###
#####################################################################
# 
# subfunc_SVM_LiblineaR_predictor <- function(selected.bestType.l, selected.bestCost.l, 
#                                             xTrain.scaled.cent, yTrain, xTest, yTest, scale.center.xTest.as.xTrain = T){
#   # Can internally scale & center data 
#   if(scale.center.xTest.as.xTrain) {s2 = scale(xTest, attr(xTrain.scaled.cent, "scaled:center"), attr(xTrain.scaled.cent, "scaled:scale"))} 
#   m <- as.list()
#   mclapply(seq_along(selected.bestType.l), function(i) {
#     bestType.i <- selected.bestType.l[[i]]
#     
#     lapply(seq_along(selected.bestCost.l), fuction(j) {
#      m = LiblineaR(data = xTrain.scaled.cent, 
#                    target = yTrain, 
#                    type = bestType.i, 
#                    cost = bestCost[[j]]) #, bias = 1, verbose = FALSE) # are defaults
#       # Set probability output to FALSE
#       pr = FALSE
#       if(bestType.i == 0 || bestType.i == 7) pr = TRUE # only type 0 & type 2 models can generate probabilities
#       p = predict(m, s2, proba = pr, decisionValues = TRUE)
#       res <- list(mod.fit = m, probs.pred = p)
#       return(res)
#     })
#   })
# }

#############################################################################
### SUBFUNCTION TO RUN LIBLINEAR SVM - COST TUNER with MCLAPPLY           ###
###   for a given model type (type 0)                                     ###
#############################################################################

subfunc_svm_liblinear_train_tuner_mc <- function(data.scaled.xTrain, 
                                                 target.yTrain, 
                                                 mod.type = 0,
                                                 C.base = 10, C.min = -3, C.max = 3,
                                                 bias = 1, #default liblinear setting
                                                 n.CV = 5, 
                                                 verbose = T, 
                                                 seed = 1234, 
                                                 parallel = T, 
                                                 mc.cores){ 
  
  # Cost C grid + give feedback and Sys.time                                     
  Cost.l <- as.list(C.base^(C.min:C.max))
  message("Cost (C) = ", paste(simplify2array(Cost.l), sep = " ", collapse = " ; "),
          " ; \nNr. of CV folds: ", n.CV,
          "\nStart at ", Sys.time())
  # Predefine empty list for results 
  cvfit.liblineaR.C.tuner <- list() 
  
  # Parallelized for Cost tuning
  if(parallel){
    cvfit.liblineaR.C.tuner <- mclapply(seq_along(Cost.l), function(i){
      #message("Tuning LIBLINEAR type 0 - L2-reg. LR with C = ", Cost.l[[i]], " @ ", Sys.time())   
      set.seed(seed+1, kind ="L'Ecuyer-CMRG")
      LiblineaR(data = data.scaled.xTrain, target = target.yTrain, 
                cost = Cost.l[[i]], 
                type = mod.type, 
                bias = bias, cross = n.CV, verbose = verbose)
    }, mc.preschedule = T, mc.set.seed = T, mc.cores = mc.cores)            
    print(Sys.time())                                                       
    return(cvfit.liblineaR.C.tuner)                                         
  } else { # Sequential 
    cvfit.liblineaR.C.tuner <- lapply(seq_along(Cost.l), function(i){
      message("Tuning LIBLINEAR type 0 - L2-reg. LR with C = ", Cost.l[[i]], " @ ", Sys.time())   
      set.seed(seed+1, kind ="L'Ecuyer-CMRG")
      LiblineaR(data = data.scaled.xTrain, target = target.yTrain, type = mod.type, 
                cost = Cost.l[[i]], bias = 1, cross = n.CV, verbose = verbose) 
      # cross = LiblineaR help: if an integer value k>0 is specified, a k-fold cross validation on data is performed 
      #         to assess the quality of the model via a measure of the accuracy. 
      #         Note that this metric might not be appropriate if classes are largely unbalanced. Default is 0.
    })
    print(Sys.time())
    return(cvfit.liblineaR.C.tuner)
  }
}    

#############################################################
### SUBFUNCTION - SVM-LIBLINEAR - Cost parameter SELECTOR ###
#############################################################

# Literature: A practical guide to LIBLINEAR - Fan, Chang, Hsiesh, Wang and Lin 2008 
# <https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf>
# Appendix - N.5. Parameter Selection:
# 1. Solvers in LIBLINEAR are not very sensitive to C. Once C is large than a certain value, the obtained models have similar performances. 
# Theoretical proof: Theorem 3 of Keerthi and Lin (2003)
  

subfunc_svm_liblinear_C_selector <- function(results.cvfit.liblineaR.C.tuner, 
                                             C.base = 10, C.min = -3, C.max = 3, 
                                             n.CV = 5){
    
  Costs.l <- as.list(C.base^(C.min:C.max))
  # Print simplified version for eventual manual selection
  names(results.cvfit.liblineaR.C.tuner) <- paste0("Cost_", Costs.l)
  print(simplify2array(results.cvfit.liblineaR.C.tuner))
  # Chooses the smallest C with highest 5-fold cross-validated accuracy among possible choices
  C.selected <- Costs.l[[which.max(results.cvfit.liblineaR.C.tuner)]] 
  message("\nCost parameter with highest ", n.CV, "-fold CV accuracy : C = ", C.selected, " ; ", 
          "\n If more than one maximal accuracy exists, C returns the smallest cost parameter with highest accuracy")
  return(C.selected)
}

####################################################################
### RE-FIT LINEAR SVM/LR MODEL with TUNED COST on s.xTRAIN       ###
###    THUS get model object to predict on test set              ###
####################################################################

subfunc_svm_liblinear_refit_train <- function(C.tuned, 
                                              data.scaled.xTrain, 
                                              target.yTrain, 
                                              mod.type = 0, 
                                              type4.CramSing = T, 
                                              verbose = T){
  if(type4.CramSing){ 
    # Crammer & Singer 
    # ca. 30 sec # tolerance f type is 1, 3, 4, 7, 12 or 13 # epsilon=0.1
    m.fit.ty4 <- LiblineaR(data = data.scaled.xTrain, target = target.yTrain, type = 4, cost = C.tuned, bias = 1, verbose = verbose)
    
    # L2-regularized Logsitic Regression - L2R_LR by default otherwise mod.type according to LiblineaR
    # 11mins # tolerance if type is 0, 2, 5 or 6 # epsilon=0.01
    m.fit <- LiblineaR(data = data.scaled.xTrain, target = target.yTrain, type = mod.type, cost = C.tuned, bias = 1, verbose = verbose) 
    res <- list(m.fit.mod.type = m.fit, m.fit.ty4.cs = m.fit.ty4)
  } else {
    m.fit <- LiblineaR(data = data.scaled.xTrain, target = target.yTrain, type = mod.type, cost = C.tuned, bias = 1, verbose = verbose)
    res <- list(m.fit.mod.type = m.fit, m.fit.ty4.cs = NULL)
  }
  return(res)
}