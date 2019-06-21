#---------------------------------------------------------------------------
# ml4calibrated450k - Support Vector Machines (SVM) - Utility/Subfunctions 
# 
#                   - GPU-accelerated, Linear Kernel SVM (Rgtsvm)
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#--------------------------------------------------------------------------

# Nested CV scheduler / stopper -----------------------------------------------------------------------------------------------
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

# my_func_train.LIN.SVM.GPU_v2 -----------------------------------------------------------------------------------------------
# Version 2 - no time measurment - maybe screws up within a for cycle 
my_func_train.LIN.SVM.GPU_v2 <- function(x.mtx2train, y.fac.train, 
                                         scale.x.mtx = T, 
                                         cost.list = as.list(10^(-5:-2)), 
                                         n.cross = 5, 
                                         probs = T, 
                                         GPU.id = 0, 
                                         verb = T){
  
  message("\nStart fitting Cost (C) values @ ", Sys.time())
  l.SVM.LIN.TRAIN.mod.obj.C.i <- lapply(seq_along(cost.list), function(i){
    model.Rgtsvm.train.C.i <- Rgtsvm::svm(x = x.mtx2train, 
                                          y = y.fac.train, 
                                          scale = scale.x.mtx, 
                                          type = "C-classification", 
                                          kernel = "linear" , 
                                          cost = cost.list[[i]],
                                          cross = n.cross,
                                          probability = probs, 
                                          gpu.id = GPU.id, # gpu.id = 0 # 1080Ti
                                          verbose = verb)
  })
  message("Finished fitting Cost values @ ", Sys.time())
  # l.res.train.Rgtsvm.C.i <- list(l.mod.obj.LIN.SVM.Rgtsvm = l.SVM.LIN.TRAIN.mod.obj.C.i,
  #                                time2fit.cost.list = t.SVM.LIN.full.cost.l.run)
  # return(l.res.train.Rgtsvm.C.i)
  return(l.SVM.LIN.TRAIN.mod.obj.C.i)
}

# my_func_predictor_error -----------------------------------------------------------------------------------------------
my_func_predictor_error <- function(object.svm, data2pred, prob=T, verb=T, y.factor.levels=y.test.1.1, betas.df=betas, nfolds.str=nfolds, K=1, k=1, inner=2, y.K.k=y.test.1.1){
  # Predict 
  pred.fit.func <- predict(object = object.svm, newdata = data2pred, probability = prob, verbose = verb)
  message("Note: \n predict.gtsvm {Rgtsvm}: \n If the training data was scaled by the svm calling", 
          " this function shall scale the test data accordingly using scale and center attributes of the training data.")
  # OTHERWISE 
  # s.betas10k.Test <- scale(betas10k[fold$test, ], attr(s.betas10k.Train, "scaled:center"), attr(s.betas10k.Train, "scaled:scale"))
  # str(pred.fit.func)
  # Get the attribute (i.e. metadata) "probabilities" which is a matrix
  mtx.pred.fit.probs <- attr(pred.fit.func, which = "probabilities") # mtx  
  # Check rownames - later for performance evaluation and calibration important
  message(paste("\nRownames are identical to initial fold.K.k ordering: ",
                identical(rownames(mtx.pred.fit.probs), rownames(betas.df[nfolds.str[[K]][[inner]][[k]]$test, ]))))
  colnames(mtx.pred.fit.probs) <- levels(y.factor.levels)
  ### Predict labels ###
  y.pred.test.K.k <- colnames(mtx.pred.fit.probs)[apply(mtx.pred.fit.probs, 1, which.max)]
  err.pred <- sum(y.pred.test.K.k!=y.K.k)/length(y.K.k)
  message("\nMisclassification error: ", err.pred)
  l.res.pred <- list(pred.svm.fit.obj = pred.fit.func, 
                     mtx.pred.probs = mtx.pred.fit.probs, 
                     y.pred.test.K.k = y.pred.test.K.k, 
                     err.pred.test.K.k = err.pred)
  return(l.res.pred)
}

# my_func_predictor_error_nestedCV -----------------------------------------------------------------------------------------------
my_func_predictor_error_nestedCV <- function(object.svm, data2pred, 
                                             prob=T, 
                                             verb=T,
                                             verb.messages = F,
                                             y.factor.levels, 
                                             betas.df, 
                                             fold.str, 
                                             y.K.k){
  # Predict 
  pred.fit.func <- predict(object = object.svm, newdata = data2pred, probability = prob, verbose = verb)
  if(verb.messages) message("<NOTE>: predict.gtsvm {Rgtsvm}: if the training data was scaled by the svm calling", 
                            " this function shall scale the test data accordingly using scale and center of the training data.")
  mtx.pred.fit.probs <- attr(pred.fit.func, which = "probabilities") # mtx  
  # Check rownames - later for performance evaluation and calibration important
  if(verb.messages) message(paste("Rownames are identical to initial fold.K.k ordering: ", 
                                  identical(rownames(mtx.pred.fit.probs), rownames(betas.df))))
  colnames(mtx.pred.fit.probs) <- levels(y.factor.levels)
  ### Predict labels ###
  y.pred.test.K.k <- colnames(mtx.pred.fit.probs)[apply(mtx.pred.fit.probs, 1, which.max)]
  err.pred <- sum(y.pred.test.K.k != y.K.k)/length(y.K.k)
  message("Misclassification error: ", err.pred)
  l.res.pred <- list(pred.svm.fit.obj = pred.fit.func, 
                     mtx.pred.probs = mtx.pred.fit.probs, 
                     y.pred.test.K.k = y.pred.test.K.k, 
                     err.pred.test.K.k = err.pred)
  return(l.res.pred)
}

# my_func_predictor_error_CV_tuner -----------------------------------------------------------------------------------------------
my_func_predictor_error_CV_tuner <- function(object.svm, data2pred, prob=T, verb=T, y.factor.levels=y.test.1.1, 
                                             x.test.CV.Fold.i, y.test.CV.Fold.i=y.test.1.1){
  # Predict 
  pred.fit.func <- predict(object = object.svm, newdata = data2pred, probability = prob, verbose = verb)
  if(verb.messages) message("<NOTE>: predict.gtsvm {Rgtsvm}: if the training data was scaled by the svm calling", 
                            " this function shall scale the test data accordingly using scale and center of the training data.")
  str(pred.fit.func)
  # Get the attribute (i.e. metadata) "probabilities" which is a matrix
  mtx.pred.fit.probs <- attr(pred.fit.func, which = "probabilities") # mtx  
  # Check rownames - later for performance evaluation and calibration important
  message(paste("\nRownames are identical to initial fold.K.k ordering: ",
                identical(rownames(mtx.pred.fit.probs), rownames(x.test.CV.Fold.i))))
  colnames(mtx.pred.fit.probs) <- levels(y.factor.levels)
  ### Predict labels ###
  y.pred.test.K.k <- colnames(mtx.pred.fit.probs)[apply(mtx.pred.fit.probs, 1, which.max)]
  err.pred <- sum(y.pred.test.K.k!=y.test.CV.Fold.i)/length(y.test.CV.Fold.i)
  message("\nMisclassification error: ", err.pred)
  l.res.pred <- list(pred.svm.fit.obj = pred.fit.func, 
                     mtx.pred.probs = mtx.pred.fit.probs, 
                     y.pred.test.CV.fold.i = y.pred.test.K.k, 
                     err.pred.test.CV.fold.i = err.pred)
  return(l.res.pred)
}


# my_func_select_cost_value_LIN.SVM -----------------------------------------------------------------------------------------------
my_func_select_cost_value_LIN.SVM <- function(pred.err.LIN.SVM.list, cost.list = as.list(10^(-5:-2))){
  l.errors.TRAIN.C.i.Rgtsvm <- lapply(seq_along(pred.err.LIN.SVM.list), function(i){
    pred.err.LIN.SVM.list[[i]]$err.pred.test.K.k
  })
  # Print pairs
  Cost.Err.mtx <- cbind(cost.list, l.errors.TRAIN.C.i.Rgtsvm)
  print(Cost.Err.mtx)
  # Select
  id.min <- which.min(l.errors.TRAIN.C.i.Rgtsvm)
  cost.selected <- cost.list[[which.min(l.errors.TRAIN.C.i.Rgtsvm)]]
  
  # Results list
  res.l <- list(Mtx.Cost.Err.pairs = Cost.Err.mtx,
                ID.which.min = id.min,
                C.min.select = cost.selected
  )
  return(res.l)
}