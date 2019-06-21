#--------------------------------------------------------------------
# ml4calibrated450k - tuned RF (tRF) - Run the nested CV scheme 
#
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-24 
#--------------------------------------------------------------------


### Fit tuned random forests (tRF) in the integrated nested CV scheme

run_nestedcv_tunedRF <- function(y.. = NULL, 
                                 betas.. = NULL, 
                                 path.betas.var.filtered = "/home/rstudio/data/betas.train.test.10k.filtered/",
                                 fname.betas.p.varfilt = "betas",
                                 subset.CpGs.1k = T, 
                                 n.cv.folds = 5, 
                                 nfolds.. = NULL, # load("nfolds.RData")  
                                 K.start = 1, k.start = 0,
                                 K.stop = NULL, k.stop = NULL,
                                 n.cv = 5, n.rep = 1, # caret # extra nested tuning
                                 mtry.min = 80, mtry.max = 110, length.mtry = 4, 
                                 # If mtry.min & mtry.max are left at NULL # floor(sqrt(ncol(betas)))*0.5) and floor(sqrt(ncol(betas))) is equally divided to length.mtry parts
                                 ntrees.min = 500, ntrees.max = 2000, ntree.by = 500,
                                 use.default.nodesize.1.only = T, 
                                 nodesize.proc = c(0.01, 0.05, 0.1), 
                                 # CRITICAL/Troubleshooting: it adds the default values nodesize = 1 for classification and = 5 for regression in the subfunc_rf_caret_tuner_customRF()
                                 p.n.pred.var = c(100, 200, 500, 1000, 2000, 5000, 7500, 10000),
                                 cores = 10, 
                                 seed = 1234,
                                 out.path = "tRF", 
                                 out.fname = "CVfold"){
  
  # Check:
  # Check whether y.. is provided
  if(is.null(y..)){
    if(exists("y")){
      y.. <- get("y", envir = .GlobalEnv)
      message(" `y` outcome label was fetched from .GlobalEnv")
    } else {
      stop("Please provide `y..` outcome labels corresponding to the reference cohort 2801 cases",
           " in Capper et al. 2018 (Nature). For instance, load the `y.RData` file.")
    }
  }
  
  # Check whether nfolds.. is provided
  if(is.null(nfolds..) && exists("nfolds")){
    nfolds.. <- get("nfolds", envir = .GlobalEnv)
    message(" `nfolds` nested CV scheme assignment was fetched from .GlobalEnv")
  } else {
    stop("Please provide a fold structure for nested resampling. For instance, load the nfolds.RData")
  }
  
  # Feedback messages
  # Check if K.stop & k.stop was provided
  if(is.null(K.stop) && is.null(k.stop)) {
    message("\nK.stop & k.stop are at default NULL => the full `n.cv.folds` = ", n.cv.folds, 
            " nested CV will be performed.")
    K.stop <- n.cv.outer.folds <- n.cv.folds
    k.stop <- n.cv.inner.folds <- n.cv.folds
  } else { # !is.null() # NOT NULL
    message("K.stop & k.stop are provided & nested CV is limited accordingly.")
    n.cv.outer.folds <-  K.stop
    n.cv.inner.folds <- k.stop
  }
  
  # Start nested CV scheme:
  # Outer loop
  for(K in K.start:n.cv.outer.folds){
    
    # Schedule/Stop nested CV
    ncv.scheduler  <- subfunc_nestedcv_scheduler(K = K, 
                                                 K.start = K.start, K.stop = K.stop, 
                                                 k.start = k.start, k.stop = k.stop, 
                                                 n.cv.folds = n.cv.folds, 
                                                 n.cv.inner.folds = n.cv.inner.folds)
    k.start <- ncv.scheduler$k.start
    n.cv.inner.folds <- ncv.scheduler$n.cv.inner.folds
    
    # Inner/Nested loop
    for(k in k.start:n.cv.inner.folds){ 
      
      if(k > 0){ message("\n \nCalculating inner/nested fold ", K,".", k,"  ... ",Sys.time())  # Inner CV loops 1.1-1.5 (Fig. 1.)
        fold <- nfolds..[[K]][[2]][[k]]  ### [[]][[2]][[]] means inner loop
      } else{                                                                          
        message("\n \nCalculating outer fold ", K,".0  ... ",Sys.time()) # Outer CV loops 1.0-5.0 (Fig. 1.)
        fold <- nfolds..[[K]][[1]][[1]]   ### [[]][[1]][[]] means outer loop 
      }
      
      # Default is betas.. = NULL => Loads data from path
      if(is.null(betas..)) {
        # Load pre-filtered normalized but non-batchadjusted betas for fold K.k
        message("Loading pre-filtered normalized but non-batchadjusted betas for (sub)fold ", K, ".", k)
        # Safe loading into separate env
        env2load <- environment()
        # Define path (use defaults)
        path2load <- file.path(path.betas.var.filtered) # file.path("./data/betas.var.filtered/") # default 
        fname2load <- file.path(path2load, paste(fname.betas.p.varfilt, K, k, "RData", sep = "."))  
        # Load into env
        load(file = fname2load, envir = env2load)
        # Get 
        betas.train <- get(x = "betas.train", envir = env2load)
        betas.test <- get(x = "betas.test", envir = env2load)
        # Note that betas.train and betas.test columns/CpGs are ordered in deacreasing = T => simply subset [ , 1:1000] => 1k most variable
        if(subset.CpGs.1k) {
          betas.train <- betas.train[ , 1:1000]
          betas.test <- betas.test[ , 1:1000]
        }
      } else { # User provided `betas..` (e.g. `betas2801.1.0.RData`) including (`betas2801`) both $train & $test 
        # (only for a given fold => set K.start, k.start accordingly)
        message("\n<NOTE>: This is a legacy option. The `betas.. object should contain variance filtered cases of 2801", 
                " cases according to the respective training set of (sub)fold ", K, ".", k, ".", 
                "\nThis option should be used only for a single fold corresponding to ", K.start, ".", k.start)
        betas.train <- betas..[fold$train, ] 
        betas.test <- betas..[fold$test, ]
        # If subset to 1k TRUE 
        if(subset.CpGs.1k) {
          betas.train <- betas.train[ , 1:1000]
          betas.test <- betas.test[ , 1:1000]
        }
      }
      
      # Check 
      if(max(p.n.pred.var) > ncol(betas.train)) { stop("< Error >: maximum value of `p.n.pred.var` [", 
                                                       max(p.n.pred.var), "] is larger than available in `betas.train` :  [",
                                                       ncol(betas.train), "]. Please adjust the function call.")}
      
      message("\nStart tuning on training set using customRF function within CARET ... ", Sys.time())
      # trainRF
      rfcv.tuned <- trainRF_caret_custom_tuner(y. = y..[fold$train], 
                                               betas. = betas.train, # betas..[fold$train, ], 
                                               cores = cores, 
                                               mtry.min = mtry.min, mtry.max = mtry.max, length.mtry = length.mtry,
                                               ntrees.min = ntrees.min, ntrees.max = ntrees.max, ntree.by = ntree.by,
                                               use.default.classif.nodesize.1 = use.default.nodesize.1.only,
                                               nodesize.proc = nodesize.proc, 
                                               n.cv = n.cv, n.rep =  n.rep,
                                               p = p.n.pred.var, 
                                               p.tuning.brier = T, p.tuning.miscl.err = T, p.tuning.mlogloss = T, 
                                               seed = seed+1, 
                                               allowParallel = T)  
      
      # NOTE: rfcv.tuned variable contains 
      # res <- list(rf.pred.best.brier, rf.pred.best.err, rf.pred.best.mlogl, imp.perm, 
      #             rf.pred.l, score.pred.l, brier.p.l, err.p.l, mlogloss.p.l, t0, t1, t2)
      
      # Use tuned random forest modell fit rfcv.tuned[[1]] = rf.pred.best => 
      # to predict the corresponding CALIBRATION or TEST SETS (if innerfold then calibration set ; if outer fold then test set)
      message("\nFit tuned random forests (tRF) on  test set ", K, ".", k, " in parallel (3 threads) ... ", Sys.time())
      iterator <- 1:3 
      scores.pred.rf.tuned.l <- mclapply(seq_along(iterator), function(i){
        predict(object = rfcv.tuned[[i]], 
                newdata = betas.test[ , match(rownames(rfcv.tuned[[i]]$importance), colnames(betas.test))],
                type = "prob")
      }, mc.preschedule = T, mc.set.seed = T, mc.cores = length(iterator))
      
      # Rewrite into new variable for saving (not very drive/memory efficient but needed for downstream compatibility)
      scores.pred.rf.tuned.brier <- scores.pred.rf.tuned.l[[1]]
      scores.pred.rf.tuned.miscerr <- scores.pred.rf.tuned.l[[2]]
      scores.pred.rf.tuned.mlogl <- scores.pred.rf.tuned.l[[3]]
      
      # Calculate Misclassification Errors (ME)
      # Calculates ME for BS tunedRF
      err.scores.rf.tuned.brier <- sum(colnames(scores.pred.rf.tuned.brier)[apply(scores.pred.rf.tuned.brier, 1, which.max)] != y..[fold$test]) / length(fold$test) 
      # Calculates ME for ME tunedRF
      err.scores.rf.tuned.miscerr <- sum(colnames(scores.pred.rf.tuned.miscerr)[apply(scores.pred.rf.tuned.miscerr, 1, which.max)] != y..[fold$test]) / length(fold$test) 
      # Calculates ME for mLL tunedRF
      err.scores.rf.tuned.mlogl <- sum(colnames(scores.pred.rf.tuned.mlogl)[apply(scores.pred.rf.tuned.mlogl, 1, which.max)] != y..[fold$test]) / length(fold$test) 
      
      # Print MEs
      message("\nMisclassification error on [Test Set] CVfold.", K, ".", k, 
              "\n 1. Brier optimized model: ", err.scores.rf.tuned.brier,                   # scores.pred.rf.tuned.l[[1]]
              "; \n 2. Misclassif. error optimized model: ",  err.scores.rf.tuned.miscerr,  # scores.pred.rf.tuned.l[[2]]
              "; \n 3. Mlogloss optimized model: ", err.scores.rf.tuned.mlogl,              # scores.pred.rf.tuned.l[[3]]
              "; @ ", Sys.time())      
      
      # Create output directory  
      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, recursive = T, showWarnings = F)
      #RData.path <- file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      
      # Save scores, RF-Modell, fold
      save(scores.pred.rf.tuned.brier, scores.pred.rf.tuned.miscerr, scores.pred.rf.tuned.mlogl, 
           rfcv.tuned, fold,
           file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      )
      # CRITICAL/Troubleshooting: the output .RData file can be large, as it contains multiple copies of large matrices (2,801 x 10,000 approx. 215 MB each) 
      # adding up to 1 - 1.5 Gb. Hence, the complete nested CV scheme might require 40-50Gb free space on the respective drive.
    }
  }
  message("Finished ... ", Sys.time())
}  


