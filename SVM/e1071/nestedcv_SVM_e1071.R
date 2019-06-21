#--------------------------------------------------------------------------------
# ml4calibrated450k - Support Vector Machines (SVM) -  Run the nested CV scheme  
# 
#                   - Linear Kernel SVM (e1071)
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#-------------------------------------------------------------------------------

### Fit linear kernel SVM (SVM-LK) of the e1071 package in the integrated nested CV scheme 

run_nestedcv_SVM_e1071 <- function(y.. = NULL, 
                                   betas.. = NULL, 
                                   path.betas.var.filtered = "/home/rstudio/data/betas.train.test.10k.filtered/",
                                   fname.betas.p.varfilt = "betas",
                                   subset.CpGs.1k = F, 
                                   n.cv.folds = 5, 
                                   nfolds.. = NULL,   
                                   K.start = 1, k.start = 0,
                                   K.stop = NULL, k.stop = NULL, 
                                   n.CV. = 5,  # extra nested tuning in training loop
                                   C.base = 10, C.min = -3, C.max = 3, 
                                   n.mc.cores = 8L,   # standard 4 cores/8 threads machines
                                   seed = 1234, 
                                   out.path = "SVM-e1071-10k", 
                                   out.fname = "CVfold"){
  
  # Check:
  # Check whether y.. is provided
  if(is.null(y..)){
    if(exists("y")){
      y.. <- get("y", envir = .GlobalEnv)
      message("\n `y` outcome label was fetched from .GlobalEnv")
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
  
  # Check if K.stop & k.stop was provided
  # <NOTE> if one of K|k.stop = NULL this gives an error. # Error in 1:n.cv.inner.folds : argument of length 0
  if(is.null(K.stop) && is.null(k.stop)) {
    message("\nK.stop & k.stop are at default NULL => the full `n.cv.folds` = ", n.cv.folds, " nested CV will be performed.")
    n.cv.outer.folds <- n.cv.folds
    n.cv.inner.folds <- n.cv.folds
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
      
      # 1. Load `betas.. 
      # Default is betas.. = NULL => Loads data from path
      if(is.null(betas..)) {
        # Load pre-filtered normalized but non-batchadjusted betas for fold K.k
        message("Setp 1. Loading pre-filtered normalized but non-batchadjusted betas for (sub)fold ", K, ".", k)
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
          betas.train <- betas.train[ , 1:1000] # matrix
          betas.test <- betas.test[ , 1:1000]   # matrix
        }
      } else { 
        # User provided `betas..` (e.g. `betas2801.1.0.RData`) including (`betas2801`) both $train & $test 
        # (only for a given fold => set K.start, k.start accordingly)
        message("\n<NOTE>: This is a legacy option. The `betas.. object should contain variance filtered cases of 2801 cases", 
                " according to the respective training set of (sub)fold ", K, ".", k, ".", 
                "\nThis option should be used only for a single fold corresponding to ", K.start, ".", k.start)
        betas.train <- betas..[fold$train, ] 
        betas.test <- betas..[fold$test, ]
        # If subset to 1k TRUE 
        if(subset.CpGs.1k) {
          betas.train <- betas.train[ , 1:1000]
          betas.test <- betas.test[ , 1:1000]
        }
      }
      
      message("\nStep 2. Pre-processing: scaling and centering training set ... ", Sys.time())
      s.betas.Train <- scale(betas.train, center = TRUE, scale = TRUE)
      
      # # Tune & train on training set # use internal scaling of e1071::svm()
      message("\nStart tuning on training set (scale & center happens internally within e1071::svm() ... ", Sys.time())
      message("\nScaling is performed internally within e1071::svm() before training,", 
              "and also using predict.svm() on the test/validation set according to the attributes of the training data")
      svm.linearcv <- train_SVM_e1071_LK(y = y..[fold$train], 
                                         betas.Train = betas.train, 
                                         seed = seed + 1, 
                                         nfolds = n.CV., 
                                         mc.cores = n.mc.cores,
                                         C.base = C.base, C.min = C.min, C.max = C.max)
      # Predict test set
      message("\nPredict SVM linear kernel {e1071} model with tuned cost (C) parameter ... ", 
              "\n Note predict.svm(): \'If the training set was scaled by svm (done by default),", 
              " the new data is scaled accordingly using scale and center of the training data.\' ...", Sys.time())
      scores.pred.svm.e1071.obj <- predict(object = svm.linearcv[[1]], 
                                           newdata = betas.test, 
                                           probability = T, 
                                           decisionValues = TRUE) 
      # probs.pred.SVM.e1071.obj => is a factor with attributes 
      # Get probabilities
      scores.pred.svm.e1071.mtx <- attr(scores.pred.svm.e1071.obj, "probabilities")
      # !!!CAVE: colnames() order might not be the same as in levels(y) originally!!!
      
      # Calculate Error                                                                                 
      # SVM Linear e1071: 
      err.svm.e1071.probs <- sum(colnames(scores.pred.svm.e1071.mtx)[apply(scores.pred.svm.e1071.mtx, 1, which.max)] != y..[fold$test]) / length(fold$test) 
      message("\nMisclassification error on test set estimated using [probabilities matrix] output: ",
              err.svm.e1071.probs, " ; ", Sys.time())
      
      # Control Steps
      message("\nControl step: whether rownames are identical (betas$K.k.fold$test):", 
              identical(rownames(scores.pred.svm.e1071.mtx), rownames(betas.test))) 
      message("Control step: whether colnames are identical (betas$K.k.fold$test):", 
              identical(colnames(scores.pred.svm.e1071.mtx), levels(y..))) 
      if(identical(colnames(scores.pred.svm.e1071.mtx), levels(y..[fold$test])) == FALSE){
        message("CAVE: Order of levels(y) and colnames(probs.pred.SVM.e1071.mtx)", 
                " => needs matching during performance evaluation!")
      }
      
      message("\nStep 5. Saving output objects & creating output folder (if necessary) @ ", Sys.time())
      # Create output directory  
      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, recursive = T, showWarnings = F)
      #RData.path <- file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      
      # Save scores, SVM-LIBLINEAR-Modell, fold
      save(scores.pred.svm.e1071.mtx, 
           scores.pred.svm.e1071.obj, 
           svm.linearcv, 
           fold, 
           file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      )
    }
  }
  message("Full run finished @ ", Sys.time())
}