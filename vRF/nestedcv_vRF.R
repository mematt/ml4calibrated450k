#--------------------------------------------------------------------
# ml4calibrated450k - vanilla RF (vRF) - Run the nested CV scheme 
#
#
# Matt Maros & Martin Sill
# maros@uni-heidelberg.de & m.sill@dkfz.de
#
# 2019-04-18 
#--------------------------------------------------------------------


### Fit vanilla random forests (vRF) in the integrated nested CV scheme


run_nestedcv_vRF <- function(y.. = NULL,
                             betas.. = NULL,
                             path.betas.var.filtered = "/home/rstudio/data/betas.train.test.10k.filtered/", # needs path to `betas.p.varfilt.K.k.RData`
                             fname.betas.p.varfilt = "betas",
                             subset.CpGs.1k = F,
                             n.cv.folds = 5, 
                             nfolds.. = NULL,
                             K.start = 1, k.start = 0,
                             K.stop = NULL, k.stop = NULL,
                             ntrees = 500, 
                             p = 200,
                             cores = 10,
                             seed = 1234,
                             out.path = "vRF-testrun-1", 
                             out.fname = "CVfold"){
  
  # Check:
  # Check whether y.. is provided
  if(is.null(y..)){
    if(exists("y")){
      y.. <- get("y", envir = .GlobalEnv)
      message(" `y` outcome label was fetched from .GlobalEnv")
    } else {
      stop("Please provide `y..` outcome labels corresponding to the reference cohort 2801 cases in Capper et al. 2018 (Nature).",
           " For instance, load the `y.RData` file.")
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
  for(K in K.start:n.cv.folds){  
    
    # Schedule/Stop nested CV
    ncv.scheduler  <- subfunc_nestedcv_scheduler(K = K, 
                                                 K.start = K.start, K.stop = K.stop, 
                                                 k.start = k.start, k.stop = k.stop, 
                                                 n.cv.folds = n.cv.folds, 
                                                 n.cv.inner.folds = n.cv.inner.folds)
    k.start <- ncv.scheduler$k.start
    n.cv.inner.folds <- ncv.scheduler$n.cv.inner.folds
    
    # Inner/Nested loop
    for(k in k.start:n.cv.folds){ 
      
      if(k > 0){ message("\n \nCalculating inner/nested fold ", K,".", k,"  ... ",Sys.time())  
        fold <- nfolds..[[K]][[2]][[k]]  ### [[]][[2]][[]] means inner loop # Inner CV loops 1.1-1.5 (Fig. 1.)
      } else{                                                                          
        message("\n \nCalculating outer fold ", K,".0  ... ",Sys.time()) 
        fold <- nfolds..[[K]][[1]][[1]]   ### [[]][[1]][[]] means outer loop # Outer CV loops 1.0-5.0 (Fig. 1.)
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
      
      # Check 
      if(max(p) > ncol(betas.train)) { stop("< Error >: maximum value of `p` [", 
                                            max(p), "] is larger than available in `betas.train` :  [",
                                            ncol(betas.train), "]. Please adjust the function call.")}
      
      # train vRF
      message("Start tuning on training set using `trainRF` function ... ", Sys.time())
      rfcv <- trainRF(y = y[fold$train],
                      betas = betas.train, 
                      ntrees = ntrees,
                      p = p,
                      seed = seed,
                      cores = cores) # MM: see trainRF.R script
      
      # Predict > test/Calibration set
      message("\nFit tuned vanilla RF on : test/calibration set ", "( n_cases: ", nrow(betas.test), ") : ", K, ".", k, " ... ", Sys.time())
      scores <- predict(rfcv[[1]], 
                        betas.test[ , match(rownames(rfcv[[1]]$importance), colnames(betas.test))],
                        type = "prob")
      
      # Calculate Misclassification Errors (ME) on test/calibration set
      err <- sum(colnames(scores)[apply(scores, 1, which.max)] != y[fold$test]) / length(fold$test)
      # Print ME
      message("Misclassification error: ", err, " @ ", Sys.time())
      
      # Create output directory  
      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, showWarnings = F)
      #RData.path <- file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      
      # Save scores of vanilla RF 
      save(scores, rfcv, fold,
           file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      )
    }
  }
  message("Finished ... ", Sys.time())
}