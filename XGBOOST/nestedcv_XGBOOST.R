#--------------------------------------------------------------------
# ml4calibrated450k - XGBOOST - - Run the nested CV scheme 
#
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-28
#--------------------------------------------------------------------


### Fit & tuned XGBOOST-ed trees in the integrated nested CV scheme

run_nestedcv_XGBOOST <- function(y.. = NULL, 
                                 betas.. = NULL, 
                                 path.betas.var.filtered = "/home/rstudio/data/betas.train.test.10k.filtered/",
                                 fname.betas.p.varfilt = "betas",
                                 subset.CpGs.1k = T, 
                                 n.cv.folds = 5, 
                                 nfolds.. = NULL,   
                                 K.start = 1, k.start = 0,
                                 K.stop = NULL, k.stop = NULL, 
                                 n.CV. = 3, n.rep. = 1, # caret nested tuning
                                 max_depth = 6, 
                                 eta = c(0.1, 0.3), 
                                 gamma = c(0, 0.01),
                                 colsample_bytree = c(0.01, 0.02, 0.05, 0.2), 
                                 subsample = 1, 
                                 min.chwght = 1,
                                 nrounds = 100,       # use default to limit computational burden 
                                 early_stopping_rounds = 50,
                                 n.mc.cores = 72L,   # AWS EC2 c5n.18xlarge v72CPU setting
                                 seed = 1234, 
                                 out.path = "XGBOOST", 
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
        fold <- nfolds..[[K]][[2]][[k]]  
      } else{                                                                          
        message("\n \nCalculating outer fold ", K,".0  ... ",Sys.time()) # Outer CV loops 1.0-5.0 (Fig. 1.)
        fold <- nfolds..[[K]][[1]][[1]]   
      }
      
      # 1. Load `betas.. 
      # Default is betas.. = NULL => Loads data from path
      if(is.null(betas..)) {
        # Load pre-filtered normalized but non-batchadjusted betas for fold K.k
        message("Step 1. Loading pre-filtered normalized but non-batchadjusted betas for (sub)fold ", K, ".", k)
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
        # Note that betas.train and betas.test columns/CpGs are ordered in deacreasing = T => 
        # simply subset [ , 1:1000] => 1k most variable
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
      
      # 2. Data preprocessing for xgboost
      message("\nStep 2. Data preprocessing: reformatting [betas] as xgb.DMatrix & [y] outcome as numeric-1 ... ", Sys.time())
      y.num <- as.numeric(y..) # xgboost needs numeric outcome labels 
      # Train
      y.xgb.train <- y.num[fold$train]-1  # 0-90 # numeric starting from 0 #xgboost's pre-req.
      train.K.k.mtx <- betas.train # dense mtx
      xgb.train.K.k.dgC <- as(betas.train, "CsparseMatrix")  # sparse mtx (supposed to be faster)
      xgb.train.K.k.l <- list(data = xgb.train.K.k.dgC, label = y.xgb.train)
      dtrain <- xgb.DMatrix(data = xgb.train.K.k.l$data, label = xgb.train.K.k.l$label) # xgboost own data structure -> supposed to be the fastest (?!)
      # Test
      y.xgb.test <- y.num[fold$test]-1
      test.K.k.mtx <- betas.test
      xgb.test.K.k.dgC <- as(betas.test, "CsparseMatrix")
      xgb.test.K.k.l <- list(data = xgb.test.K.k.dgC, label = y.xgb.test)
      dtest <- xgb.DMatrix(data = xgb.test.K.k.l$data, label = xgb.test.K.k.l$label)
      # Watchlist 
      watchlist <- list(train = dtrain, test = dtest) # to enable watchlist functionality of xgb.train
      
      message("\nStep 3. Start xgboost-CARET tuning on training set ... ", Sys.time())
      xgb.train.fit.caret <- trainXGBOOST_caret_tuner(y = y..[fold$train],  # factor - caret does not need numeric conversion #y.xgb.train, # numeric 
                                                      train.K.k.mtx = train.K.k.mtx, 
                                                      K. = K, 
                                                      k. = k, 
                                                      dtrain. = dtrain, 
                                                      watchlist. = watchlist,
                                                      n.CV = n.CV., n.rep = n.rep.,  # 3; 1
                                                      seed. = seed, 
                                                      allow.parallel = T,
                                                      mc.cores = n.mc.cores, 
                                                      max_depth. = max_depth, #6, 
                                                      eta. = eta,             #0.1, 0.3 
                                                      gamma. = gamma,         #0, 0.01,
                                                      colsample_bytree. = colsample_bytree, #c(0.01, 0.02, 0.05, 0.2), 
                                                      subsamp = subsample,    #1, 
                                                      minchwght = min.chwght, #1,
                                                      early_stopping_rounds. = early_stopping_rounds, #50,
                                                      nrounds. = nrounds,     #100,  # best prototype models niter=62; d6e0.1g0cs0.01  
                                                      objective. = "multi:softprob", 
                                                      eval_metric. = "merror",    
                                                      save.xgb.model = T)
      
      message("\nStep 4. Predict on test set using best xgb.train boosted tree model ... ", Sys.time())
      # Model PROBABILITY OUTPUTS on test set
      scores.pred.xgboost.vec.test <- predict(object = xgb.train.fit.caret[[1]], 
                                              newdata = test.K.k.mtx,  # dtest (xgb.DMatrix) would be also feasible
                                              ntreelimit = xgb.train.fit.caret[[1]]$best_iteration, 
                                              outputmargin = F)
      
      # Generate a matrix from the vector of probabilities
      message("\n Re-format vector output into a matrix object ... ", Sys.time())
      scores.pred.xgboost <- matrix(scores.pred.xgboost.vec.test, nrow = nrow(test.K.k.mtx), 
                                    ncol = length(levels(y..)), 
                                    byrow = T)
      # Assign row & colnames
      rownames(scores.pred.xgboost) <- rownames(betas.test)
      colnames(scores.pred.xgboost) <- levels(y..)
      
      # Error rate 
      err.xgb.K.k <- sum(colnames(scores.pred.xgboost)[apply(scores.pred.xgboost, 1, which.max)] != y..[fold$test]) / length(fold$test)
      message("\n Misclassification Error on test set ", K, ".", k, " = ", err.xgb.K.k)
      
      message("\nStep 5. Saving output objects & creating output folder (if necessary) @ ", Sys.time())
      # Create output directory  
      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, recursive = T, showWarnings = F)
      #RData.path <- file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      
      # Save scores/probs, glmnet-modell, fold
      save(scores.pred.xgboost, 
           xgb.train.fit.caret,         # list of 4: 1. best model, fitted object
           fold,
           file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      )
    }
  }
  message("Finished @ ", Sys.time())
}