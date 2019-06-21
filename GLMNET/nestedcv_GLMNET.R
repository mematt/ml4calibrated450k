#-------------------------------------------------------------------------------------------------------------
# ml4calibrated450k - Elastic Net penalized Multinomial Logistic Regression (ELNET) - Run the nested CV scheme
# 
#                   
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#-------------------------------------------------------------------------------------------------------------

## Fit tuned `GLMNET` models in the integrated nested CV scheme

### `run_nestedcv_glmnet` - concurrent tuning of hyperparameters (alpha, lamda) using 5-fold extra nested CV within the training set of each (sub)fold.

run_nestedcv_GLMNET <- function(y.. = NULL,
                                betas.. = NULL,
                                path.betas.var.filtered = "/home/rstudio/data/betas.train.test.10k.filtered/",  
                                fname.betas.p.varfilt = "betas",  
                                subset.CpGs.1k = F,
                                n.cv.folds = 5, 
                                nfolds.. = NULL,
                                K.start = 1, k.start = 0,
                                K.stop = NULL, k.stop = NULL,                          
                                n.cv.folds.cvglmnet = 5,
                                alpha.min. = 0, alpha.max. = 1, by. = 0.1,  
                                cores = 11, 
                                seed. = 1234, 
                                out.path = "GLMNET",
                                out.fname = "CVfold"){
  # Check:
  # Check whether y.. is provided
  if(is.null(y..) & exists("y")){
    y.. <- get("y", envir = .GlobalEnv)
    message(" `y` outcome label was fetched from .GlobalEnv")
  } else {
    stop("Please provide `y..` outcome labels corresponding to the reference cohort 2801 cases in Capper et al. 2018 (Nature).", 
         " For instance, load the `y.RData` file.")
  }
  
  # Check whether nfolds.. is provided
  if(is.null(nfolds..) & exists("nfolds")){
    nfolds.. <- get("nfolds", envir = .GlobalEnv)
    message(" `nfolds` nested CV scheme assignment was fetched from .GlobalEnv")
  } else {
    stop("Please provide a fold structure for nested resampling. For instance, load the nfolds.RData")
  }
  
  # Check whether betas.. is provided and is a matrix object <CAVE>: glmnet needs matrix not df
  if(!is.null(betas..) || class(betas..) != "matrix"){
    message("betas.. is either NULL or a 'matrix' object: TRUE")
  } else {
    stop("Please provide a matrix object for argument `betas..` ")
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
      
      # trainGLMNET
      message("Start tuning on training set using trainGLMNET function ... ", Sys.time())
      # Run train-myTunedCV.GLMNET on all $TRAIN sets both outer and inner folds => "Train the model on train set"
      glmnetcv.tuned <- trainGLMNET(y = y..[fold$train], 
                                    betas = betas.train,  
                                    seed = seed., 
                                    alpha.min = alpha.min., 
                                    alpha.max = alpha.max., 
                                    by = by.,  
                                    nfolds.cvglmnet = n.cv.folds.cvglmnet,  # 5 computationally more tractable # nfolds for cv.glmnet default = 10
                                    mc.cores = cores)
      # NOTE: glmnetcv.tuned variable contains res <-list(res.cvfit.glmnet.tuned$opt.mod, probs.glmnet.tuned, res.cvfit.glmnet.tuned, t1, t2)  
      
      # Test/Calibration set
      
      # Use tuned glmnet modell fit glmnetcv.tuned[[1]] = res.cvfit.glmnet.tuned$opt.mod => to predict the corresponding CALIBRATION or TEST SETS 
      # (if innerfold then calibration set ; if outer fold then test set)
      message("\nFit tuned glmnet on : test set ", K, ".", k, " ... ", Sys.time())
      probs <- predict(glmnetcv.tuned[[1]], 
                       newx = betas.test,
                       type="response")[,,1] # Note: the output of predict.glmnet() is an array 
      
      # Calculate Misclassification Errors (ME)
      err.probs.glmnet <- sum(colnames(probs)[apply(probs, 1, which.max)] != y..[fold$test]) / length(fold$test)
      # Print MEs
      message("\nMisclassification error on [Test Set] CVfold.", K, ".", k,
              "\n of GLMNET with alpha = ", glmnetcv.tuned[[3]]$opt.alpha, 
              " and lambda = ", glmnetcv.tuned[[3]]$opt.lambda, 
              " setting is: ", err.probs.glmnet, 
              " @ ", Sys.time())
      
      # Create output directory  
      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, showWarnings = F, recursive = T)
      #RData.path <- file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      
      # Save scores, RF-Modell, fold
      save(probs, 
           glmnetcv.tuned, 
           fold,
           file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      )     
    }
  }
  message("Finished @ ",Sys.time())
}