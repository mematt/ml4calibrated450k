#------------------------------------------------------------------------------
# ml4calibrated450k - Support Vector Machines (SVM) - Run the nested CV scheme
# 
#                   - Linear Kernel SVM (LiblineaR)
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#-----------------------------------------------------------------------------


### Fit linear kernel SVM (SVM-LK) of the `LiblineaR`` package in the integrated nested CV scheme 

## 3. Fit tuned `SVM LiblineaR` models in the integrated nested CV scheme

# 1. type 0: L2-regularized logistic regression (L2LR) and 
# 2. type 4: Crammer & Singer model 

### `run_nestedcv_SVM_LiblineaR` - hyperparameter (C, cost) tuning using 5-fold extra nested CV within the training loop 

run_nestedcv_SVM_LiblineaR <- function(y.. = NULL, 
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
                                       mod.type = 0, # L2-LR with probability output
                                       type4.CramSing = T, # only class estimates no probability estimates
                                       verbose = T,
                                       parallel = T,
                                       n.mc.cores = 8L,   # standard 4 cores/8 threads machines
                                       seed = 1234, 
                                       out.path = "SVM-LiblineaR", 
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
      
      if(k > 0){ message("\n \nCalculating inner/nested fold ", K,".", k,"  ... ",Sys.time())  
        fold <- nfolds..[[K]][[2]][[k]]  ### Inner CV loops 1.1-1.5 (Fig. 1.)
      } else{                                                                          
        message("\n \nCalculating outer fold ", K,".0  ... ",Sys.time()) 
        fold <- nfolds..[[K]][[1]][[1]]   ### Outer CV loops 1.0-5.0 (Fig. 1.)
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
        # Note that betas.train and betas.test columns/CpGs are ordered in deacreasing = T 
        # "Fast track" => simply subset [ , 1:1000] => 1k most variable 
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
      
      message("\nStep 3. Start tuning on training set ... ", Sys.time())
      # Run train- on scaled s.betas.TRAIN sets both Outer and Inner folds => "Train the model on train set (either nested or outer)"
      liblinearcv <- train_SVM_LiblineaR(y = y..[fold$train], 
                                         s.betas.Train = s.betas.Train, 
                                         n.CV = n.CV., 
                                         seed = seed+1, 
                                         multicore = parallel,  # if parallel=F (sequential) mc.cores is ignored
                                         mc.cores = n.mc.cores, 
                                         C.base = C.base, C.min = C.min, C.max = C.max, 
                                         mod.type = mod.type, 
                                         type4.CramSing = type4.CramSing, verbose = verbose)
      
      # Scale test set according to ranges of training set above
      message("\nStep 4. Scaling and centering test set using attributes of the trainin set ... ", Sys.time())
      s.betas.Test <- scale(betas.test, attr(s.betas.Train, "scaled:center"), attr(s.betas.Train, "scaled:scale"))  
      
      #message("\nCreating output folder (if necessary) @ ", Sys.time())
      # Create output directory  
      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, recursive = T, showWarnings = F)
      
      # Use fitted model to predict the corresponding test set 
      message("\nStep 5. Predict on scaled `test set` using tuned model object ... ", Sys.time())
      # PROBABILITY OUTPUTS
      # only possible if the model was fitted with type=0, type=6 or type=7, i.e. a Logistic Regression. 
      # Default is FALSE
      if(mod.type == 0 || mod.type == 6 || mod.type == 7 && type4.CramSing == T){
        message("Probability output is only possible for model type = ", paste(c(0, 6, 7), sep = " ; "))
        scores.pred.svm.liblinear.mod.type <- predict(liblinearcv[[1]], s.betas.Test, proba = T, decisionValues=TRUE) 
        # [[1]] => modfit.liblinear.strain$m.fit.mod.type
        
        # Type 4 - CS model object 
        class.pred.svm.liblinear.ty4.CraSi <- predict(liblinearcv[[2]], s.betas.Test, proba = F, decisionValues=TRUE) 
        # [[2]] => modfit.liblinear.strain$m.fit.ty4.cs # if CS=T then exists otherwise NULL
        
        # Calculate Error                                                                                 
        # Specified model type: > default L2R-LogReg (type0)
        err.svm.mod.type <- sum(colnames(scores.pred.svm.liblinear.mod.type$probabilities)[apply(scores.pred.svm.liblinear.mod.type$probabilities,
                                                                                                 1, which.max)] != y..[fold$test])/length(fold$test) 
        message("\nMisclassification error of type (", mod.type, ") model on fold ", K, ".", k, " : ",
                err.svm.mod.type, " ; @ ", Sys.time())
        
        # Save scores, SVM-LIBLINEAR-Modell, fold
        message("\nStep 6. Saving output objects & creating output folder (if necessary) @ ", Sys.time())
        save(scores.pred.svm.liblinear.mod.type, 
             class.pred.svm.liblinear.ty4.CraSi, 
             liblinearcv, 
             fold, 
             file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = ".")))
        
      } else {
        # Model types with only CLASS OUTPUTS                                                                   
        class.pred.svm.liblinear.ty4.CraSi <- predict(liblinearcv[[1]], s.betas.Test, proba = F, decisionValues=TRUE)
        # Calculate Error                                                                                 
        # Type 4 - Crammer & Singer
        err.svm.ty4 <- sum(class.pred.svm.liblinear.ty4.CraSi$predictions != y..[fold$test])/length(fold$test)
        message("\nMisclassification error fold ", K, ".", k, " : ", err.svm.ty4, " ; @ ", Sys.time())
        
        # Save scores, SVM-LIBLINEAR-Modell, fold
        message("\nStep 6. Saving output objects & creating output folder (if necessary) @ ", Sys.time())
        save(class.pred.svm.liblinear.ty4.CraSi, 
             liblinearcv, 
             fold, 
             file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = ".")))
        
      }
    }
  }
  message("Full run finished ...", Sys.time())
}