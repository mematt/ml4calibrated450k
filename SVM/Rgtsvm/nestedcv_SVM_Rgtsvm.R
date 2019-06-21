#--------------------------------------------------------------------------------
# ml4calibrated450k - Support Vector Machines (SVM) -  Run the nested CV scheme  
# 
#                   - GPU-accelerated, Linear Kernel SVM (Rgtsvm)
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-27 
#-------------------------------------------------------------------------------


### Fit tuned linear kernel SVM `Rgtsvm` in the integrated nested CV scheme on NVIDIA GPU using CUDA acceleration

run_nestedcv_SVM_Rgtsvm <- function(y.. = NULL, 
                                    betas.. = NULL, 
                                    path.betas.var.filtered = "/home/rstudio/data/betas.train.test.10k.filtered/",
                                    fname.betas.p.varfilt = "betas",
                                    subset.CpGs.1k = T, 
                                    n.cv.folds = 5, 
                                    nfolds.. = NULL,   
                                    K.start = 1, k.start = 0,
                                    K.stop = NULL, k.stop = NULL, 
                                    n.CV. = 5,  # Rgtsm nested CV tuning of Cost parameter
                                    Cost = c(10^(-5:-2)),
                                    scale.training.n.test.sets = T, 
                                    probability.output = T,
                                    GPU.ID = 0, 
                                    verbose.training = T,
                                    seed = 1234, 
                                    out.path = "SVM-Rgtsvm-10k", 
                                    out.fname = "CVfold"){
  
  message("Start @ ", Sys.time())
  # Check:
  # Check whether y.. is provided
  if(is.null(y..)){
    if(exists("y")){
      y.. <- get("y", envir = .GlobalEnv)
      message("\n `y` outcome label was fetched from .GlobalEnv")
    } else {
      stop("Please provide `y..` outcome labels corresponding to the reference cohort 2801 cases in Capper et al. 2018 (Nature). For instance, load the `y.RData` file.")
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
      
      # 2. Data preprocessing for SVM Rgtsvm (GPU)
      # List of costs to evaluate # Convert betas (df) => to matrix
      Cost.list <- as.list(Cost)
      betas.mtx.train.K.k <- betas.train
      y.train.K.k <- y..[fold$train]
      
      message("\nStart tuning on training set using GPU ... ", Sys.time())
      # Fit Rgtsvm on GPU # Scale = T 
      l.model.obj.LIN.SVM.TRAIN.C.i <- my_func_train.LIN.SVM.GPU_v2(x.mtx2train = betas.mtx.train.K.k, 
                                                                    y.fac.train = y.train.K.k, 
                                                                    cost.list = Cost.list, 
                                                                    scale.x.mtx = scale.training.n.test.sets,  
                                                                    probs = probability.output, n.cross = n.CV.,
                                                                    GPU.id = GPU.ID, 
                                                                    verb = verbose.training)
      
      # Scale test set according to ranges of training set above
      message("\nScaling and centering of test set of betas is automatically incorporated in to predict.Rgtsvm() ... ", Sys.time())
      betas.mtx.test.K.k <- betas.test
      y.test.K.k <- y..[fold$test]
      
      message("\nPredict {Rgtsvm} linear kernel model objects on TEST SET for all cost.list (C) parameters ... ", Sys.time())
      # Apply predictor function
      l.LIN.SVM.pred.test.K.k <- lapply(seq_along(l.model.obj.LIN.SVM.TRAIN.C.i), function(i){
        my_func_predictor_error_nestedCV(object.svm = l.model.obj.LIN.SVM.TRAIN.C.i[[i]], 
                                         data2pred = betas.mtx.test.K.k, 
                                         y.factor.levels = y.test.K.k,
                                         betas.df = betas.test,  
                                         y.K.k = y.test.K.k, 
                                         fold.str = fold)
      })
      # NOTE: 
      # str(l.LIN.SVM.pred.test.K.k) # list of 5 (length Cost.list) # 
      # l.LIN.SVM.pred.test.K.k[[1]]$             # Cost.list[[1]]  # 10^-5, [[2]] 10^-4
      # list(pred.svm.fit.obj = pred.fit.func,    # predict.Rgtsvm object of test.K.k
      #      mtx.pred.probs = mtx.pred.fit.probs, # mtx. of probabilities # rownames are kept ! # it is the attr(..., "probabilities") of predict() factor output 
      #      y.pred.test.K.k = y.pred.test.K.k,   # factor length of test.K.k with labels 
      #      err.pred.test.K.k = err.pred)        # numeric value misclassif. Error for fold K.k 
      
      # Find lowest error with lowest Cost (C)
      message("\nSelect best Cost value from tuning grid @ ", Sys.time())
      l.C.opt.selected <- my_func_select_cost_value_LIN.SVM(pred.err.LIN.SVM.list = l.LIN.SVM.pred.test.K.k, cost.list = Cost.list)
      # l.C.opt.selected contains list of:
      # Mtx.Cost.Err.pairs = Cost.Err.mtx,
      # ID.which.min = id.min,
      # C.min.select = cost.selected
      
      # Return objects
      pred.svm.fit.obj.K.k <- l.LIN.SVM.pred.test.K.k[[l.C.opt.selected$ID.which.min]]$pred.svm.fit.obj
      scores.pred.svm.Rgtsvm <- l.LIN.SVM.pred.test.K.k[[l.C.opt.selected$ID.which.min]]$mtx.pred.probs
      class.pred.svm.Rgtsvm <- l.LIN.SVM.pred.test.K.k[[l.C.opt.selected$ID.which.min]]$y.pred.test.K.k
      err.pred.test.K.k <- l.LIN.SVM.pred.test.K.k[[l.C.opt.selected$ID.which.min]]$err.pred.test.K.k
      
      # Present Test Error of C                                                                                 
      message("\nOptimal cost (C) value:", l.C.opt.selected$cost.selected, 
              "; Misclassification error (ME): ", err.pred.test.K.k, " @ ", Sys.time())
      
      
      message("\nSaving output objects & creating output folder (if necessary) @ ", Sys.time())
      # Create output directory  
      folder.path <- file.path(getwd(), out.path)
      dir.create(folder.path, recursive = T, showWarnings = F)
      #RData.path <- file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      
      # Save scores, SVM-Rgtsvm object, fold
      save(scores.pred.svm.Rgtsvm, class.pred.svm.Rgtsvm, err.pred.test.K.k,
           pred.svm.fit.obj.K.k,           # test  predict object # scaled according to train attributes
           l.model.obj.LIN.SVM.TRAIN.C.i,  # train list with Cost values # scaled 
           fold, 
           file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))  
           # NOTE: Rgtsm generates/outputs actually calibrated probabilities thus it should be `probs` instead of `scores`.
      )
    } # for k
  } # for K
  message("Finished full run ...",Sys.time())
} # end of function
