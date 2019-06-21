#--------------------------------------------------------------------
# calibrate RF (vRF & tRF) scores with Platt scaling - Log. Reg. (LR)
#
#
# Mate Maros
# maros@uni-heidelberg.de
#
# 2019-04-25 UTC 
#--------------------------------------------------------------------


# Define utility/subfunctions for `calibrate_RF_LR()` function -----------------------------------------------------------------------------------------------------------

# Sub-Function-1 for training the Platt-calibration model on the _calibration set_ (=inner.i.j$test)
subfunc_Platt_train_calibration  <- function(y.i.j.innertest, scores.i.j.innertest, diagn.class){
  y.calib.true.diagn01 <- ifelse(y.i.j.innertest == diagn.class, 1, 0) # 1-vs-all
  y.calib.pred.diagn <- scores.i.j.innertest[ , colnames(scores.i.j.innertest) == diagn.class] # class2show
  calib.df <- data.frame(cbind(y.calib.true.diagn01, y.calib.pred.diagn)) # # slow step DF are rewritten in the memory; list are not <-  AdvancedR H.Wickham.
  colnames(calib.df) <- c("y", "x")
  calib.model.Platt.diagn.i <- glm(y ~ x, calib.df, family=binomial)
  return(calib.model.Platt.diagn.i)
}  

# Sub-Function-2 for scoring the test set (outer) using Platt-calibrated classifier from inner-calibration set (sub-function) 
# Inherits calibrated.model.fit as argument from sub-function-1 above
# outer$test; fold (outer) $train & $test
subfunc_Platt_fit_testset <- function(y.i.0.outertest, scores.i.0.outertest, calib.model.Platt.diagn.i, diagn.class){  
  y.test.true.diagn01 <- ifelse(y.i.0.outertest == diagn.class, 1, 0)  # 1-vs-all                        
  y.test.pred.diagn <- scores.i.0.outertest[ , colnames(scores.i.0.outertest) == diagn.class] # CAVE: here scores <= OUTER FOLD$TEST!
  test.df <- data.frame(y.test.pred.diagn) # slow step DFs are rewritten in the memory; 
  colnames(test.df) <- c("x")
  probs.platt.diagn.i <- predict(calib.model.Platt.diagn.i, newdata = test.df, type = "response")
  return(probs.platt.diagn.i)
}  

# HINT/NOTE: there are faster implementations with lapply() or mapply();
# however the speed-up is not relevant. Plus the concept of training calibration model + fitting outer test set is more apparent with for function.


# Define `calibrate_LR()` function --------------------------------------------------------------------------------------------------------------------------------

calibrate_LR <- function(out.path = "vRF-calibrated-LR/", 
                            out.fname = "probsCVfold",
                            nfolds.. = NULL,
                            y.. = NULL,
                            load.path.w.name = "./vRF/CVfold.", # default output of the `run_nestedcv_vRF()` function
                            which.optimized.metric.or.algorithm = c("brier", "miscerr", "mlogl", "vanilla", "svm", "xgboost"),
                            save.metric.name.into.output.file = T,
                            verbose.messages = F){
  
  # 1. Checks
  # Check whether nfolds.. is available
  if(is.null(nfolds..) & exists("nfolds")){nfolds.. <- get("nfolds", envir = .GlobalEnv)} else {
    stop("Please provide a fold structure for nested resampling. For instance, load the nfolds.RData")}
  # Check whether y true outcome is available
  if(is.null(y..) & exists("y")){y.. <- get("y", envir = .GlobalEnv)} else {
    stop("Please provide the true class labels vector (y) for further analyses. For instance, load the y.RData or the MNPbetas10Kvar.RData")}
  
  
  # HINT: by default the `run_nestedcv_tunedRF()` function outputs "tRF/CVfold.K.k.RData" files, each of which contains the following objects:
  # scores.pred.rf.tuned.brier, scores.pred.rf.tuned.miscerr, scores.pred.rf.tuned.mlogl, rfcv.tuned, fold.
  
  # 2. Load .RData output files of the tRF classifier
  for(i in 1:length(nfolds..)){  
    # Create loading environment for safe loading the RData files of outerfolds
    env2load.outer <- environment()
    
    # Load the test set from the "i-th" outerfold (1.0, ..., 5.0; Figure 1. red boxes)
    if(verbose.messages) message("\nLoad the test set from the ", i, ".0 outer fold @ ", Sys.time())
    load(paste0(load.path.w.name, i, ".", 0, ".RData"), envir = env2load.outer)
    if(which.optimized.metric.or.algorithm == "brier"){scores <- get("scores.pred.rf.tuned.brier", envir = env2load.outer)}     
    if(which.optimized.metric.or.algorithm == "miscerr"){scores <- get("scores.pred.rf.tuned.miscerr", envir = env2load.outer)} 
    if(which.optimized.metric.or.algorithm == "mlogl"){scores <- get("scores.pred.rf.tuned.mlogl", envir = env2load.outer)} 
    if(which.optimized.metric.or.algorithm == "vanilla"){scores <- get("scores", envir = env2load.outer)}
    if(which.optimized.metric.or.algorithm == "svm"){scores <- get("scores.pred.svm", envir = env2load.outer)}
    if(which.optimized.metric.or.algorithm == "xgboost"){scores <- get("scores.pred.xgboost", envir = env2load.outer)}
    
    # Re-assign to new variable
    scores.i.0.outertest <- scores
    idx.i.0.outertest <- env2load.outer$fold$test 
    y.i.0.outertest <- y..[idx.i.0.outertest]
    
    # Create empty lists for raw scores of nested inner folds
    scoresl <- list() 
    idxl <- list()
    for(j in 1:length(nfolds..)){
      # Create loading environment for safe loading the RData files of nested inner folds
      env2load.inner <- environment()
      
      # Load all test/calibration sets from the nested subfolds & combine them together for training the calibration model (Figure 1. red rectangles S1.1 - S1.5)
      if(verbose.messages) message("Load & combine all test/calibration sets from the nested subfolds ", i, ".", j)
      load(paste0(load.path.w.name, i, ".", j, ".RData"))   
      if(which.optimized.metric.or.algorithm == "brier"){scores <- get("scores.pred.rf.tuned.brier", envir = env2load.inner)}     
      if(which.optimized.metric.or.algorithm == "miscerr"){scores <- get("scores.pred.rf.tuned.miscerr", envir = env2load.inner)} 
      if(which.optimized.metric.or.algorithm == "mlogl"){scores <- get("scores.pred.rf.tuned.mlogl", envir = env2load.inner)}
      if(which.optimized.metric.or.algorithm == "vanilla"){scores <- get("scores", envir = env2load.inner)}
      if(which.optimized.metric.or.algorithm == "svm"){scores <- get("scores.pred.svm", envir = env2load.inner)}
      if(which.optimized.metric.or.algorithm == "xgboost"){scores <- get("scores.pred.xgboost", envir = env2load.inner)}
      scoresl[[j]] <- scores
      idxl[[j]] <- env2load.inner$fold$test 
    }
    # Collapse lists in to matrix objects
    scores.i.j.innertest.all <- do.call(rbind, scoresl)
    idx.i.j.innertest.all <- unlist(idxl)   
    y.i.j.innertest.all <- y..[idx.i.j.innertest.all]
    
    message("\nTraining calibration model using Platt scaling by LR @ ", Sys.time())
    # Platt scaling with LR 
    probs.platt.diagn.l <- list() 
    for(c in seq_along(colnames(scores.i.j.innertest.all))){  
      # 3. Train calibration model: a glm() model for each diagnosis (levels(y)) in a `1-vs-all` manner # => NOTE: => warnings() # complete or quasi-complete separation
      diagn <- colnames(scores.i.j.innertest.all)[c] 
      # Training the Platt-calibration model on the sum of all (inner fold) calibration/test sets
      # High-frequency (annoying) "pop-up" messages 
      if(verbose.messages) message("Training Platt-scaling on the sum of innerfold test/calibration sets ", i, ".", 1, " - ", i, ".", length(nfolds..), " @ ", Sys.time())  
      platt.calfit <- subfunc_Platt_train_calibration(y.i.j.innertest = y.i.j.innertest.all, scores.i.j.innertest = scores.i.j.innertest.all, diagn.class = diagn) # for each "c" diagnosis 
      
      # 4. Predict the outerfold test set using the trained calibration model (Figure 1. green arrow "predict") to generate calibrated probabilities (Figure 1, yellow box P1.0)
      # High-frequency (annoying) "pop-up" messages 
      if(verbose.messages) message("Fitting Platt-scaled LR model on the raw scores of outer fold test set ", i, ".0", " @ ", Sys.time())  
      probs.platt.diagn.l[[c]] <- subfunc_Platt_fit_testset(y.i.0.outertest = y.i.0.outertest, scores.i.0.outertest = scores.i.0.outertest, 
                                                            calib.model.Platt.diagn.i = platt.calfit, 
                                                            diagn.class = diagn) 
    }  
    probs <- do.call(cbind, probs.platt.diagn.l)
    # Rename columns
    colnames(probs) <- levels(y.i.0.outertest)
    
    # 5. Calculate misclassification errors (ME)
    # NOTE: the predicted class label is the one with the highest score | probability (no further thresholding is applied)
    errs <- sum(colnames(scores.i.0.outertest)[apply(scores.i.0.outertest, 1, which.max)] != y..[nfolds..[[i]][[1]][[1]]$test])/length(nfolds..[[i]][[1]][[1]]$test)
    errp <- sum(colnames(probs)[apply(probs, 1, which.max)] != y..[nfolds..[[i]][[1]][[1]]$test])/length(nfolds..[[i]][[1]][[1]]$test)
    # Print ME
    message("\nMisclassification error of tRF (", which.optimized.metric.or.algorithm, ") raw scores on the ", i, ".0 fold: ", errs)
    message("Misclassification error of Platt-LR-calibrated RF (", which.optimized.metric.or.algorithm, ") probabilities on the ", i, ".0 fold: ", errp)
    
    # 6. Save raw classifier scores and calibrated probabilities into a separate folder and RData file
    folder.path <- file.path(getwd(), out.path)
    dir.create(folder.path, recursive = T, showWarnings = T) # warning if the folder exists
    # RData file
    scores <- scores.i.0.outertest # rewrites `scores` variable for saving to achieve a simpler naming convention
    if(save.metric.name.into.output.file) {
      save(probs, scores, errs, errp, file = file.path(folder.path, paste(out.fname, which.optimized.metric.or.algorithm, "LR", i, 0, "RData", sep = ".")))
    } else {
      save(probs, scores, errs, errp, file = file.path(folder.path, paste(out.fname, "LR", i, 0, "RData", sep = ".")))
    }
  } # for() outer loop i
}


# Function call  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # Run time @ single core (rMBP 15" i9) < 30s
# Sys.time() # 16:36:48 CEST"
# calibrate_LR(out.path = "vRF-calibrated-LR/", 
#                 out.fname = "probsCVfold",
#                 nfolds.. = NULL,
#                 y.. = NULL,
#                 load.path.w.name = "./vRF/CVfold.", 
#                 which.optimized.metric.or.algorithm = "vanilla",
#                 verbose.messages = F)
# Sys.time() # 16:37:15 CEST

# warnings()
# 1: glm.fit: algorithm did not converge
# 2: glm.fit: fitted probabilities numerically 0 or 1 occurred


# vRF - ntree= 500, p = 200, mtry = sqrt(10000) = 100 ;
# Misclassification error of tRF (vanilla) raw scores on the 1.0 fold: 0.0536013400335008
# Misclassification error of Platt-LR-calibrated RF (vanilla) probabilities on the 1.0 fold: 0.0536013400335008
# 
# Misclassification error of tRF (vanilla) raw scores on the 2.0 fold: 0.0551724137931034
# Misclassification error of Platt-LR-calibrated RF (vanilla) probabilities on the 2.0 fold: 0.0586206896551724
# 
# Misclassification error of tRF (vanilla) raw scores on the 3.0 fold: 0.0407079646017699
# Misclassification error of Platt-LR-calibrated RF (vanilla) probabilities on the 3.0 fold: 0.0407079646017699
# 
# Misclassification error of tRF (vanilla) raw scores on the 4.0 fold: 0.0553505535055351
# Misclassification error of Platt-LR-calibrated RF (vanilla) probabilities on the 4.0 fold: 0.0664206642066421
# 
# Misclassification error of tRF (vanilla) raw scores on the 5.0 fold: 0.0328820116054159
# Misclassification error of Platt-LR-calibrated RF (vanilla) probabilities on the 5.0 fold: 0.0406189555125725
# There were 50 or more warnings (use warnings() to see the first 50)
