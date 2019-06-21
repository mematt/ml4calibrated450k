#----------------------------------------------------------------------
# calibrate RF (vRF & tRF) scores with Platt scaling 
#                                 - Firth penalize Log. Reg. (FLR)
#
#
# Mate Maros
# maros@uni-heidelberg.de
#
# 2019-04-25 UTC 
#--------------------------------------------------------------------


# Define utility/subfunctions for `calibrate_RF_LR()` function -----------------------------------------------------------------------------------------------------------

# PLATT scaling Firth's penalized logistic regressoin (1993) - in case of complete and quasi-complete separation 

# Package - brglm (newer version: brglm2)
# The ... **brglm** R package can only handle generalized linear models with binomial responses. 
# The **brglm2**  R package (a newer version of brglm) # can handle  multinomial responses and it will become a wrapper for brglm # 

if (!requireNamespace("brglm", quietly = FALSE)) { install.packages("brglm")
  } else { library(brglm) }
# Load package
library(brglm) 

# Subfunction-1 for training the Platt-calibration model with BRGLM - Firth Penalized Log-Reg (Kosmidis) on the _calibration set_ ( = inner.i.j$test)
subfunc_Platt_train_calibration_Firth  <- function(y.i.j.innertest, scores.i.j.innertest, diagn.class, brglm.control.max.iteration){
  y.calib.true.diagn01 <- ifelse(y.i.j.innertest == diagn.class, 1, 0)
  y.calib.pred.diagn <- scores.i.j.innertest[ , colnames(scores.i.j.innertest) == diagn.class] # class2show
  calib.df <- data.frame(cbind(y.calib.true.diagn01, y.calib.pred.diagn)) # slow step DF are rewritten in the memory
  colnames(calib.df) <- c("y", "x")
  calib.model.Platt.Firth.diagn.i <- brglm(formula = y ~ x, data = calib.df, family = binomial(logit), 
                                           method = "brglm.fit", 
                                           control.brglm = brglm.control(br.maxit = brglm.control.max.iteration)) # default is br.maxit = 100!  
                                           # further controls left at default: br.epsilon = 1e-08 ;  
  return(calib.model.Platt.Firth.diagn.i)
}  

# Subfunction-2 for scoring the test set (outer) using Platt-calibrated classifier with trained Firth LR model on the sum of inner fold calibration ses (subfunction-1) 
# Inherits calibrated.model.fit as argument from sub-function-1
subfunc_Platt_fit_testset_Firth <- function(y.i.0.outertest, scores.i.0.outertest, calib.model.Platt.Firth.diagn.i, diagn.class){  
  y.test.true.diagn01 <- ifelse(y.i.0.outertest == diagn.class, 1, 0)                         
  y.test.pred.diagn <- scores.i.0.outertest[ , colnames(scores.i.0.outertest) == diagn.class] # CAVE here scores = outer fold $test
  test.df <- data.frame(y.test.pred.diagn) # slow step DF are rewritten in the memory
  colnames(test.df) <- c("x")
  probs.platt.firth.diagn.i <- predict(calib.model.Platt.Firth.diagn.i, newdata = test.df, type = "response")
  return(probs.platt.firth.diagn.i)
}

# HINT/NOTE: there are faster implementations with lapply() or mapply();
# however the speed-up is not relevant. Plus the concept of training calibration model + fitting outer test set is more apparent with for function.


# Define `calibrate_FLR()` function --------------------------------------------------------------------------------------------------------------------------------

calibrate_FLR <- function(out.path = "vRF-calibrated-FLR/", 
                             out.fname = "probsCVfold",
                             nfolds.. = NULL,
                             y.. = NULL,
                             load.path.w.name = "./vRF/CVfold.", # default output of the `run_nestedcv_vRF()` function
                             which.optimized.metric.or.algorithm = c("brier", "miscerr", "mlogl", "vanilla", "svm", "xgboost"),
                             save.metric.name.into.output.file = T, 
                             brglm.ctrl.max.iter = 10000, # default in brglm is br.maxit = 100
                             verbose.messages = T){
  
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
    
    message("\nTraining calibration model using Firth's penalized LR @ ", Sys.time())
    # Platt scaling with Firth's penalized LR (FLR)
    probs.platt.firth.brglm.l <- list() 
    
    for(c in seq_along(colnames(scores.i.j.innertest.all))){
      diagn <- colnames(scores.i.j.innertest.all)[c]
      # 3. Train calibration model: a brglm() model for each diagnosis (levels(y)) in a `1-vs-all` manner # => NOTE: => warnings() # complete or quasi-complete separation
      # High-frequency (annoying) "pop-up" messages
      if(verbose.messages) {
      message("Training Platt-scaling (Firth) on the sum of innerfold test/calibration sets ", i, ".", 1, " - ", i, ".", length(nfolds..), " @ ", Sys.time())
      message("nrow: ", nrow(scores.i.j.innertest.all), " ncol: ", ncol(scores.i.j.innertest.all))
      message("1-vs-all [class] ", diagn)
      } 
      platt.brglm.calfit <- subfunc_Platt_train_calibration_Firth(y.i.j.innertest = y.i.j.innertest.all, scores.i.j.innertest = scores.i.j.innertest.all, 
                                                                  diagn.class = diagn, brglm.control.max.iteration = brglm.ctrl.max.iter) # for 1 diagnosis 
      
      # 4. Predict the outerfold test set using the trained calibration model (Figure 1. green arrow "predict") to generate calibrated probabilities (Figure 1, yellow box P1.0)
      # High-frequency (annoying) "pop-up" messages 
      if(verbose.messages) message("Fitting Platt-scaled (Firth penalized) LR model on the raw scores of outer fold test set ", i, ".0", " @ ", Sys.time())
      probs.platt.firth.brglm.l[[c]] <- subfunc_Platt_fit_testset_Firth(y.i.0.outertest = y.i.0.outertest, scores.i.0.outertest = scores.i.0.outertest, 
                                                                        calib.model.Platt.Firth.diagn.i = platt.brglm.calfit, diagn.class = diagn) # length outer$test = 597
    }  
    probs <- do.call(cbind, probs.platt.firth.brglm.l) 
    colnames(probs) <- colnames(scores.i.j.innertest.all) # or levels(y..)

    # 5. Calculate misclassification errors (ME)
    # NOTE: the predicted class label is the one with the highest score | probability (no further thresholding is applied)
    errs <- sum(colnames(scores.i.0.outertest)[apply(scores.i.0.outertest, 1, which.max)] != y..[nfolds..[[i]][[1]][[1]]$test])/length(nfolds..[[i]][[1]][[1]]$test)
    errp <- sum(colnames(probs)[apply(probs, 1, which.max)] != y..[nfolds..[[i]][[1]][[1]]$test])/length(nfolds..[[i]][[1]][[1]]$test)
    # Print ME
    message("\nMisclassification error of tRF (", which.optimized.metric.or.algorithm, ") raw scores on the ", i, ".0 fold: ", errs)
    message("Misclassification error of Platt-FLR-calibrated RF (", which.optimized.metric.or.algorithm, ") probabilities on the ", i, ".0 fold: ", errp)
    
    # 6. Save raw classifier scores and calibrated probabilities into a separate folder and RData file
    folder.path <- file.path(getwd(), out.path)
    dir.create(folder.path, recursive = T, showWarnings = T) # warning if the folder exists
    # RData file
    scores <- scores.i.0.outertest # rewrites `scores` variable for saving to achieve a simpler naming convention
    if(save.metric.name.into.output.file) {
      save(probs, scores, errs, errp, file = file.path(folder.path, paste(out.fname, which.optimized.metric.or.algorithm, "FLR", i, 0, "RData", sep = ".")))
    } else {
      save(probs, scores, errs, errp, file = file.path(folder.path, paste(out.fname, "FLR", i, 0, "RData", sep = ".")))
    }
    
    
  } # for() outer loop i
}


# Function call  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Run time - single core (rMBP 15" i9) - 8-9mins 
# # NOTE: max itarations (brglm.ctrl.max.iter = 10 000) instead of default = 100.
#
# Sys.time() # 18:27:44 CEST
# calibrate_FLR(out.path = "vRF-calibrated-FLR/",
#                  out.fname = "probsCVfold",
#                  nfolds.. = NULL,
#                  y.. = NULL,
#                  load.path.w.name = "./vRF/CVfold.",
#                  which.optimized.metric.or.algorithm = "vanilla",
#                  verbose.messages = T)
# Sys.time() # 18:36:31 CEST"



