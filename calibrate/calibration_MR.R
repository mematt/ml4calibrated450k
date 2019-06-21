#------------------------------------------------------------------------
# calibrate RF (vRF & tRF) scores with mutlinomial ridge regression (MR)
#
#     tRF tuned for >
#       - Brier score (BS),
#       - misclassification error (ME), 
#       - multiclass log loss (LL)
#
#     vRF > 
#       - vanilla (no hyperparam. tuning)
#
# Mate Maros
# maros@uni-heidelberg.de
#
# 2019-04-25 UTC 
#------------------------------------------------------------------------

if (!requireNamespace("glmnet", quietly = TRUE)) { 
  install.packages("glmnet", dependencies = T)
  library(glmnet) 
} else {library(glmnet)}

# Define `calibrate_MR()` function --------------------------------------------------------------------------------------------------------------------------------

calibrate_MR <- function(out.path = "vRF-calibrated-MR/", 
                         out.fname = "probsCVfold",
                         nfolds.. = NULL,
                         y.. = NULL,
                         load.path.w.name = "./vRF/CVfold.", # default output of the `run_nestedcv_tunedRF()` function
                         which.optimized.metric.or.algorithm = c("brier", "miscerr", "mlogl", "vanilla", "svm", "xgboost"),
                         save.metric.name.into.output.file = T, 
                         verbose.messages = F,
                         parallel.cv.glmnet = T,
                         setseed = 1234){
  
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
    
    # 3. Train calibration model: multinomial ridge regression (MR) on all the combined inner fold calibration sets (Figure 1., green rectangle)
    message("\nTraining the multinomial ridge (MR) calbriation model on the sum of innerfold test/calibration sets ", i, ".", 1, " - ", i, ".", length(nfolds..), " @ ", Sys.time())     
    # Fit multinomial ridge regression (MR) model
    set.seed(setseed, kind = "L'Ecuyer-CMRG")
    suppressWarnings( # Ignore warnings of glmnet: "one multinomial or binomial class has fewer than 8  observations; dangerous ground"
      cv.calfit <- cv.glmnet(y = y.i.j.innertest.all,
                             x = scores.i.j.innertest.all,
                             family = "multinomial",
                             type.measure = "mse",        
                             alpha = 0, # corresponds to ridge penalized multinomial regression
                             nlambda = 100, lambda.min.ratio = 10^-6, # default values
                             parallel = parallel.cv.glmnet) # uses foreach under the hood 
    ) 
    
    # 4. Predict the outerfold test set using the trained calibration model (Figure 1. green arrow "predict") to generate calibrated probabilities (Figure 1, green box P1.0)
    message("Fitting the trained calibration model (MR) on the raw scores of outer fold test set ", i, ".0", " @ ",Sys.time())
    probs <- predict(cv.calfit$glmnet.fit,          # get the trained model object
                     newx = scores.i.0.outertest,   # predict on the i-th outer test set (Figure 1. red box)
                     type = "response", 
                     s = cv.calfit$lambda.1se)[,,1] # use lambda estimated by (default) 10-fold CV   
    # use lambda.1se instead of .min => more robust estimates
    # TROUBLESHOOTING: [,,1] is needed because `glmnet::predict.cv.glmnet()` generates its output as an array
    
    # Calculate misclassification errors (ME)
    # NOTE: the predicted class label is the one with the highest score | probability (no further thresholding is applied)
    errs <- sum(colnames(scores.i.0.outertest)[apply(scores.i.0.outertest, 1, which.max)] != y..[nfolds..[[i]][[1]][[1]]$test])/length(nfolds..[[i]][[1]][[1]]$test)
    errp <- sum(colnames(probs)[apply(probs, 1, which.max)] != y..[nfolds..[[i]][[1]][[1]]$test])/length(nfolds..[[i]][[1]][[1]]$test)
    # Print ME
    message("\nMisclassification error of tRF (", which.optimized.metric.or.algorithm, ") raw scores on the ", i, ".0 fold: ", errs)
    message("Misclassification error of MR-calibrated tRF (", which.optimized.metric.or.algorithm, ") probabilities on the ", i, ".0 fold: ", errp)
    
    # Save raw classifier scores and calibrated probabilities into a separate folder and RData file
    folder.path <- file.path(getwd(), out.path)
    dir.create(folder.path, recursive = T, showWarnings = T) # warning if the folder exists
    # RData file
    scores <- scores.i.0.outertest # rewrites `scores` variable for saving to achieve a simpler naming convention
    if(save.metric.name.into.output.file) {
      save(probs, scores, errs, errp, file = file.path(folder.path, paste(out.fname, which.optimized.metric.or.algorithm, "MR", i, 0, "RData", sep = ".")))
    } else {
      save(probs, scores, errs, errp, file = file.path(folder.path, paste(out.fname, "MR", i, 0, "RData", sep = ".")))
    }
    
  } # for() outer loop i
}
# 
# # Function call ----------------------------------------------------------------------------------------------------
# library(glmnet)
# library(doMC)
# #n_threads <- detectCores()-1
# #n_threads
# #registerDoMC(cores = n_threads)
# registerDoMC(cores = 11)
# getDoParWorkers()
# getDoParVersion()
# getDoParRegistered()
# 
# # vRF + MR ----------------------------------------------------------------------------------------------------
# # Run time 7-8mins @ 10 threads 
# Sys.time() 
# calibrate_RF_MR(out.path = "/vRF-calibrated-MR/", 
#                  out.fname = "probsCVfold",
#                  nfolds.. = NULL,
#                  y.. = NULL,
#                  load.path.w.name = "./vRF/CVfold.", 
#                  verbose.messages = T,
#                  which.optimized.metric.or.algorithm = "vanilla",
#                  parallel.cv.glmnet = T,
#                  setseed = 1234)
# Sys.time()
# 
# 
# # XGBOOST + MR ----------------------------------------------------------------------------------------------------
# 
# t_calib_MR <- system.time(
#   calibrate_RF_XGB_MR(out.path = "XGB-calibrated-MR", 
#                       load.path.w.name = "./XGBOOST/CVfold.", 
#                       which.optimized.metric.or.algorithm = "xgboost", 
#                       verbose.messages = T, 
#                       parallel.cv.glmnet = T, 
#                       setseed = 1234)
# )
# 
# # t_calib_MR
# # user   system  elapsed 
# # 2759.784   30.337  470.298  # 7-8 mins