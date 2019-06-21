#----------------------------------------------------------------------
# Integrated Performance evaluator function 
#
#             - including row matching (for RF)                    
#             - column reordering (for SVMs)
#
# Mate Maros
# maros@uni-heidelberg.de
#
# 2019-04-16  
#--------------------------------------------------------------------


# Integrated Performance evaluator function -------------------------------------------------------------------------------------------------------------------------------------------------------

# Source evaluation metrics
# the required evaluation metrics (BS, ME, LL, AUC) and their libraries are loaded or installed if needed
source("evaluation_metrics.R")

# Define: Performance evaluator function (all metrics integrated) ---------------------------------------------------------------------------------------------------------------------------------

performance_evaluator <- function(load.path.folder = "./vRF",
                                  load.fname.stump = "CVfold",
                                  name.of.obj.to.load = NULL, # as.character # defaults to "probs"  # vRF "scores"
                                  nfolds.. = NULL,
                                  y.. = NULL, # defaults to BTMD obects
                                  anno.. = NULL,
                                  scale.rowsum.to.1 = T,
                                  reorder.columns = F,
                                  reorder.columns.svm.e1071 =F,
                                  reorder.rows = T, 
                                  misc.err = T, 
                                  multi.auc.HandTill2001 = T, 
                                  brier = T, 
                                  mlogLoss = T,
                                  verbose = T){
  # Start @ 
  if(verbose) {
    message("\nStart performance evaluation @ ", Sys.time())
  }
  message("\nFitting folder-file path: '", file.path(load.path.folder, load.fname.stump), "' ")
  
  # 1. Checks
  # Check whether nfolds.. is available
  if(is.null(nfolds..) & exists("nfolds")) { 
    message("\n `nfolds` nested CV scheme assignment was fetched from .GlobalEnv")
    nfolds.. <- get("nfolds", envir = .GlobalEnv)
  } else {
    stop("Please provide a fold structure for nested resampling. For instance, load the nfolds.RData")}
  
  # Check whether y true outcome is available
  if(is.null(anno..) & exists("anno")) {
    message(" `anno` data frame of `betas_v11.h5` was fetched from .GlobalEnv")
    anno.. <- get("anno", envir = .GlobalEnv) 
  } else {
    stop("Please provide the rownames (anno$sentrix) labels vector for further analyses. For instance, load the `anno.RData`")}
  
  if(is.null(y..) & exists("y")){y.. <- get("y", envir = .GlobalEnv)} else {
    stop("Please provide the true class labels vector (y) for further analyses. For instance, load the `y.RData`")}
  
  # Check whether provided betas../anno..$sentrix and y.. has the same dimension
  if(length(anno..$sentrix) == length(y..)) { 
    message("Checked: `anno..` and `y..` have the same corresponding dimensions. OK. \n")
  } else {
    stop("Error: anno.. and y.. have different dimensions.")}
  
  # Set default "probs" for name.of.obj.to.load
  if(is.null(name.of.obj.to.load)) name.of.obj.to.load <- "probs"
  
  # Load objects with scores/probes
  probs.l <- list()
  for(i in 1:length(nfolds..)){
    # Create loading environment for safe loading the RData files of outerfolds
    env2load.outer <- environment()
    # Load
    if(verbose) message("\nLoad the test set from the ", i, ".0 outer fold @ ", Sys.time())
    path2load <- file.path(load.path.folder) 
    fname2load <- file.path(path2load, paste(load.fname.stump, i, 0, "RData", sep = "."))
    load(fname2load, envir = env2load.outer)                                        
    probs.loaded <- get(as.character(name.of.obj.to.load), envir = env2load.outer)
    if(reorder.columns.svm.e1071) {
      if(verbose) message("  Are predicted class labels (colnames) identically ordered as levels of the true outcome (y..) in ", i, ".0 fold: ", identical(colnames(probs.loaded), levels(y..)))
      message("  Reorder colnames of score/probability output to levels(y..) ... ")
      probs.loaded <- probs.loaded[ , levels(y..)]
      if(verbose) message("  Check whether colnames are identically ordered after re-matching: ", identical(colnames(probs.loaded), levels(y..)))
    }  
    probs.l[[i]] <-  probs.loaded
  }
  
  # Patch the score & prob list together as a matrix
  if(verbose) message("\nCombine loaded outer fold test sets  1.0 ... ", length(nfolds..), ".0")
  probs <- do.call(rbind, probs.l)
  if(verbose) message("Dimensions of the combined matrix: ", nrow(probs), " x ", ncol(probs))
  
  # Visual inspection of row and column levels (e.g. SVM mixes the order of both)
  #print(rownames(probs)[1:5])
  #print(rownames(betas..)[1:5])
  
  if(verbose) message("\nAre predicted class labels (colnames) identically ordered as levels of the true outcome (y..): ", identical(colnames(probs), levels(y..)))
  # Reorder columns if input is not from SVM (e1071) package (FALSE by default)
  if(reorder.columns){
    message("Reorder colnames of score/probability output to levels(y..) ... ")
    probs <- probs[ , levels(y..)] # probs.pred.SVM.e1071.mtx.reordered <- probs.pred.SVM.e1071.mtx[ , levels(y)]
    if(verbose) message("Check whether colnames are identically ordered after re-matching: ", identical(colnames(probs), levels(y..)))
  }
  
  # Reorder cases/patients as in anno.. 
  if(reorder.rows){
    message("\nReorder sample (i.e. row) names as in anno..$sentrix & y (true outcome)")   
    probs <- probs[match(anno..$sentrix, rownames(probs)), ]
  }
  message("Check whether rownames identical:", identical(rownames(probs), anno..$sentrix))
  
  # Rowsum check
  rowsum.p <- apply(probs, 1, sum)
  if(verbose) message("\nRow sum of first 10 cases :", paste(as.matrix(rowsum.p[1:10]), collapse = " ; "))
  # Show scores / probs ranges:
  p.range <- range(probs)
  if(verbose) message("The range of raw and/or calibrated object", " '", as.character(name.of.obj.to.load), "' ", 
                      "is : ", "\n", p.range[[1]], " - ", p.range[[2]])
  
  # Scale rowsum to 1 - default
  if(scale.rowsum.to.1){
    if(verbose) message("Scaling row sums to 1 (by default = TRUE): ")
    probs.rowsum1 <- t(apply(probs, 1, function(x) x/sum(x)))
  } else{probs.rowsum1 <-  probs}
  
  # Predicted class column with highest prob for each row (patient)
  y.p.rowsum1 <- colnames(probs.rowsum1)[apply(probs.rowsum1,1, which.max)] 
  y.l <- list(y.p.rowsum1)
  # Misclassification Error
  if(misc.err){
    err.misc.l <- lapply(y.l, subfunc_misclassification_rate, y.true.class = y..) 
    err.misc <- unlist(err.misc.l)
    message("\nMisclassification Error: ", err.misc)
  }
  # AUC HandTIll2001
  if(multi.auc.HandTill2001){
    results.sc.p.rowsum1.l <- list(probs.rowsum1)
    message("Calculating multiclass AUC (Hand&Till 2001) ... ", Sys.time())
    auc.HT2001.l <- lapply(results.sc.p.rowsum1.l, subfunc_multiclass_AUC_HandTill2001, y.true.class = y..)
    auc.HT2001 <- unlist(auc.HT2001.l)
    message("Multiclass AUC (Hand&Till 2001): ", auc.HT2001)
  } else{
    message("Multiclass AUC (Hand&Till 2001) is set to FALSE =>", 
            "Calculating RAW probs (row sum != 1). auc.HT2001 is set to NA.")
    auc.HT2001 <- NA
  }
  # Brier 
  if(brier){
    brierp.rowsum1 <- brier(scores = probs.rowsum1, y = y..)  
    message("Brier score (BS): ", brierp.rowsum1)
  }
  # mlogloss
  if(mlogLoss){
    loglp.rowsum1 <- mlogloss(scores = probs.rowsum1, y = y..)
    message("Multiclass log loss (LL): ", loglp.rowsum1)
  }
  # End of run
  message("Finished @ ", Sys.time())
  
  # Results
  res <- list(misc.error = err.misc, 
              auc.HandTill = auc.HT2001, 
              brier = brierp.rowsum1, 
              mlogloss = loglp.rowsum1)
  return(res)
} # end of func         


# Examples of function calls and outputs -------------------------------------------------------------------------------------------------------------------------------------------------------
# Uncomment the respective segment below by using the keyboard shortcuts (cmd+shift+c | ctrl+shitf+c) 
# 1.   Rgtsvm
# 2-3. SVM-LK e1071
# 4.   vRF
# 5.   tRF


# Function calls: for SVM-Rgtsvm -------------------------------------------------------------------------------------------------------------------------------------------------------

# # Load prereq. objects
# rMBP 15" 
# load("/Users/mme/Documents/GitHub/np-data/data/anno.RData")
# load("/Users/mme/Documents/GitHub/np-data/data/probenames.RData")
# load("/Users/mme/Documents/GitHub/np-data/data/y.RData")
# load("/Users/mme/Documents/GitHub/np-data/data/nfolds.RData")

# # 1. Rgtsvm - NO SCALING - because already scaled - (probs) -------------------------------------------------------------------------------------------------------------------------------------------------------
#SVM_Rgtsvm_perfeval <- performance_evaluator(load.path.folder = "./SVM-Rgtsvm/", load.fname.stump = "CVfold", name.of.obj.to.load = "scores.pred.svm.Rgtsvm")
# save(SVM_Rgtsvm_perfeval, file = file.path("./performance evaluation/SVM_Rgtsvm_perfeval.RData"))
# score range: 1.25359853475269e-13 - 0.999991113024383

# # 2. e1071 raw (scores) -------------------------------------------------------------------------------------------------------------------------------------------------------
# SVM_e1071_perfeval <- performance_evaluator(load.path.folder = "./SVM-e1071-10k/", load.fname.stump = "CVfold", name.of.obj.to.load = "scores.pred.svm.e1071.mtx", 
#                                             reorder.columns.svm.e1071 = T, # IMPORTANT! 1-vs-1 coupling changes the order of columns for each K.k (sub)fold differently => without this results are poor
#                                             reorder.columns = F,  
#                                             reorder.rows = T)
#save(SVM_e1071_perfeval, file = file.path("./performance evaluation/SVM_e1071_perfeval.RData"))
#score range: 5.08402283422852e-05 - 0.965712778864327

### Run all for each metric and all calibration algorithms -------------------------------------------------------------------------------------------------------------------------------------------------------
# # 3. e1071 + LR + FLR + MR (scores) ------------------------------------------------------------------------------------------------------------------------------------------------------- 

### Run all for each metric and all calibration algorithms -------------------------------------------------------------------------------------------------------------------------------------------------------
# try.package.name <- "e1071"
# try.folder.path <- as.list(paste("SVM", try.package.name, "calibrator-integrated", sep = "-"))
# try.algorithm <- rep(c("svm"), each = 3)
# try.calibrator <- c("LR", "FLR", "MR")
# try.fname.stump <- as.list(paste("probsCVfold", try.algorithm, rep(try.calibrator), sep = "."))
# try.fname.stump
# 
# # Short call using mapply
# try.l.folder.file.paths <- file.path(try.folder.path, try.fname.stump)
# try.l.perfevals.all.SVM.e1071 <- mapply(FUN = performance_evaluator, load.path.folder = try.folder.path, load.fname.stump = try.fname.stump)
# try.l.perfevals.all.SVM.e1071 # matrix
# # Add colnames
# colnames(try.l.perfevals.all.SVM.e1071) <- file.path(try.folder.path, try.fname.stump)
# colnames(try.l.perfevals.all.SVM.e1071) <- try.fname.stump
# # Rename
# SVM_e1071_ALL_LR_FLR_MR_perfeval <- try.l.perfevals.all.SVM.e1071

# Save
# #save(SVM_e1071_ALL_LR_FLR_MR_perfeval, file = "./performance evaluation/SVM_e1071_ALL_LR_FLR_MR_perfeval.RData")

# try.l.perfevals.all.SVM.e1071 <- mapply(FUN = performance_evaluator, load.path.folder = try.folder.path, load.fname.stump = try.fname.stump)
# 
# Start performance evaluation @ 2019-04-30 17:27:59
# 
# Fitting folder-file path: 'SVM-e1071-calibrator-integrated/probsCVfold.svm.LR' 
# 
# `nfolds` nested CV scheme assignment was fetched from .GlobalEnv
# `anno` data frame of `betas_v11.h5` was fetched from .GlobalEnv
# Checked: `anno..` and `y..` have the same corresponding dimensions. OK. 
# 
# 
# Load the test set from the 1.0 outer fold @ 2019-04-30 17:27:59
# 
# Load the test set from the 2.0 outer fold @ 2019-04-30 17:27:59
# 
# Load the test set from the 3.0 outer fold @ 2019-04-30 17:27:59
# 
# Load the test set from the 4.0 outer fold @ 2019-04-30 17:27:59
# 
# Load the test set from the 5.0 outer fold @ 2019-04-30 17:27:59
# 
# Combine loaded outer fold test sets  1.0 ... 5.0
# Dimensions of the combined matrix: 2801 x 91
# 
# Are predicted class labels (colnames) identically ordered as levels of the true outcome (y..): TRUE
# 
# Reorder sample (i.e. row) names as in anno..$sentrix & y (true outcome)
# Check whether rownames identical:TRUE
# 
# Row sum of first 10 cases :1.01693871339887 ; 1.01319989097865 ; 1.01312669512845 ; 1.01316695506381 ; 1.01257346013941 ; 1.02158778935526 ; 1.02870305717123 ; 1.01310568504558 ; 1.01658753986284 ; 1.01628377950492
# The range of raw and/or calibrated object 'probs' is : 
#   2.22044604925031e-16 - 1
# Scaling row sums to 1 (by default = TRUE): 
#   
#   Misclassification Error: 0.0253480899678686
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-30 17:27:59
# Multiclass AUC (Hand&Till 2001): 0.999053463839414
# Brier score (BS): 0.0429358200055616
# Multiclass log loss (LL): 0.111580603185538
# Finished @ 2019-04-30 17:28:21
# 
# Start performance evaluation @ 2019-04-30 17:28:21
# 
# Fitting folder-file path: 'SVM-e1071-calibrator-integrated/probsCVfold.svm.FLR' 
# 
# `nfolds` nested CV scheme assignment was fetched from .GlobalEnv
# `anno` data frame of `betas_v11.h5` was fetched from .GlobalEnv
# Checked: `anno..` and `y..` have the same corresponding dimensions. OK. 
# 
# 
# Load the test set from the 1.0 outer fold @ 2019-04-30 17:28:21
# 
# Load the test set from the 2.0 outer fold @ 2019-04-30 17:28:21
# 
# Load the test set from the 3.0 outer fold @ 2019-04-30 17:28:21
# 
# Load the test set from the 4.0 outer fold @ 2019-04-30 17:28:21
# 
# Load the test set from the 5.0 outer fold @ 2019-04-30 17:28:21
# 
# Combine loaded outer fold test sets  1.0 ... 5.0
# Dimensions of the combined matrix: 2801 x 91
# 
# Are predicted class labels (colnames) identically ordered as levels of the true outcome (y..): TRUE
# 
# Reorder sample (i.e. row) names as in anno..$sentrix & y (true outcome)
# Check whether rownames identical:TRUE
# 
# Row sum of first 10 cases :1.03022889203968 ; 1.02351881228286 ; 1.02306576249891 ; 1.02318473952594 ; 1.02245328644475 ; 1.03622750326617 ; 1.04584814573852 ; 1.02286532540742 ; 1.02948355375492 ; 1.02968429049641
# The range of raw and/or calibrated object 'probs' is : 
#   2.22044604925031e-16 - 1
# Scaling row sums to 1 (by default = TRUE): 
#   
#   Misclassification Error: 0.020706890396287
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-30 17:28:21
# Multiclass AUC (Hand&Till 2001): 0.999145190720591
# Brier score (BS): 0.0436517337289733
# Multiclass log loss (LL): 0.134862810246374
# Finished @ 2019-04-30 17:28:43
# 
# Start performance evaluation @ 2019-04-30 17:28:43
# 
# Fitting folder-file path: 'SVM-e1071-calibrator-integrated/probsCVfold.svm.MR' 
# 
# `nfolds` nested CV scheme assignment was fetched from .GlobalEnv
# `anno` data frame of `betas_v11.h5` was fetched from .GlobalEnv
# Checked: `anno..` and `y..` have the same corresponding dimensions. OK. 
# 
# 
# Load the test set from the 1.0 outer fold @ 2019-04-30 17:28:43
# 
# Load the test set from the 2.0 outer fold @ 2019-04-30 17:28:43
# 
# Load the test set from the 3.0 outer fold @ 2019-04-30 17:28:43
# 
# Load the test set from the 4.0 outer fold @ 2019-04-30 17:28:43
# 
# Load the test set from the 5.0 outer fold @ 2019-04-30 17:28:43
# 
# Combine loaded outer fold test sets  1.0 ... 5.0
# Dimensions of the combined matrix: 2801 x 91
# 
# Are predicted class labels (colnames) identically ordered as levels of the true outcome (y..): TRUE
# 
# Reorder sample (i.e. row) names as in anno..$sentrix & y (true outcome)
# Check whether rownames identical:TRUE
# 
# Row sum of first 10 cases :1 ; 1 ; 1 ; 1 ; 1 ; 1 ; 1 ; 1 ; 1 ; 1
# The range of raw and/or calibrated object 'probs' is : 
#   7.09741404553869e-25 - 0.999999999999958
# Scaling row sums to 1 (by default = TRUE): 
#   
#   Misclassification Error: 0.0214209210996073
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-30 17:28:43
# Multiclass AUC (Hand&Till 2001): 0.999909351510176
# Brier score (BS): 0.0388549815738284
# Multiclass log loss (LL): 0.0848454571143461
# Finished @ 2019-04-30 17:29:04


# 4. Function calls: for vRF -------------------------------------------------------------------------------------------------------------------------------------------------------

# # Load prereq. objects
# load("/Users/mme/Documents/GitHub/np-data/data/anno.RData")
# load("/Users/mme/Documents/GitHub/np-data/data/probenames.RData")
# load("/Users/mme/Documents/GitHub/np-data/data/y.RData")
# load("/Users/mme/Documents/GitHub/np-data/data/nfolds.RData")
# 
# # Generate performance evaluation metrics - Run time - 30s
# 
# # 1. # Vanilla RF (vRF) -------------------------------------------------------------------------------------------------------------------------------------------------------
# vRF_perfeval <- performance_evaluator(load.path.folder = "./vRF", load.fname.stump = "CVfold", name.of.obj.to.load = "scores")
# # Save output list 
# save(vRF_perfeval, file = file.path("./performance evaluation/vRF_perfeval.RData"))
# 
# # 2. # vRF + Platt-LR calibrated - vRF + LR ------------------------------------------------------------------------------------------------------------------------------------
# vRF_LR_perfeval <- performance_evaluator(load.path.folder = "./vRF-calibrated-LR/", load.fname.stump = "probsCVfold.vanilla", name.of.obj.to.load = "probs")
# # Save output list 
# save(vRF_LR_perfeval, file = file.path("./performance evaluation/vRF_LR_perfeval.RData"))
# 
# # 2.1. # vRF + LR - row sum NOT SCALED = 1 
# vRF_LR_perfeval_rowsum_not_1 <- performance_evaluator(load.path.folder = "./vRF-calibrated-LR/", load.fname.stump = "probsCVfold.vanilla", name.of.obj.to.load = "probs", 
#                                                       scale.rowsum.to.1 = F, multi.auc.HandTill2001 = F)
# # Save output list 
# save(vRF_LR_perfeval_rowsum_not_1, file = file.path("./performance evaluation/vRF_LR_perfeval_rowsum_not_1.RData"))
# 
# # 3. # vRF + Platt-FLR calibrated - vRF + FLR ---------------------------------------------------------------------------------------------------------------------------------
# vRF_FLR_perfeval <- performance_evaluator(load.path.folder = "./vRF-calibrated-FLR/", load.fname.stump = "probsCVfold.vanilla", name.of.obj.to.load = "probs")
# # Save output list 
# save(vRF_FLR_perfeval, file = file.path("./performance evaluation/vRF_FLR_perfeval.RData"))
# 
# # 3.1. # vRF + FLR - row sum NOT SCALED = 1 
# vRF_FLR_perfeval_rowsum_not_1 <- performance_evaluator(load.path.folder = "./vRF-calibrated-FLR/", load.fname.stump = "probsCVfold.vanilla", name.of.obj.to.load = "probs", 
#                                                       scale.rowsum.to.1 = F, multi.auc.HandTill2001 = F)
# # Save output list 
# save(vRF_FLR_perfeval_rowsum_not_1, file = file.path("./performance evaluation/vRF_FLR_perfeval_rowsum_not_1.RData"))
# 
# # 4. # vRF + MR calibrated - vRF + MR ----------------------------------------------------------------------------------------------------------------------------------------
# vRF_MR_perfeval <- performance_evaluator(load.path.folder = "./vRF-calibrated-MR/", load.fname.stump = "probsCVfold.vanilla", name.of.obj.to.load = "probs")
# # Save output list 
# save(vRF_MR_perfeval, file = file.path("./performance evaluation/vRF_MR_perfeval.RData"))


# 5. Function calls: for tRF ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load prereq. objects
# list.files("/Users/mme/Documents/GitHub/np-tRF/tRF/")
# load("/Users/mme/Documents/GitHub/np-data/data/anno.RData")
# load("/Users/mme/Documents/GitHub/np-data/data/probenames.RData")
# load("/Users/mme/Documents/GitHub/np-tRF/tRF/")

# Saved Objects within RData
# scores.pred.rf.tuned.brier, scores.pred.rf.tuned.miscerr, scores.pred.rf.tuned.mlogl, 

# # Generate performance evaluation metrics - Run time - 30s
# 
# # 1. # tuned RF (tRF) {BS ; ME; LL} - NO SCALING - (scores) -------------------------------------------------------------------------------------------------------------------------------------------------------
# tRF_BS_perfeval <- performance_evaluator(load.path.folder = "./tRF", load.fname.stump = "CVfold", name.of.obj.to.load = "scores.pred.rf.tuned.brier")
# save(tRF_BS_perfeval, file = file.path("./performance evaluation/tRF_BS_perfeval.RData"))
# # score range: 0 - 0.972739361702128
# 
# tRF_ME_perfeval <- performance_evaluator(load.path.folder = "./tRF", load.fname.stump = "CVfold", name.of.obj.to.load = "scores.pred.rf.tuned.miscerr")
# save(tRF_ME_perfeval, file = file.path("./performance evaluation/tRF_ME_perfeval.RData"))
# # score range: 0 - 0.880984042553192
# 
# tRF_LL_perfeval <- performance_evaluator(load.path.folder = "./tRF", load.fname.stump = "CVfold", name.of.obj.to.load = "scores.pred.rf.tuned.mlogl")
# save(tRF_LL_perfeval, file = file.path("./performance evaluation/tRF_LL_perfeval.RData"))
# # score range: 0 - 0.986037234042553


### Run all for each metric and all calibration algorithms -------------------------------------------------------------------------------------------------------------------------------------------------------
# try.metrics <- c("BS", "ME", "LL")
# try.folder.path <- as.list(rep(paste("tRF", try.metrics, "calibrator-integrated", sep = "-"), each = 3))
# try.metrics.long <- rep(c("brier", "miscerr", "mlogl"), each = 3)
# try.calibrator <- c("LR", "FLR", "MR")
# try.fname.stump <- as.list(paste("probsCVfold", try.metrics.long, rep(try.calibrator, 3), sep = "."))

# Short call using mapply
# try.l.folder.file.paths <- file.path(try.folder.path, try.fname.stump)
# try.l.perfevals.all <- mapply(FUN = performance_evaluator, load.path.folder = try.folder.path, load.fname.stump = try.fname.stump)
# try.l.perfevals.all # matrix
# # Add colnames
# colnames(try.l.perfevals.all) <- file.path(try.folder.path, try.fname.stump)
# colnames(try.l.perfevals.all) <- try.fname.stump
# # Rename
# tRF_ALL_LR_FLR_MR_perfeval <- try.l.perfevals.all
# Save
# #save(tRF_ALL_LR_FLR_MR_perfeval, file = "./performance evaluation/tRF_ALL_LR_FLR_MR_perfeval.RData")


# Function output - Full run time ca. 3-4 min > LR + FLR @ single core MRl 11 cores i9 rMBP ----------------------------------------------------------------------------------------------------------------------
# 1. tRF-BS 
# 1.1 tRF-BS + LR (Platt-LR) ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Start performance evaluation @ 2019-04-27 09:04:16
# Fitting folder-file path: 'tRF-BS-calibrator-integrated/probsCVfold.brier.LR'
# The range of raw and/or calibrated object 'probs' is : 2.22044604925031e-16 - 1
# Misclassification Error: 0.0560514102106391
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-27 09:04:16
# Multiclass AUC (Hand&Till 2001): 0.996604957179769
# Brier score (BS): 0.0863397351563208
# Multiclass log loss (LL): 0.266366430678506
# Finished @ 2019-04-27 09:04:37
# 
# # 1.2 tRF-BS + FLR (Platt-Firth-FLR) -------------------------------------------------------------------------------------------------------------------------------------------------------
# Start performance evaluation @ 2019-04-27 09:04:37
# Fitting folder-file path: 'tRF-BS-calibrator-integrated/probsCVfold.brier.FLR' 
# The range of raw and/or calibrated object 'probs' is : 2.22044604925031e-16 - 1
# Misclassification Error: 0.0542663334523385
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-27 09:04:37
# Multiclass AUC (Hand&Till 2001): 0.999682168091817
# Brier score (BS): 0.0860258464686398
# Multiclass log loss (LL): 0.194278323948679
# Finished @ 2019-04-27 09:04:59
# 
# # 1.3 tRF-BS + MR (ridge-MR) --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Start performance evaluation @ 2019-04-27 09:04:59
# Fitting folder-file path: 'tRF-BS-calibrator-integrated/probsCVfold.brier.MR' 
# The range of raw and/or calibrated object 'probs' is :  1.63359987331075e-15 - 0.999999975437933
# Misclassification Error: 0.0510531952873974
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-27 09:04:59
# Multiclass AUC (Hand&Till 2001): 0.999713715730864
# Brier score (BS): 0.08260337504529
# Multiclass log loss (LL): 0.175765980663776
# Finished @ 2019-04-27 09:05:20
# 
# # 2. tRF-ME 
# # 2.1 tRF-ME + LR (Platt-LR) -------------------------------------------------------------------------------------------------------------------------------------------------------
# Start performance evaluation @ 2019-04-27 09:05:20
# Fitting folder-file path: 'tRF-ME-calibrator-integrated/probsCVfold.miscerr.LR' 
# The range of raw and/or calibrated object 'probs' is : 2.22044604925031e-16 - 1
# Misclassification Error: 0.0417707961442342
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-27 09:05:20
# Multiclass AUC (Hand&Till 2001): 0.997674754080271
# Brier score (BS): 0.0620031812933429
# Multiclass log loss (LL): 0.15568772911359
# Finished @ 2019-04-27 09:05:40
# 
# # 2.2 tRF-ME + FLR (Platt-Firth-FLR) -------------------------------------------------------------------------------------------------------------------------------------------------------
# Start performance evaluation @ 2019-04-27 09:05:40
# Fitting folder-file path: 'tRF-ME-calibrator-integrated/probsCVfold.miscerr.FLR' 
# The range of raw and/or calibrated object 'probs' is : 2.22044604925031e-16 - 1
# Misclassification Error: 0.0374866119243127
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-27 09:05:40
# Multiclass AUC (Hand&Till 2001): 0.999494150778859
# Brier score (BS): 0.0617565289651846
# Multiclass log loss (LL): 0.150068845737808
# Finished @ 2019-04-27 09:06:01
# 
# # 2.3 tRF-ME + MR (ridge-MR) -------------------------------------------------------------------------------------------------------------------------------------------------------
# Start performance evaluation @ 2019-04-27 09:06:01
# Fitting folder-file path: 'tRF-ME-calibrator-integrated/probsCVfold.miscerr.MR'
# The range of raw and/or calibrated object 'probs' is : 1.37766798531274e-19 - 0.999999932022973
# Misclassification Error: 0.0271331667261692
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-27 09:06:01
# Multiclass AUC (Hand&Till 2001): 0.999920471210109
# Brier score (BS): 0.0460707671932824
# Multiclass log loss (LL): 0.0946644960491068
# Finished @ 2019-04-27 09:06:22
# 
# # 3. tRF-LL 
# # 3.1 tRF-LL + LR (Platt-LR) -------------------------------------------------------------------------------------------------------------------------------------------------------
# Start performance evaluation @ 2019-04-27 09:06:22
# Fitting folder-file path: 'tRF-LL-calibrator-integrated/probsCVfold.mlogl.LR' 
# The range of raw and/or calibrated object 'probs' is : 2.22044604925031e-16 - 1
# Misclassification Error: 0.0581935023205998
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-27 09:06:22
# Multiclass AUC (Hand&Till 2001): 0.995144371098635
# Brier score (BS): 0.0892489953083336
# Multiclass log loss (LL): 0.290836681398512
# Finished @ 2019-04-27 09:06:42
# 
# # 3.2 tRF-LL + FLR (Platt-Firth-FLR) -------------------------------------------------------------------------------------------------------------------------------------------------------
# Start performance evaluation @ 2019-04-27 09:06:42
# Fitting folder-file path: 'tRF-LL-calibrator-integrated/probsCVfold.mlogl.FLR' 
# The range of raw and/or calibrated object 'probs' is : 9.20272906609289e-07 - 1
# Misclassification Error: 0.0564084255622992
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-27 09:06:43
# Multiclass AUC (Hand&Till 2001): 0.99966914211173
# Brier score (BS): 0.0889350682822319
# Multiclass log loss (LL): 0.205378849489624
# Finished @ 2019-04-27 09:07:03
# 
# # 3.3 tRF-LL + MR (ridge-MR) -------------------------------------------------------------------------------------------------------------------------------------------------------
# Start performance evaluation @ 2019-04-27 09:07:03
# Fitting folder-file path: 'tRF-LL-calibrator-integrated/probsCVfold.mlogl.MR' 
# The range of raw and/or calibrated object 'probs' is : 2.32138658779454e-14 - 0.999999944055397
# Misclassification Error: 0.0549803641556587
# Calculating multiclass AUC (Hand&Till 2001) ... 2019-04-27 09:07:03
# Multiclass AUC (Hand&Till 2001): 0.999698532020897
# Brier score (BS): 0.0864587148277675
# Multiclass log loss (LL): 0.187512548532441
# Finished @ 2019-04-27 09:07:24
# 
# # try.l.perfevals.all # matrix
# # probsCVfold.brier.LR probsCVfold.brier.FLR probsCVfold.brier.MR probsCVfold.miscerr.LR probsCVfold.miscerr.FLR probsCVfold.miscerr.MR probsCVfold.mlogl.LR probsCVfold.mlogl.FLR probsCVfold.mlogl.MR
# # misc.error   0.05605141           0.05426633            0.0510532            0.0417708              0.03748661              0.02713317             0.0581935            0.05640843            0.05498036          
# # auc.HandTill 0.996605             0.9996822             0.9997137            0.9976748              0.9994942               0.9999205              0.9951444            0.9996691             0.9996985           
# # brier        0.08633974           0.08602585            0.08260338           0.06200318             0.06175653              0.04607077             0.089249             0.08893507            0.08645871          
# # mlogloss     0.2663664            0.1942783             0.175766             0.1556877              0.1500688               0.0946645              0.2908367            0.2053788             0.1875125  
# 

