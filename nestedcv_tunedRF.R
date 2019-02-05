############################################################
###     RF HYPER PARAMETER TUNING                        ###
###         using CARET CUSTOM FUNCTION =                ###
###           1. mtry                                    ###
###           2. ntree                                   ###
###           3. nodesize                                ###
###           4. p (subset of variables/CpGs)            ###
###                                                      ###
###     PROTOTYPING SHOWED                               ###
###       mtry  = ~default (60-100) @ 10% steps          ### 
###       ntree = 500-2000  @ 500 steps                  ### 
###       nodesize = default (1, 5, 1%, 5%, 10%)         ###
############################################################

# Load pkgs
library(parallel)
library(doParallel) # to run glmnet in parallel
#library(foreach) # glmnet uses foreach for lambda search
#library(doMC)    # doMC: shared-memory parallelism # in my experience handles RAM often better
library(caret)
library(randomForest)

#########################################################################################################################################################################################################
### SOURCE SUBFUNCTIONS ###
###########################

# Source subfunctions and train function for tunedRF
source("subfunctions_tunedRF.R")   # rfp() customRF {CARET} 
source("train_tunedRF.R")          # Hyperparameter tuning , Variable selection, Re-fitting 

# Support functions
source("rfp.R")            
source("makefolds.R")
# Performance eval
source("brier.R")
source("mlogloss.R")

#########################################################################################################################################################################################################
### Load data ###
#################

load("MNPbetas10Kvar.RData") # contains betas and y 
load("nfolds.RData")

# Inspect the loaded objects
str(betas)
str(y)
str(nfolds)

#########################################################################################################################################################################################################
### Parallel backend variations ###
###################################
# Use all threads
cores <- detectCores()
# Leave one thread for the OS
cores <- detectCores()-1

#########################################################################################################################################################################################################
#####################################
### Settings for nestedcv_tunedRF ###
#####################################
cores <- detectCores()-1 # Ubuntu =10 threads
mc.cores <- cores
seed <- 1234

# Trim down betas (10000 seems to be too large)
class(betas)  # df CAVE: ==> glmnet needs a matrix input
betas.mtx <- as.matrix(betas)

# Subset betas (10 000 most variable CpG probes) to 1000 or 100
#betas100 <- betas.mtx[ , 1:100]
#betas100 <- betas[ ,1:100]
#betas1000 <- betas.mtx[ , 1:1000] 

#########################################################################################################################################################################################################    
################################################
###   Run K x k-fold nested CV for tunedRF   ###
################################################

run_nestedcv_tunedRF <- function(y.., betas.., n.cv.folds = 5, nfolds..=NULL, # load("nfolds.RData")
                                 cores = 10, seed = 1234, K.start = 1, k.start = 0, out.path = "tRF/", out.fname = "CVfold", 
                                 mtry.min = NULL, mtry.max = NULL, length.mtry = 2, # If mtry.min and mtry.max arguments are left at default (NULL). The function equally divides floor(sqrt(ncol(betas)))*0.5) and floor(sqrt(ncol(betas))) to length.mtry parts 
                                 ntrees.min = 500, ntrees.max = 2000, ntree.by = 500,
                                 nodesize.proc = c(0.01, 0.05, 0.1), # CRITICAL/Troubleshooting: The argument nodesize.proc requires three elements and adds the default values nodesize = 1 for classification and nodesize = 5 for regression in the subfunc_rf_caret_tuner_customRF()
                                 n.cv = 5, n.rep = 1,
                                 p.n.pred.var = c(100, 500, 1000, 10000)){
  
  # Check whether nfolds.. is provided
  if(is.null(nfolds..) & exists("nfolds")){nfolds.. <- get("nfolds", envir = .GlobalEnv)} else {
    stop("Please provide a fold structure for nested resampling. For instance, load the nfolds.RData")}
  
  for(K in K.start:n.cv.folds){
    
    for(k in k.start:n.cv.folds){ 
      
      if(k > 0){ message("\n \nCalculating inner/nested fold ", K,".", k,"  ... ",Sys.time())  # Inner CV loops 1.1-1.5 (Fig. 1.)
        fold <- nfolds..[[K]][[2]][[k]]  ### [[2]] means inner loop
      } else{                                                                          
        message("\n \nCalculating outer fold ", K,".0  ... ",Sys.time()) # Outer CV loops 1.0-5.0 (Fig. 1.)
        fold <- nfolds..[[K]][[1]][[1]]   ### [[]][[1]][[]] means outer loop 
      }
      
      # Print fold structure
      print("Fold structure", str(fold))
      
      message("Start tuning on training set using customRF function within CARET ... ", Sys.time())
      # trainRF
      rfcv.tuned <- trainRF_caret_custom_tuner(y. = y..[fold$train], betas. = betas..[fold$train, ], cores = cores, 
                                               mtry.min = mtry.min, mtry.max = mtry.max, length.mtry = length.mtry,
                                               ntrees.min = ntrees.min, ntrees.max = ntrees.max, ntree.by = ntree.by, 
                                               nodesize.proc = nodesize.proc, 
                                               n.cv = n.cv, n.rep =  n.rep,
                                               p = p.n.pred.var, 
                                               p.tuning.brier = T, p.tuning.miscl.err = T, p.tuning.mlogloss = T, 
                                               seed = seed+1, allowParallel = T)  
      
      # NOTE: rfcv.tuned variable contains res <- list(rf.pred.best.brier, rf.pred.best.err, rf.pred.best.mlogl, imp.perm, rf.pred.l, score.pred.l, brier.p.l, err.p.l, mlogloss.p.l, t0, t1, t2)
      
      # Use tuned random forest modell fit rfcv.tuned[[1]] = rf.pred.best => to predict the corresponding CALIBRATION or TEST SETS (if innerfold then calibration set ; if outer fold then test set)
      message("\nFit tuned random forests on : test set ", K, ".", k, " ... ", Sys.time())
      
      iterator <- 1:3 
      scores.pred.rf.tuned.l <- mclapply(seq_along(iterator), function(i){
        predict(object = rfcv.tuned[[i]], 
                newdata = betas..[fold$test, match(rownames(rfcv.tuned[[i]]$importance), colnames(betas..))], 
                type = "prob")
      }, mc.preschedule = T, mc.set.seed = T, mc.cores = length(iterator))
      
      # Rewrite into new variable for saving (not very drive/memory efficient)
      scores.pred.rf.tuned.brier <- scores.pred.rf.tuned.l[[1]]
      scores.pred.rf.tuned.miscerr <- scores.pred.rf.tuned.l[[2]]
      scores.pred.rf.tuned.mlogl <- scores.pred.rf.tuned.l[[3]]
      
      # Calculate Misclassification Errors (ME)
      err.scores.rf.tuned.brier <- sum(colnames(scores.pred.rf.tuned.brier)[apply(scores.pred.rf.tuned.brier, 1, which.max)] != y..[fold$test]) / length(fold$test) # Calculates ME for BS tunedRF
      err.scores.rf.tuned.miscerr <- sum(colnames(scores.pred.rf.tuned.miscerr)[apply(scores.pred.rf.tuned.miscerr, 1, which.max)] != y..[fold$test]) / length(fold$test) # Calculates ME for ME tunedRF
      err.scores.rf.tuned.mlogl <- sum(colnames(scores.pred.rf.tuned.mlogl)[apply(scores.pred.rf.tuned.mlogl, 1, which.max)] != y..[fold$test]) / length(fold$test) # Calculates ME for mLL tunedRF
      
      # Print MEs
      message("\nMisclassification error on [Test Set] CVfold.", K, ".", k, 
              "\n 1. Brier optimized model: ", err.scores.rf.tuned.brier,                   # scores.pred.rf.tuned.l[[1]]
              "; \n 2. Misclassif. error optimized model: ",  err.scores.rf.tuned.miscerr,  # scores.pred.rf.tuned.l[[2]]
              "; \n 3. Mlogloss optimized model: ", err.scores.rf.tuned.mlogl,              # scores.pred.rf.tuned.l[[3]]
              "; @ ", Sys.time())      
      
      # Create output directory  
      f.path <- file.path(getwd(), out.path)
      dir.create(f.path, showWarnings = F)
      # Save file as .RData
      save(scores.pred.rf.tuned.brier, scores.pred.rf.tuned.miscerr, scores.pred.rf.tuned.mlogl, rfcv.tuned, fold, 
           file =  paste0(f.path, paste("CVfold", K, k, "RData", sep = ".")) 
           )
      # CRITICAL/Troubleshooting: the output .RData file can be large, as it contains multiple copies of large matrices (2,801 x 10,000 approx. 215 MB each) 
      # adding up to 1 - 1.5 Gb. Hence, the complete nested CV scheme might require 40-50Gb free space on the respective drive.
      
    }
  }
  message("Finished ... ", Sys.time())
}  

#########################################################################################################################################################################################################    

# Testrun 
#Sys.time()
#run_nestedcv_tunedRF(y.. = y, betas.. = betas100, n.cv.folds = 5, #nfolds.. = nfolds,
#                     cores = 11, seed = 1234, 
#                     K.start = 1, k.start = 0, 
#                     mtry.min = 50, mtry.max = 100, 
#                     length.mtry = 2, p.n.pred.var = c(50, 100)))
#Sys.time()

 
