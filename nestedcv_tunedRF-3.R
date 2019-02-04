############################################################
###     RF HYPER PARAMETER TUNING                        ###
###         using CARET MY CUSTOM FUNCTION =             ###
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
library(foreach) # 
library(doMC)    # doMC: shared-memory parallelism # in my experience handles memory better
library(snow)
library(pryr)    # For memory handling with FORK clusters http://gforge.se/2015/02/how-to-go-parallel-in-r-basics-tips/
library(magrittr)
library(caret)
library(randomForest)

#########################################################################################################################################################################################################
### SOURCE SUBFUNCTIONS ###
###########################

# Source subfunctions and train function for tunedRF
source("subfunctions_tunedRF.R")   # rfp() customRF {CARET} 
source("train_tunedRF.R")          # Hyperparameter tuning , Variable selection, Re-fitting 

# HINT: the 5x CV is wrapped into a function:
# TO RUN THE CODE USE: 
#my_func_RUN_RF_tuned_custom_caret_speedy_K.0_grid64(y = y, betas = betas10k, nfolds = nfolds, K.start = 1,
#                                                    p.n.pred.var = c(100, 200, 500, 1000, 2000, 5000, 7500, 10000),
#                                                    out.path = "./RF-Hypar-Tuned-Caret/RF-Tuned-Caret-customRF2-ALL-models-speedy-grid64-betas10k-p-8-100to10k-CVfold")

# Support functions
source("memorymgmt.r")
source("rfp.R")            
source("trainRF.R")
source("makefolds.R")
# Performance eval
source("brier.R")
source("mlogloss.R")

#########################################################################################################################################################################################################
### Load data ###
#################

load("MNPbetas10Kvar.RData")
load("nfolds.RData")

y <- anno$V5
dim(betas)
str(nfolds)

#########################################################################################################################################################################################################
### Parallel backend variations ###
###################################
library(doParallel)
cores <- detectCores()-1
registerDoMC(cores)
getDoParVersion()
getDoParWorkers()
getDoParRegistered()
Sys.time()
#########################################################################################################################################################################################################

#####################################
### Settings for nestedcv_tunedRF ###
#####################################
cores <- detectCores()-1 # Ubuntu =10 threads
seed <- 20170505
folds <- 5 
mc.cores <- cores
# ntrees <- 100
# p <- 100

# Trim down betas (10000 seems to be too large?)
dim(betas); class(betas)           # ==> df ==> glmnet need matrix as input
betas.mtx <- as.matrix(betas)
# Tests before betas is already variance sorted 1->10000 highest to lowest
betas100 <- betas.mtx[ , 1:100]
betas100 <- betas[ ,1:100]
betas1000 <- betas.mtx[ , 1:1000]   # THIS WAS USED OTHERWISE WOULD HAVE TAKEN TOO LONG



#########################################################################################################################################################################################################    


run_nestedcv_tunedRF <- function(y.., betas.., n.cv.folds = 5, nfolds..=NULL, # load("nfolds.RData")
                                 cores = 10, seed = 1234, K.start = 1, k.start = 0, out.path = "tRF/", out.fname = "CVfold", 
                                 mtry.min = NULL, mtry.max = NULL, length.mtry = 2,
                                 ntrees.min = 500, ntrees.max = 700, ntree.by = 100,
                                 nodesize.proc = c(0.01, 0.05, 0.1), 
                                 n.cv = 5, n.rep = 1,
                                 p.n.pred.var = c(50, 100)){
  
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
                                               # defaults used => # mtry.min = , mtry.max = , length.mtry = , ntrees.min = , ntrees.max = , ntree.by = ,
                                               # defaults used => # nodesize.proc = , n.cv = 5, n.rep = 1,
                                               p = p.n.pred.var, 
                                               p.tuning.brier = T, p.tuning.miscl.err = T, p.tuning.mlogloss = T, 
                                               seed = seed+1, allowParallel = T)  
      
      # NOTE: rfcv.tuned variable contains res <- list(rf.pred.best.brier, rf.pred.best.err, rf.pred.best.mlogl, imp.perm, rf.pred.l, score.pred.l, brier.p.l, err.p.l, mlogloss.p.l, t0, t1, t2)
      
      # Use tuned random forest modell fit rfcv.tuned[[1]] = rf.pred.best => to predict the corresponding CALIBRATION or TEST SETS (if innerfold then calibration set ; if outer fold then test set)
      message("\nFit tuned random forest on : test set ", K, ".", k, " ... ", Sys.time())
      
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
      
      # Calculate Error
      err.scores.rf.tuned.brier <- sum(colnames(scores.pred.rf.tuned.brier)[apply(scores.pred.rf.tuned.brier, 1, which.max)] != y..[fold$test]) / length(fold$test) # Calculates ME for BS tunedRF
      err.scores.rf.tuned.miscerr <- sum(colnames(scores.pred.rf.tuned.miscerr)[apply(scores.pred.rf.tuned.miscerr, 1, which.max)] != y..[fold$test]) / length(fold$test) # Calculates ME for ME tunedRF
      err.scores.rf.tuned.mlogl <- sum(colnames(scores.pred.rf.tuned.mlogl)[apply(scores.pred.rf.tuned.mlogl, 1, which.max)] != y..[fold$test]) / length(fold$test) # Calculates ME for mLL tunedRF
      
      # Print error rates
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
           file =  paste0(f.path, paste("CVfold", 1, 0, "RData", sep = ".")) #paste(out.fname, K, k, "RData", sep = ".")
           )
    }
  }
  message("Finished ... ", Sys.time())
}  

#########################################################################################################################################################################################################    
###########
### RUN ###
###########
Sys.time()
tryCatch(run_nestedcv_tunedRF(y.. = y, betas.. = betas, n.cv.folds = 5, #nfolds.. = nfolds,
                              cores = 11, seed = 1234, K.start = 1, k.start = 0, mtry.min = 50, mtry.max = 100, length.mtry = 2, p.n.pred.var = c(50, 100)))
message("Finished ... ", Sys.time()) 

 
