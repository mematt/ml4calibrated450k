########################################
### SUBFUNCTIONS for train_tunedRF.R ###
########################################

library(caret)
library(randomForest)

############################################
### 1. Parallelized randomForest wrapper ###
############################################

rfp <- function(xx, ..., ntree = ntree, mc = mc, seed = NULL) 
{
  if(!is.null(seed)) set.seed(seed, "L'Ecuyer") # help("mclapply") # reproducibility
  rfwrap <- function(ntree, xx, ...) randomForest::randomForest(x = xx, ntree = ntree, norm.votes = FALSE, ...)
  rfpar <- mclapply(rep(ceiling(ntree / mc), mc), mc.cores = mc, rfwrap, xx = xx, ...) 
  do.call(randomForest::combine, rfpar) # > ?randomForest::combine => combines two or more trees/ensembles of trees (rF objects) into one # get the subset of trees from each core and patch them back together
}

###########################################################################################
### 2. Custom random forest function within the caret package to allow for ntree tuning ###
###########################################################################################

# Custom function within the caret package to tune ntree, mtry and node.size paramters of randomForest() 
# This function uses the parllelized/wrapped rfp() function from above

# CustomRF for caret
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)

customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), 
                                  class = rep("numeric", 3), 
                                  label = c("mtry", "ntree", "nodesize"))

customRF$grid <- function(x, y, len = NULL, search = "grid"){}

customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...){
  rfp(xx = x, y = y, 
      mc = 4L,             # set multi-core fixed to 4 # can be changed according to hardware
      mtry = param$mtry,  # tuneable parameter
      ntree=param$ntree,  # tuneable parameter
      nodesize = param$nodesize, #tuneable parameter
      strata = y, 
      sampsize = rep(min(table(y)), length(table(y))), # downsampling to the minority class 
      importance = TRUE, 
      replace = FALSE, 
      ...)
}

customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL) predict(modelFit, newdata)

customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL) predict(modelFit, newdata, type = "prob")

customRF$sort <- function(x) x[order(x[, 1]), ]

customRF$levels <- function(x) x$classes


###################################################
###       FINE TUNE RF hyperparameters          ###
###       using customRF() and caret            ###
###                                             ###
###################################################

# This function is used within the train_tunedRF.R script and the trainRF_caret_custom_tuner() function

subfunc_rf_caret_tuner_customRF <- function(y.subfunc, betas.subfunc, 
                                            mtry.min = NULL, mtry.max = NULL, length.mtry = 6, 
                                            ntrees.min = 1000, ntrees.max = 2000, ntree.by = 500,
                                            nodesize.proc = c(0.01, 0.05, 0.1),
                                            n.cv = 5, n.rep = 1,
                                            seed, allowParall = T,
                                            print.res = T){
  # Train control
  control <- trainControl(method = "repeatedcv", number = n.cv, repeats = n.rep, classProbs = F, summaryFunction = multiClassSummary, allowParallel = allowParall)
  
  # Tune grid
  if(is.null(mtry.min)){mtry.min <- floor(sqrt(ncol(betas.subfunc)))*0.5}
  if(is.null(mtry.max)){mtry.max <- floor(sqrt(ncol(betas.subfunc)))} # defaults to sqrt(10000) = 100 
  
  tunegrid <- expand.grid(.mtry = floor(seq(mtry.min, mtry.max, length.out = length.mtry)),
                          .ntree = seq(ntrees.min, ntrees.max, ntree.by), 
                          .nodesize = floor(c(1, 5, nrow(betas.subfunc)*nodesize.proc[[1]], nrow(betas.subfunc)*nodesize.proc[[2]], nrow(betas.subfunc)*nodesize.proc[[3]]))
  )
  # Train - 5x CV (n.cv) for tuning parameters with single repeat (n.rep)
  set.seed(seed, kind = "L'Ecuyer-CMRG")
  rf.tuned.caret.customRF <- train(x = betas.subfunc, y = y.subfunc, method = customRF, metric = "Accuracy", tuneGrid = tunegrid, trControl = control)
  if(print.res){print(rf.tuned.caret.customRF)}
  return(rf.tuned.caret.customRF)
}

# Get out best tuned models
#rf.tuned.caret.customRF$bestTune
#rf.tuned.caret.customRF$results

# Sys.time() # [1] "2019-01-29 10:08:48 CET"
# subfunc.tRF.testrun <- subfunc_rf_caret_tuner_customRF(y = y[nfolds[[1]][[1]][[1]]$train], betas = betas100[nfolds[[1]][[1]][[1]]$train, ], seed = 1234, ntrees.min = 250, ntrees.max = 500, ntree.by = 250, length.mtry = 2, nodesize.proc = c(0.1, 0.05, 0.1))
# Sys.time() # [1] "2019-01-29 10:10:13 CET"

# Sys.time()
# subfunc.tRF.testrun <- subfunc_rf_caret_tuner_customRF(y = y[nfolds[[1]][[1]][[1]]$train], betas = betas100[nfolds[[1]][[1]][[1]]$train, ], seed = 1234, ntrees.min = 250, ntrees.max = 500, ntree.by = 250, length.mtry = 2, nodesize.proc = c(0.1))
# Sys.time()
#
# "2019-01-29 09:32:17 CET"
# subfunc.tRF.testrun <- subfunc_rf_caret_tuner_customRF(y = y[nfolds[[1]][[1]][[1]]$train], betas = betas100[nfolds[[1]][[1]][[1]]$train, ], seed = 1234, ntrees.min = 250, ntrees.max = 500, ntree.by = 250, length.mtry = 2, nodesize.proc = c(0.1))
# 2204 samples
# 100 predictors
# 91 classes: 'ETMR', 'MB, G3', 'MB, G4', 'MB, WNT', 'MB, SHH CHL AD', 'MB, SHH INF', 'ATRT, MYC', 'ATRT, SHH', 'ATRT, TYR', 'CNS NB, FOXR2', 'HGNET, BCOR', 'DMG, K27', 'GBM, G34', 'GBM, MES', 'GBM, RTK I', 'GBM, RTK II', 'GBM, RTK III', 'GBM, MID', 'GBM, MYCN', 'CN', 'DLGNT', 'LIPN', 'LGG, DIG/DIA', 'LGG, DNT', 'LGG, RGNT', 'RETB', 'ENB, A', 'ENB, B', 'PGG, nC', 'LGG, GG', 'SCHW', 'SCHW, MEL', 'CPH, ADM', 'CPH, PAP', 'PITAD, ACTH', 'PITAD, FSH LH', 'PITAD, PRL', 'PITAD, STH SPA', 'PITAD, TSH', 'PITAD, STH DNS A', 'PITAD, STH DNS B', 'PITUI', 'EPN, RELA', 'EPN, PF A', 'EPN, PF B', 'EPN, SPINE', 'EPN, YAP', 'EPN, MPE', 'SUBEPN, PF', 'SUBEPN, SPINE', 'SUBEPN, ST', 'CHGL', 'LGG, SEGA', 'LGG, PA PF', 'LGG, PA MID', 'ANA PA', 'HGNET, MN1', 'IHG', 'LGG, MYB', 'LGG, PA/GG ST', 'PXA', 'PTPR, A', 'PTPR, B', 'PIN T,  PB A', 'PIN T,  PB B', 'PIN T, PPT', 'CHORDM', 'EWS', 'HMB', 'MNG', 'SFT HMPC', 'EFT, CIC', 'MELAN', 'MELCYT', 'PLEX, AD', 'PLEX, PED A ', 'PLEX, PED B', 'A IDH', 'A IDH, HG', 'O IDH', 'LYMPHO', 'PLASMA', 'CONTR, ADENOPIT', 'CONTR, CEBM', 'CONTR, HEMI', 'CONTR, HYPTHAL', 'CONTR, INFLAM', 'CONTR, PINEAL', 'CONTR, PONS', 'CONTR, REACT', 'CONTR, WM' 
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold, repeated 1 times) 
# Summary of sample sizes: 1760, 1763, 1766, 1762, 1765 
# Resampling results across tuning parameters:
#   
#   mtry  ntree  nodesize  Accuracy   Kappa      Mean_F1    Mean_Sensitivity  Mean_Specificity
# 5    250      1       0.9051884  0.9032212        NaN  0.9212518         0.9989353       
# 5    250      5       0.8906355  0.8883641        NaN  0.8952812         0.9987745       
# 5    250    220       0.1751679  0.1583694        NaN  0.1522716         0.9907589       
# 5    500      1       0.9010819  0.8990304        NaN  0.9175059         0.9988892       
# 5    500      5       0.8987999  0.8967045        NaN  0.9104265         0.9988646       
# 5    500    220       0.2276271  0.2110224        NaN  0.1939153         0.9913438       
# 10    250      1       0.8997327  0.8976530        NaN  0.9056940         0.9988761       
# 10    250      5       0.8965241  0.8943837        NaN  0.9040576         0.9988405       
# 10    250    220       0.1475070  0.1310423        NaN  0.1305773         0.9904575       
# 10    500      1       0.9087786  0.9068849  0.9139349  0.9208708         0.9989763       
# 10    500      5       0.8974311  0.8953028        NaN  0.9061011         0.9988495       
# 10    500    220       0.1782796  0.1617238        NaN  0.1527953         0.9907964       
# Mean_Pos_Pred_Value  Mean_Neg_Pred_Value  Mean_Precision  Mean_Recall  Mean_Detection_Rate
# 0.9195190            0.9989359            0.9195190       0.9212518    0.009947125        
# 0.9050969            0.9987760            0.9050969       0.8952812    0.009787203        
# NaN            0.9913054                  NaN       0.1522716    0.001924922        
# 0.9160552            0.9988908            0.9160552       0.9175059    0.009901999        
# 0.9212325            0.9988652            0.9212325       0.9104265    0.009876922        
# NaN            0.9920180                  NaN       0.1939153    0.002501396        
# 0.9193061            0.9988769            0.9193061       0.9056940    0.009887173        
# 0.9122122            0.9988412            0.9122122       0.9040576    0.009851914        
# NaN            0.9909198                  NaN       0.1305773    0.001620956        
# 0.9285534            0.9989772            0.9285534       0.9208708    0.009986578        
# 0.9217303            0.9988507            0.9217303       0.9061011    0.009861881        
# NaN            0.9913428                  NaN       0.1527953    0.001959116        
# Mean_Balanced_Accuracy
# 0.9600935             
# 0.9470278             
# 0.5715153             
# 0.9581975             
# 0.9546456             
# 0.5926296             
# 0.9522851             
# 0.9514490             
# 0.5605174             
# 0.9599236             
# 0.9524753             
# 0.5717959             
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were mtry = 10, ntree = 500 and nodesize = 1.
# 
# Warning message:
# In nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,  :
# There were missing values in resampled performance measures.
# Sys.time()
# [1] "2019-01-29 09:33:42 CET"
