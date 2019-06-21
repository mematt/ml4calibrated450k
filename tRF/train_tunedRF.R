#--------------------------------------------------------------------
# ml4calibrated450k - tuned RF (tRF) - Train function 
#
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-24 
#--------------------------------------------------------------------


### Training & tuning function - integrating the utility/subfuctions from `subfunctions_tunedRF.R`

trainRF_caret_custom_tuner <- function(y., 
                                       betas.,
                                       cores = 4L, # some variation of parallel backend needs to be defined
                                       mtry.min = NULL, mtry.max = NULL, length.mtry = 6,
                                       ntrees.min = 1000, ntrees.max = 2000, ntree.by = 500,
                                       use.default.classif.nodesize.1 = T,
                                       nodesize.proc = c(0.01, 0.05, 0.1),
                                       n.cv = 5, n.rep = 1,
                                       p = c(100, 500, 1000, 10000), # number of values in p should not be more than the number of available cores/threads in the system
                                       p.tuning.brier = T, p.tuning.miscl.err = T, p.tuning.mlogloss = T,
                                       seed, 
                                       allowParallel = T){
  
  # Info block about settings
  set.seed(seed, kind = "L'Ecuyer-CMRG")
  #message("Start @ ", Sys.time())
  message("seed: ", seed)
  message("cores: ", cores)
  
  # Calcualte tune tunegrid size
  if(use.default.classif.nodesize.1) {
    n.nodesize <- 1
    tunegrid.size <- length.mtry*length(seq(ntrees.min, ntrees.max, by = ntree.by))
  } else { 
    n.nodesize <- length(c(1, 5, nodesize.proc))
    tunegrid.size <- length.mtry*length(seq(ntrees.min, ntrees.max, by = ntree.by))*length(c(1, 5, nodesize.proc))
  }
  
  message("\nStep 1.: Random Forest - Hyperparameter tuning (extra nested loop) with the following number of parameters : ", 
          "\nlength.mtry = ", length.mtry, 
          "; \nntree = ", paste(seq(ntrees.min, ntrees.max, by = ntree.by), sep = ";", collapse = ", "), 
          "; \nlength.nodesize = ", n.nodesize,  # default classification =1, def. regression = 5, Kruppa et al. = 10%
          "; \nTotal GRID size: ", tunegrid.size, 
          "; \nUsing custom function in {caret} with (extra nested) repeated CV (n.cv = ", n.cv, "; n.rep = ", n.rep, ") @ ", Sys.time())
  
  # Run 1 - Tune RF hyperparameters using grid search (tuning measure = "Accuracy")
  t0 <- system.time(
    # caret() gives warnings for empty measures => suppress
    suppressWarnings( 
      # Uses the customRF() function using grid search of the caret training/tuning framework
      rf.hypar.tuner <- subfunc_rf_caret_tuner_customRF(y.subfunc = y., 
                                                        betas.subfunc = betas., 
                                                        mtry.min = mtry.min, mtry.max = mtry.max, length.mtry = length.mtry, 
                                                        ntrees.min = ntrees.min, ntrees.max = ntrees.max, ntree.by = ntree.by,
                                                        use.default.nodesize.1 = use.default.classif.nodesize.1, # defaults to TRUE
                                                        nodesize.proc = nodesize.proc, # if use.default.nodesize = T => this is ignored
                                                        n.cv = n.cv, n.rep = n.rep,
                                                        seed = seed, 
                                                        allowParall = allowParallel,
                                                        print.res = T)
      
    )  
  )
  # Output hyperparameter results in the console
  message("Hyperparameter Tuning Results:", 
          "\n bestTune mtry: ", rf.hypar.tuner$bestTune$mtry, 
          "\n bestTune ntree: ", rf.hypar.tuner$bestTune$ntree, 
          "\n bestTune nodesize: ", rf.hypar.tuner$bestTune$nodesize)
  message("Run time (s) of hyperparameter tuning: ")
  print(t0) 
  
  
  # Run 2 - Fit RF with best tuning parameters for variable selection
  message("\nStep. 2.: Variable selection using random forest with tuned hyperparameters ... ", Sys.time())
  message("Multi-core using `rfp()` wrapper (assigned nr. of threads): ", cores)
  t1 <- system.time(
    rf.varsel <- rfp(xx = betas., 
                     y = y.,
                     mc = cores,
                     mtry = rf.hypar.tuner$bestTune$mtry,
                     ntree = rf.hypar.tuner$bestTune$ntree,
                     nodesize = rf.hypar.tuner$bestTune$nodesize,
                     strata = y.,
                     sampsize = rep(min(table(y.)),length(table(y.))), # downsampling to the minority class # smallest class in betas = 8 cases | 4-6 cases
                     importance = TRUE,                              # permutation based mean deacrease in accuracy 
                     replace = FALSE)                                # without replacement see Strobl et al. BMC Bioinf 2007
  )                                                                  # http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25
  message("Run time (s) of variable selection: ")
  print(t1) 
  
  # Extract permutation based importance measure
  message("\nStep. 3.: Variable importance and subsetting of methylation data (betas) using p = [", paste(p, collapse = "; "), "] ... ", Sys.time())
  imp.perm <- importance(rf.varsel, type = 1)
  
  # Variable selection ("p" most important variables are selected)
  p.l <- as.list(p)
  message("Reordering betas using multi-core: ", length(p.l))
  or <- order(imp.perm, decreasing = T)
  # Reorder - using mutli-core
  betasy.l <- mclapply(seq_along(p.l), function(i){
    p.l.i <- p.l[[i]]
    betas.[ , or[1 : p.l.i]]
  }, mc.preschedule = T, mc.set.seed = T, mc.cores = length(p.l))
  
  message("\nStep 4.: Refit the subsetted [train set] using only the 'p' most important variables (CpGs) in parallel ... ", Sys.time())
  message("p: ", paste(p, collapse = "; "))  
  message("Multi-core: ", length(betasy.l))
  set.seed(seed+1, kind = "L'Ecuyer-CMRG")
  t2 <- system.time(
    rf.pred.l <- mclapply(seq_along(betasy.l), function(i){
      rfp(x = betasy.l[[i]], 
          y = y.,
          mc = 4L, # can be tweaked depending on the CPU core count
          mtry = rf.hypar.tuner$bestTune$mtry,
          ntree = rf.hypar.tuner$bestTune$ntree,
          nodesize = rf.hypar.tuner$bestTune$nodesize,
          strata = y.,
          sampsize = rep(min(table(y.)), length(table(y.))), # downsampling to the minority class # smallest class in betas = 8 cases | 4-6 cases
          proximity = TRUE,
          oob.prox = TRUE,
          importance = TRUE, # permutation based mean deacrease in accuracy 
          keep.inbag = TRUE,
          do.trace = FALSE,
          seed = seed)
    }, mc.preschedule = T, mc.set.seed = T, mc.cores = length(betasy.l)) # number of cores/threads == number of p variables
  )
  # str(rf.pred.l) # list of 2 # rf objects
  message("Run time (s) of refitting/training the RF classifier: ")
  print(t2) 
  
  # Refit rf.pred.l models on the training data
  message("\nStep 5.: Predict the 'p' subsetted training set to obtain scores in parallel [", length(rf.pred.l), " cores] ... ", Sys.time())
  score.pred.l <- mclapply(seq_along(rf.pred.l), function(i){
    predict(object = rf.pred.l[[i]], 
            newdata = betas.[ , rownames(rf.pred.l[[i]]$importance)],
            type = "prob")
  }, mc.preschedule = T, mc.set.seed = T, mc.cores = length(rf.pred.l))
  
  message("\nStep 6.: Calculate evaluation metrics for each of the 'p' subsetted training sets (single core) ... ", Sys.time())
  # Performance Evaluators for "p" number of predictors tuning
  message("Corresponding p = [ ", paste(p, collapse = "; "), " ].")
  # Brier score (BS)
  if(p.tuning.brier){
    brier.p.l <- lapply(seq_along(score.pred.l), function(i){
      brier(scores = score.pred.l[[i]], y = y.)
    })
    message("Brier scores (BS): ", paste(brier.p.l, collapse = "; "))
  } 
  
  # Misclassification Error (ME) 
  if(p.tuning.miscl.err){
    err.p.l <- lapply(seq_along(score.pred.l), function(i){
      sum(colnames(score.pred.l[[i]])[apply(score.pred.l[[i]], 1, which.max)] != y.) / length(y.)
    })
    message("Misclassification errors (ME): ", paste(err.p.l, collapse = "; "))
  } 
  
  # Multiclass logloss (LL)
  if(p.tuning.mlogloss){
    mlogloss.p.l <- lapply(seq_along(score.pred.l), function(i){
      mlogloss(scores = score.pred.l[[i]], y = y.)
    })
    message("Multiclass logloss (LL): ", paste(mlogloss.p.l, collapse = "; "))
  } 
  
  # Assign optimal "p" number of features to new variables
  id.min.BS <- which.min(brier.p.l)
  id.min.ME <- which.min(err.p.l)
  id.min.LL <- which.min(mlogloss.p.l)
  
  # Output optimal "p" number of features (CpG probes) respective to Brier score, log loss and miscl. error
  message("\nOptimal number of predictor variables:  p(brier) = ", p[[id.min.BS]])
  message("Optimal number of predictor variables:  p(mlogl) = ", p[[id.min.LL]])
  message("Optimal number of predictor variables:  p(miscl.err) = ", p[[id.min.ME]])
  
  # Best RF model object  
  message("\nAll p(brier), p(mlogl) and p(miscl.err) models are going to be written into the return object ... @ ", Sys.time())
  rf.pred.best.brier <- rf.pred.l[[id.min.BS]]
  rf.pred.best.err <- rf.pred.l[[id.min.ME]]
  rf.pred.best.mlogl <- rf.pred.l[[id.min.LL]]
  
  # Results - list of 12 elements
  res <- list(rf.pred.best.brier, rf.pred.best.err, rf.pred.best.mlogl, 
              imp.perm, rf.pred.l,
              score.pred.l, brier.p.l, err.p.l, mlogloss.p.l,
              t0, t1, t2)
  return(res)
}