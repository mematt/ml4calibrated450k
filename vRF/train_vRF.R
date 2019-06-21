#--------------------------------------------------------------------
# ml4calibrated450k - vanilla RF (vRF) - Train function  
#
#
# Martin Sill & Matt Maros
# m.sill@dkfz.de & maros@uni-heidelberg.de
#
# 2019-04-18 
#--------------------------------------------------------------------

### Training & tuning function - integrating the utility/subfuctions from `subfunctions_vRF.R`

trainRF <- function(y, betas, ntrees, p, seed, cores){
  
  # train RF for variable selection
  set.seed(seed, kind = "L'Ecuyer-CMRG") 
  message("seed: ", seed)
  message("cores: ", cores)
  message("ntrees: ", ntrees)  
  message("n_cases: ", nrow(betas))
  message("n_CpGs: ", ncol(betas))  
  
  message("\nVariable (CpG probe) selection ...", Sys.time())
  t1 <- system.time(
    rf.varsel <- rfp(xx = betas,
                     y,
                     mc = cores,
                     ntree = ntrees,
                     strata = y,
                     sampsize = rep(min(table(y)),length(table(y))), # downsampling to the minority class
                     importance = TRUE, # permutation based importance
                     replace = FALSE) # without replacement see Strobl et al. BMC Bioinf 2007 
                                      # http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25
  )
  
  # extract permutation based importance measure
  imp.perm <- importance(rf.varsel, type = 1)
  
  # variable selection
  or <- order(imp.perm, decreasing = T)
  betasy <- betas[ , or[1:p]]   # CAVE: p (argument in trainRF) => limits the number of most important variable (importance.perm)
  # betasy = only 100! (p = 100 in MNPrandomForest.R)
  
  message("\nTraining classifier ...", Sys.time())
  message("single core")
  message("ntrees: ", ntrees)  
  message("n: ", nrow(betasy))
  message("p: ", ncol(betasy))  # CAVE: p (argument in trainRF) => limits the number of most important variable (importance.perm)
  
  t2 <- system.time(
    rf.pred <- randomForest(betasy,
                            y,
                            ntree = ntrees,
                            strata = y,
                            sampsize = rep(min(table(y)), length(table(y))),
                            proximity = TRUE,
                            oob.prox = TRUE,
                            importance = TRUE,
                            keep.inbag = TRUE,
                            do.trace = FALSE,
                            seed = seed)
  )
  
  res <- list(rf.pred, imp.perm, t1, t2)
  
  return(res)
}