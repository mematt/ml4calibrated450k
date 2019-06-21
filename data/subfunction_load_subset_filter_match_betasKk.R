#--------------------------------------------------------------------
# Loading and variance filtering of 450k betas (betas_v11.h5)   
#
#
# Matt Maros
# maros@uni-heidelberg.de
#
# 2019-04-16-19 UTC
#--------------------------------------------------------------------


# Check, install | load recquired packages ---------------------------------------------------------------------------------------------------------------------------

if(!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", dependencies = T)
  library(BiocManager)
} else {library(BiocManager)}

if(!requireNamespace("rhdf5", quietly = TRUE)) {
  BiocManager::install("rhdf5", version = "3.8")
  library(rhdf5)
} else {library(rhdf5)}

# Filtering of `betas_v11.h5` (9.5-10.2 Gb) data object with 485 000 CpG x 2801 patients --------------------------------------------------------------------------------------------

# >>> This script is moved to the repository `ml4calibrated450k` on GitHub @ <https://github.com/mematt/ml4calibrated450k>

#### Sunfunction `subfunc_load_betashdf5_subset_filter_match_save_betasKk()` -------------------------------------------------------------------------------------------------------

# 1. Load betas_v11.h5 (betashdf5) 
# 2. Subset K.k..train 
# 3. Unsupervised variance filtering (p = 10 000) 
# 4. Subset K.k.test 
# 5. Match CpG probes of filtered K.k.train to K.k.test 
# 6. Save also the full betas (2801 * 10000) but 10k CpG (are based on the respective K.k train set) 
# 7. Save betas.p.filtered.K.k into a separate folder

# Define utility / loader function ----------------------------------------------------------------------------------------------------------------

utilityfunc_read_hd5_betas <- function(sentrix, probenames, db = "./data/betas_v11.h5"){
  betas <- simplify2array(lapply(sentrix, function(i){
    h5read(db, i)  
  }))
  H5close() 
  colnames(betas) <- sentrix
  rownames(betas) <- probenames
  return(betas)
}


# Define the subfunction ----------------------------------------------------------------------------------------------------------------

# Load & filter the training (sub)fold and subset the test (sub)fold accorinding to the most variable CpG probes in the corresponding training fold
# <Please note>: this function uses a fixed default loading path ("./data/") to fetch `nfolds.RData`, `anno.RData`, and `betas_v11.h5` files. 

subfunc_load_betashdf5_subset_filter_match_save_betasKk <- function(K.start = 1, k.start = 0, n.cv.folds = 5, 
                                                                    nfolds.. = NULL,
                                                                    fpath.betasv11.hdf5 = NULL,
                                                                    p.var.filtering = 10000,
                                                                    out.path = "betas.varfilt.10k", out.fname = "betas.K.k"){
  
  # Check whether nfolds.. is provided
  if(is.null(nfolds..) & exists("nfolds")) {
    nfolds.. <- get("nfolds", envir = .GlobalEnv)
    message("The `nfolds` object for nested CV scheme assignments was fetched from .GlobalEnv @ ", Sys.time())
    # if(length(nfolds..) == n.cv.folds){
    #   message("nfolds.. = objet has the same length as n.cv.folds: OK")
    # } else {
    #    message("Please note that `nfolds..` structure for nested resampling and `n.cv.folds` arguements",
    #            "do not match thus the nested CV will be only partially performed.")
    # }
  } else {
    stop("Please provide a fold structure for nested resampling. For instance, load the nfolds.RData")
  }
  
  # Check whether betas_v11.h5 is already in the .GlobalEnv
  if(is.null(fpath.betasv11.hdf5) & exists("betasv11.hdf5")) {
    betasv11.h5 <- get("betasv11.hdf5", envir = .GlobalEnv)
    message("Dimensions of `betasv11.hdf5` rows: ", dim(betasv11.h5)[1], "cols: ", dim(betasv11.h5)[2])
  } else {
    message("The `betas_v11.h5` file (9.5Gb) will be loaded from default path: ./data/betas_v11.h5 ;", 
            " along with `anno.RData` and `probenames.RData` .", "\n It can take up to 3-5 mins to finish.")
    fpath.betasv11.hdf5 <- file.path("./data/betas_v11.h5")
    fpath.anno <- file.path("./data/anno.RData")
    fpath.probenames <- file.path("./data/probenames.RData")
    # Load
    message(" > Load `anno.RData` from default path `", fpath.anno, "` into function/.GlonbalEnv.")
    load(fpath.anno)
    message(" > Load `probenames.RData` from default path `", fpath.probenames, "` into function/.GlonbalEnv.")
    load(fpath.probenames)
    message(" > Load `betas_v11.h5` (9.5Gb) of normalized non-batch-adjusted data of the reference cohort", 
            "(Capper et al. 2018, Nature) from default path `", 
            fpath.betasv11.hdf5, "` into function/.GlonbalEnv.")
    # Utility function => loading time: ca. 3.5-5 min
    betasv11.h5 <- utilityfunc_read_hd5_betas(sentrix = anno$sentrix[1:length(anno$sentrix)], 
                                              probenames = probenames, db = fpath.betasv11.hdf5) 
    # rows: 458 000 CpG probes # cols: 2801 patients
    message("Dimensions of loaded `betasv11.h5` data file nrows: ", nrow(betasv11.h5), "; ncols: ", ncol(betasv11.h5))
  }
  
  # Run CV scheme
  message("\nNested cross validation (CV) scheme starts ... ", Sys.time())
  for(K in K.start:n.cv.folds){  
    # Nested loop
    for(k in k.start:n.cv.folds){ 
      
      if(k > 0){ message("\n Subsetting & filtering inner/nested fold ", K,".", k,"  ... ",Sys.time())  
        fold <- nfolds..[[K]][[2]][[k]]  ### [[]][[2]][[]] means inner loop # Inner CV loops 1.1-1.5 (Fig. 1.)
      } else{                                                                          
        message("\n \nSubsetting & filtering outer fold ", K,".0  ... ",Sys.time()) 
        fold <- nfolds..[[K]][[1]][[1]]   ### [[]][[1]][[]] means outer loop # Outer CV loops 1.0-5.0 (Fig. 1.)
      }
      
      # Subset K.k$train
      message(" Step 2. Subsetting cases/columns: " , K, ".", k, " training set @ ", Sys.time())
      betas.K.k.train <- betasv11.h5[ , fold$train] # rows CpG # columns are patients! # see line 408
      
      message(" Step 3. Unsupervised variance filtering of p = ", p.var.filtering,  
              " CpG probes on " , K, ".", k, " training set @ ", Sys.time(),
              "\n  It can take up to 1-2mins to finish.")
      # sd is calculated over all cols (i.e. patients) for each row (i.e. CpG probe) 
      betas.p.filtered.K.k.train <- betas.K.k.train[order(apply(betas.K.k.train, 1, sd), decreasing = T)[1:10000], ] 
      message("  Dimension of `betas.p.filtered.K.k.train` nrows: ", 
              nrow(betas.p.filtered.K.k.train), " ncols: ", ncol(betas.p.filtered.K.k.train),
              "\n  Variance filtering finished @ ", Sys.time()) # Duration @ single core ca. 1.25-1.5mins
      message(" \n  Check whether there is NA in train set : ", sum(is.na(betas.p.filtered.K.k.train) == T))
      
      # Transposed afterwards!
      betas.p.filtered.K.k.train <- t(betas.p.filtered.K.k.train)
      # betas.p.filtered.K.k.train # matrix # size 214.6 Mb
      # fold$train (ca. 1700-2204) rows cases/patients (sentrixIDs) 
      # rows: patients; cols 10k most variable CpGs
      message("  Transposing `betas.p.filtered.K.k.train` finished @ ", Sys.time())

      # Garbage collector (note: gc is not absolutely necessary)
      message("  Clean up memory (garbage collector) @ ", Sys.time())
      gc()
      
      # Subset CpG of the corresping test set 
      # Select only 10 000 CpG (`p.var.filtering`) probes (i.e. rows of betasv11.h5) that are filtered based on
      # the training (sub)fold (i.e. columns of betas.p.varfilt.train)
      message(" Step 4. Subsetting `betas_v11.h5` cases/columns: " , K, ".", k, " test/calibration set @ ", Sys.time())
      betas.K.k.test <- betasv11.h5[ , fold$test]
      
      message(" Step 5. Matching variance filtered p = ", p.var.filtering,  
              " CpG probes corresponding to the " , K, ".", k, " training set @ ", Sys.time(),
              "\n  It can take up to 1-2mins to finish.")
      betas.p.filtered.K.k.test <- betas.K.k.test[match(colnames(betas.p.filtered.K.k.train), rownames(betas.K.k.test)), ]
      # Transpose $test
      # Note: wrappend in t() => rows: patients ; cols: CpG probenames
      betas.p.filtered.K.k.test <- t(betas.p.filtered.K.k.test)
      message("  Transposing `betas.p.filtered.K.k.test` finished @ ", Sys.time())
      message("  Dimension of `betas.p.filtered.K.k.test`  nrows: ", 
              nrow(betas.p.filtered.K.k.test), " ncols: ", ncol(betas.p.filtered.K.k.test),
              "\n CpG matching finished @ ", Sys.time())
      
      # Save also betas.K.k (2801 * 10k CpG selected on the training set)
      message(" Step 6. Matching variance filtered p = ", p.var.filtering, 
              " CpG probes corresponding to the " , K, ".", k, " training set @ ", Sys.time(),
              "\n  On the full `betas_v11.h5` data. It can take up to 1-2mins to finish.")
      betas.K.k <- betasv11.h5[match(colnames(betas.p.filtered.K.k.train), rownames(betasv11.h5)), ] 
      # rows = CpGs # columns = patients 2801 # no subsetting
      message("  Transposing `betas.K.k` finished @ ", Sys.time())
      betas.K.k <- t(betas.K.k) # rows patients # cols CpGs
      
      # Security check
      message("\nAre column names (CpG probes) of $train and $test and full betas identical? ", 
              identical(colnames(betas.p.filtered.K.k.train), colnames(betas.p.filtered.K.k.test)))
      message("Are column names (CpG probes) of $train and full `betas.K.k`` identical? ", 
              identical(colnames(betas.p.filtered.K.k.train), colnames(betas.K.k)))
      message("Are column names (CpG probes) of $test and full `betas.K.k`` identical? ", 
              identical(colnames(betas.p.filtered.K.k.train), colnames(betas.K.k)))
      
      # Create output directory  
      folder.path <- file.path(getwd(), "data", out.path)
      dir.create(folder.path, showWarnings = F, recursive = T)
      #RData.path <- file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      
      # Save unsupervised variance filtered $train and $test sets
      save(betas.p.filtered.K.k.train, 
           betas.p.filtered.K.k.test,
           betas.K.k,
           fold,
           file = file.path(folder.path, paste(out.fname, K, k, "RData", sep = "."))
      )  
      
    }
  }
}

## Run/Function call ---------------------------------------------------------------------------------------------------------------------------------------------
# # Runtime ~ 45 mins @ single core/thread on a laptop (MBP i9)
# Sys.time() # 19:45:55
# subfunc_load_betashdf5_subset_filter_match_save_betasKk()
# Sys.time() # 20:29:22
