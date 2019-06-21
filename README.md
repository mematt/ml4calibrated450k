---
output:
  html_document: default
  pdf_document: default
---
# ml4calibrated450k

## Overview 
This is a companion repository for the article *"Comparative analysis of machine learning workflows to estimate class probabilities for precision cancer diagnostics on DNA methylation microarray data"* submitted to Nature Protocols (https://www.nature.com/nprot/).

Our comaprisons included four well-established machine learning (ML) algorithms: random forests (RF), elastic net penalized multinomial logistic regression (ELNET), support vector machines (SVM) and boosted trees (XGBOOST).

For calibration, we used i) Platt scaling implemented by logistic regression (LR), Firth's penalized LR; and ii) ridge penalized multinomial regression (MR). 

All algorithms were compared on an uqinque data set of brain tumor DNA methylation reference cohort (n=2801 cases belonging to 91 classes) published in:

> Capper, D., Jones, D. T. W., Sill, M. and et al. (2018a). 
*"DNA methylation-based classification of central nervous system tumours." Nature, 555, 469 ;* 
https://www.nature.com/articles/nature26000. 

The corresponding Github repository (https://github.com/mwsill/mnp_training) presents the implementations of the MR-calibrated (untuned) RF classifier and all steps (i.e. downloading, pre-processing and filtering) required to generate the benchmarking data set `MNPbetas10Kvar.RData` (see *Fig. 1 - Part 1* in the submitted paper). 

The 450k DNA methylation array data of the reference cohort is available in the Gene Expression Omnibus under the accession number GSE109381 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109381).

The benchmarking data set was based on the 10,000 most variable CpG probes and it can be easily generated using R scripts provided in the above repository (https://github.com/mwsill/mnp_training).   

A smaller subset of the reference DNA methylation cohort data containing only the 1000 most variable CpG probes (`betas1000.RData`) is provided for direct download in this repository. The true class label vector `y.RData` is also directly downloadable from here. 

This repository focuses on the internal validation and benchmarking of the combination of these ML- and calibration algorithms (see *Figure 1 - Part 2* in the submitted paper) to develop ML-workflows for estimating class probabilities for precision cancer diagnostics.

***

## Repo content

### Data generation & resampling scheme
All algorithms were implementated and evaluated within:  

+ 5 x 5-fold nested cross-validation (CV) scheme using stratified sampling  
  + **R package**: base R  
  + **R script**: `makefolds.R` (generates the list object `nfolds`, which is also available as `nfolds.RData`)  
  + **R script**: `subfunction_load_subset_filter_match_betasKk.R` contains functions (`utilityfunc_read_hd5_betas()` and `subfunc_load_betashdf5_subset_filter_match_save_betasKk()`) to perform variance filtering based on `nfolds` resampling structure.    
  + **RData files**: `anno.RData`, `nfolds.RData`, `probenames.RData`, `y.RData`.  

### Machine learning (ML) classifiers:
+ Random Forests (RF) 
  + vanilla RF (using default settings; vRF)
  + tuned RF (tRF)
     + Brier score (BS)
     + Misclassification error (ME)
     + Multiclass log loss (LL)
  + **R package(s)**: `randomForest`, `caret`
+ Elastic net penalized multinomial logistic regression (ELNET) 
  + concurrent tuning of alpha and lambda 
  + **R package(s)**: `glmnet`
+ Support vector machines (SVM)
  + Radial Basis Function kernels (RBF)
  + Linear kernels (LK)
  + **R package(s)**: 
    + CPU: `e1071`, `ksvm` (`caret`), `LiblineaR`; 
    + GPU (NVIDIA CUDA-accelerated) `Rgtsvm`
+ Gradient boosted decision trees (XGBOOST)
  + comperehensive tuning of multiple tuning parameters
  + **R package(s)**: `xgboost`, `caret`
  
### Calibration/Post-processing algortihms:
+ Platt scaling 
  + Logistic Regression (LR)
      + **R package**: `glm` (base R function)
  + Firth's penalized LR (FLR) 
      + **R package**: `brglm` 
+ Ridge penalized multinomial logistic regression (MR)  
  +   + **R package**: `glmnet`
      
### Performance evaluation: 
We also provide scripts for evaluation (e.g. `evaluation_metrics.R` and `performance_evaluator.R`) such as:  

+ Misclassification error (ME)
+ Multiclass AUCH as published by Hand and Till (2001)
    + **R package**: `HandTill2001`
+ Brier score (BS) 
    + **R script**: `brier.R`
+ Mutliclass log loss (LL) 
    + using the Kaggle formulation https://web.archive.org/web/20160316134526/https://www.kaggle.com/wiki/MultiClassLogLoss.
    + **R script**: `mlogloss.R`

## Structure of .R scripts

Because the respective R package for each investigated ML-classifier algorithm has different built-in functionalities our R.scripts follow a **3-layered** approach to carry out the internal validation process (*Figure 1, steps 6-13*):  

1. **Low level:** scripts of utility/subfunctions (e.g. `subfunction_tunedRF.R`) are invoked to extract optimal hyperparameter settings from the respective predictor algorithm and/or ML-framework like the `caret` package.  
2. **Mid level:** scripts (e.g. `train_tunedRF.R`) containing the train function (e.g. `trainRF_caret_custom_tuner()`) that perform hyperparameter tuning using the corresponding subfunctions.  
3. **High level:**  in these scripts (e.g. `nestedcv_tunedRF.R`) the respective train function is embedded within the nested cross-validation scheme.  
4. **Integrative Run files:** Finally, we provide a R-markdown files (e.g. `run_nestedcv_tRF.Rmd`) for each ML-algorithm to perform the whole internal validation processes integrating the **low, mid and high level** scripts.  
5. Consequently, dedicated calibrator (e.g. `calibration_MR.R`) and performance evaluator (e.g. `performance_evaluator.R`) functions might be applied separately to their outputs.  

***

## Hardware requirements 
Our scripts require (possibly highly) multicore computers with sufficient RAM. 

The given runtimes were generated using various workstations with the following hardware specs (32-192 GB RAM) and CPUs: [1] 8 threads on i7 7700k @ 4.2GHz ; [2] 12 threads on MacBook Pro 15” i9-8950HK @ 2.9 GHz or [3]  i7-6850k @ 3.6 Ghz; [4] 32 threads on i9-7960X @ 2.8 Ghz; [5] 72 threads on AWS EC2 c5n.18xlarge @ 3.5Ghz.

Runtimes for GPU (NVIDIA CUDA-accelerated) SVM classifiers with RBF or LK (`Rgtsvm` package) were generated on NVIDIA GTX 1080Ti GPUs.

***
 
## OS & Setup requirements 

We tested our R scripts using local installations
+ both CPU and GPU on
  + Ubuntu  16.04.03-16.04.5 LTS
+ CPU only on
  + Mac OS X El Capitan 10.11.6, OS X Mojave 10.14.5 

R: A Language and Environment for Statistical Computing, v.3.3.3 - 3.5.2.

R and RStudio running in Docker containers (`rocker`) ensuring clear and dedicated software environments. `rocker` containers with R v3.5.2 and RStudio v1.1.463 were used. For details see https://www.rocker-project.org or https://github.com/rocker-org.

For SVM with GPU acceleration (R package `Rgtsvm`) consult the setup guide at https://github.com/Danko-Lab/Rgtsvm. 

For `Rgtsvm` we used:  
+ NVIDIA CUDA 8.0, cuDNN  
+ Boost library (v.1.67.0) ; http://www.boost.org/users/download/.

***

## Installation guide 

### 1. CPU-based implementations
Please make sure that the required R packages (listed above) and their dependencies are installed.
In order to directly install packages from GitHub install the `devtools` package and use the `install_github()` function.

```
# CRAN
install.packages("foo", dependencies=T)

# Install the devtools package to directly install packages from Github
install.packages("devtools")
# The corresponding function 
install_github("DeveloperName/PackageName")
```

### 2. GPU-accelerated SVM
For NVIDIA CUDA installation see detailed guide at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html.  
For Boost library, which is required for the `Rgtsvm` package, see the user guide at http://www.boost.org/users/download/.  


***

## A worked example to perform hyperparameter tuning for the random forests (tRF) algorithm and post-processing it with multinomial ridge regression (MR) 

Below, we present the steps needed to perform hyperparameter tuning for the RF classifier including its calibration with MR (tRF<sub>BS | ME | LL</sub> + MR) and its final performance evaluation.  

*Codes for the remaining ML-classifiers and calibration algorithms will be uploaded when the review process is finished.*

### 3. Load data sets & objects | Steps 1 - 6

All our scripts assume that the readily prepared benchmarking data sets (`betas.1.0.RData` – `betas.5.5.RData` ; 5.3Gb) that contain the variance filtered (10,000 CpG probes) training-test set (betas.train ; betas.test) pair for each *K.k (sub)fold* have been downloaded from the linked **Dropbox** folder (`betas.train.test.10k.filtered`) at http://bit.ly/2vBg8yc.

Alternatively, pre-processing using our scripts (https://github.com/mwsill/mnp_training/blob/master/preprocessing.R) can be carried out separately and the variance filtered data sets can be generated from Illumina scanner data using the `subfunc_load_betashdf5_subset_filter_match_save_betasKk()` function.   


```{r}
# Load the data sets 
load("./data/y.RData") # true outcome class labels for 2801 cases
# or “/home/rstudio/data/y.RData” # if using rocker container
load("./data/nfolds.RData") 
# or “/home/rstudio/data/nfolds.RData” # if using rocker container

# True outcome labels y
load("y.RData") # contains the y vector of true class labels (with 91 levels)
load("anno.RData") # only needed for performance evaluation
load("probenames.RData") # only needed for performance evaluation

# Nested resampling scheme 
load("nfolds.RData")
# contains the "nfolds" list object with the folds assignments to perform 
# it can be generated using the `makefolds.R` script.
# the nested 5 x 5-fold CV for internal validation.
```

***

### 4. Setup and import pre-requisite R packages.

These steps are also embedded into the `utility/subfunction.R` and integrative R-mardown (e.g. `run_nestedcv_tRF.md`) files.

```{r}
# Parallel backend
library(doMC)
# Random Forests classifier
library(randomForest)
# Caret framework for tuning randomForest hyperparameters
library(caret)

# Define number of cores for the parallel backend
# Consider leaving 1 thread for the operating system.
cores <- detectCores()-1 
```

***

### 5. Run R scripts necessary for tuning the RF classifier | Steps 7-10

Because the respective R package for each investigated ML-classifier algorithm has different built-in functionalities our R scripts follow a 3-layered approach to carry out the internal validation process: 

1) **subfunctions** are invoked to enable the tuning and/or to extract the optimal hyperparameter settings from the respective predictor algorithm and/or form the ML-framework of the `caret` package;  
2) **training function** (e.g. `trainRF_caret_custom_tuner()`) performs hyperparameter tuning by using the corresponding subfunctions;  
3) **nested CV**: finally, the training function is implemented within the nested cross-validation scheme and wrapped into a separate function (e.g. `run_nestedcv_tunedRF()`), which performs the complete internal validation.

```{r}
# 1. Utility/Subfunctions to define and perform custom grid search using the caret package
source("subfunctions_tunedRF.R")
```
1. This script contains: 
+ the `rfp()` function that provides a parallelized wrapper for the `randomForest::randomForest()`function.
+ `customRF` function for the caret package to enable tuning RF hyperparameters including `ntree`, `mtry` and `nodesize`.  
+ `subfunc_rf_caret_tuner_customRF()` to perform grid search using an extra nested n-fold CV with the `caret` package.
+ also sources the script `source("evaluation_metrics.R")` that define the evaluation metrics required for tuning tRF<sub>BS | ME | LL</sub>

```{r}
# 2. Training & Hyperparameter tuning & Variable selection are performed here
source("train_tunedRF.R")
```
2. This script contains:
+ a custom function (`trainRF_caret_custom_tuner()`) for the whole tuning process of RF hyperparameters including `mtry`, `ntree` and `nodesize` as well as `pvarsel` (i.e. the number of CpG probes that result in the lowest BS, LL and ME metrics).


```{r}
# 3. Source scripts for full evaluation of tRF in the nested CV scheme 
source("nestedcv_tunedRF.R")
```

3. This script contains: 
+ the function `run_nestedcv_tunedRF()` that integrates the (1.) **sub-** and (2.) **training functions** to perform the complete internal validation within the 5 x 5-fold nested CV scheme. 
+ It creates an output folder (by default `./tRF/`) and exports the resulting variables (hyperparemeter settings and raw classifier scores) into a `CVfold.1.0.RData`file for each (sub)fold, respectively.


4. Open and run the R markdown file (`run_nestedcv_tRF.Rmd`) that contains code chuncks to perform all the above listed tasks 1.-3. to train and tune the RF classifier.

```{r}
# 4. Last code chunck of the R markdown file (`run_nestedcv_tRF.Rmd`) that performs the complete training and tuning of the RF classifier.
 
# Timing: 
# for outerfolds ~ 4–4.5 h ; for the full 5 x 5-fold nested CV scheme 4 – 5 days depending on the tuning grid size.
# ~ 19-21 min/(sub)fold  | nCV=5 @ 72 threads AWS c5n.18xlarge

run_nestedcv_tunedRF(y.. = NULL, 
                     betas.. = NULL, 
                     path.betas.var.filtered = "/home/rstudio/data/betas.train.test.10k.filtered/",
                     fname.betas.p.varfilt = "betas",
                     subset.CpGs.1k = F, 
                     n.cv.folds = 5, 
                     nfolds..= NULL,    
                     K.start = 1, k.start = 0,
                     K.stop = NULL, k.stop = NULL,
                     n.cv = 5, n.rep = 1, 
                     mtry.min = 80, mtry.max = 110, length.mtry = 4, 
                     ntrees.min = 500, ntrees.max = 2000, ntree.by = 500,
                     use.default.nodesize.1.only = T, 
                     nodesize.proc = c(0.01, 0.05, 0.1),
                     p.n.pred.var = c(100, 200, 500, 1000, 2000, 5000, 7500, 10000),
                     cores = cores, 
                     seed = 1234,
                     out.path = "tRF", 
                     out.fname = "CVfold")
```


The output file `CVfold.K.k.RData` is comprised of the following objects:  

+ predicted (uncalibrated/raw) scores matrices of the tuned RF using `p.n.pred.var` (`pvarsel`) number of CpG probes that resulted in the lowest BS, ME and LL metrics, respectively:   
    + `scores.pred.rf.tuned.brier`  
    + `scores.pred.rf.tuned.miscerr`  
    + `scores.pred.rf.tuned.mlogl`  
+ `rfcv.tuned`: the output object of the (2.) `trainRF_caret_custom_tuner()` function  
+	`fold`: the corresponding *"K.k"* (sub)fold assignments with training and test/calibration sets  

*CRITICAL/Troubleshooting: the output `CVfold.K.k.RData` file can be quite large (1-1.5Gb) because it contains multiple large matrix objects (ca. 215 MB each). Hence, saving all `RData` files of the 5 x 5-fold nested CV scheme (altogether 30x) might require 30-50Gb free space on the hard drive.*

***

### 6. Calibration using multinomial ridge regression (MR) | Step 11 A-D

We provide separate scripts for each post-processing algorithm:  

+ Platt scaling with logistic regression: `calibration_Platt_LR.R`  
+ Platt scaling with Firth's penalized LR: `calibration_Platt_FLR.R`  
+ Multinomial ridge penalized LR: `calibration_MR.R`  
+ We also provide a wrapper function `calibrator_integrated_wrapper()` to apply all the above calibrators (LR, FLR and MR) with a single function call.  
+ Further, the R markdown file `calibrate_tRF.Rmd` contains worked examples for vRF and tRF calibration including the results (*Figure 1. steps 11 - 13  | Calibration*). 


```{r}
# Source the script
source(“calibration_MR.R”)
```

This script contains:  

  + the `calibrate_MR()`function that by default 
      + creates an output folder (*"tRF/MR-calibrated/"* within the working directory and   
      + generates `probsCVfold.{which.optimizer.metric}.K.0.RData`files, which are comprised of   
          + the raw `scores`and   
          + MR-calibrated probabilities `probs` matrices, plus misclassification errors for each fold
        
```{r}
# Parallel backend
# For cv.glmnet, it is recommended to register a "doMC" parallel backend 
# as it uses foreach functionalities for 10-fold CV of lambda  
library(doMC)
registerDoMC(cores=10)

# Function call:
calibrate_MR(out.path = "/vRF-calibrated-MR/",
             out.fname = "probsCVfold",
             nfolds.. = NULL,
             y.. = NULL,
             load.path.w.name = "./vRF/CVfold.",
             verbose.messages = T,
             which.optimized.metric.or.algorithm = "vanilla", 
             save.metric.name.into.output.file = T,
             parallel.cv.glmnet = T,
             setseed = 1234)
```

Timing: 
Multi-core (10 threads):  

  + Training the MR model + Predicting the outer test set: ca. 1 min 15-25s / fold / metric [BS | ME | LL]
  + Full run: ca. 7 min / metric [BS | ME | LL]
  
Single core:  

  + Training the MR model + Predicting the outer test set: ca. 4-5 mins / fold / metric (BS | ME | LL)
  + Full run: ca. 23 - 25 min / tRF<sub>BS | ME | LL</sub>  

Output object size: 400-500KB / fold (e.g. `probsCVfold.brier.1.0.RData`)
  

***

### 7. Performance evaluation | Step 12 

Use a comprehensive panel of performance metrics:  

+ For Discrimination - derived from the ROC plot:  
    + misclassification error (ME)
    + multiclass AUC (mAUC) 
+ Overall prediction performance - strictly proper scoring rules for evaluating the difference between observed class and predicted class probabilities:  
    + Brier score (BS)
    + multiclass log loss (LL)

```{r}
# Source the script for complete performance evaluation of tRF
source("performance_evaluator.R")

# The script above also sources the required evaluation metrics
source("evaluation_metrics.R")
```

This script contains:  

  + the `performance_evaluator()` function that sources the `brier.R` and `mlogloss.R`scripts to generate performance metrics.  
      + `performance_evaluator()` returns a list with elements `misc.error`, `auc.HandTill`, `brier`, and `mlogloss`.
  
```{r}
# Default function settings
performance_evaluator(load.path.w.name. = "./tRF/MR-calibrated/probsCVfold.brier.",
                      name.of.obj.to.load = NULL, # as.character ; defaults to the calibrated `probs` object
                      nfolds.. = NULL,       # looks for and gets `nfolds` from .Globalenv 
                      betas.. = NULL,        # looks for and gets `betas` from .Globalenv 
                      y.. = NULL,            # looks for and gets `y` from .Globalenv 
                      scale.rowsum.to.1 = T, # by default rescales for mAUC
                      reorder.columns = F,   # for tRF it is not required but for SVM set to T
                      reorder.rows = T,      # required for RF
                      misc.err = T, 
                      multi.auc.HandTill2001 = T, 
                      brier = T, 
                      mlogLoss = T,
                      verbose = T # gives a verbose output
                      )
                        
# Function call:
# Timing < 1 min
performance_evaluator(name.of.obj.to.load = "scores")
```

The code snippet below generates (in < 4 min) the complete performance evaluation of all tRF algorithms (3 tRF<sub>BS | ME | LL</sub> x 3 [calibrator <sub>LR | FLR | MR</sub>] x 4 [metrics<sub>ME | AUC | BS | LL</sub>]), see also (*Fig. 1, step 14* & *Table 2*). 

```{r}
# 1. Load prereq. objects (anno, probenames)
list.files("/Users/mme/Documents/GitHub/np-tRF/tRF/")
load("/Users/mme/Documents/GitHub/np-data/data/anno.RData")
load("/Users/mme/Documents/GitHub/np-data/data/probenames.RData")
load("/Users/mme/Documents/GitHub/np-tRF/tRF/")

# 2. Generate performance evaluation metrics - Run time - 30s 
# tuned RF (tRF) {BS ; ME; LL} - NO SCALING - (scores) 
tRF_BS_perfeval <- performance_evaluator(load.path.folder = "./tRF", 
                                         load.fname.stump = "CVfold", 
                                         name.of.obj.to.load = "scores.pred.rf.tuned.brier")

tRF_ME_perfeval <- performance_evaluator(load.path.folder = "./tRF", 
                                         load.fname.stump = "CVfold", 
                                         name.of.obj.to.load = "scores.pred.rf.tuned.miscerr")

tRF_LL_perfeval <- performance_evaluator(load.path.folder = "./tRF", 
                                         load.fname.stump = "CVfold", 
                                         name.of.obj.to.load = "scores.pred.rf.tuned.mlogl")
```

3. Evaluate all metrics and all calibrated versions of tRF<sub>BS|ME|LL</sub> 

```{r}
# 3. Evaluate all metrics and all calibrated versions of tRF 
try.metrics <- c("BS", "ME", "LL")
try.folder.path <- as.list(rep(paste("tRF", try.metrics, "calibrator-integrated", sep = "-"), each = 3))
try.metrics.long <- rep(c("brier", "miscerr", "mlogl"), each = 3)
try.calibrator <- c("LR", "FLR", "MR")
try.fname.stump <- as.list(paste("probsCVfold", try.metrics.long, rep(try.calibrator, 3), sep = "."))
try.l.folder.file.paths <- file.path(try.folder.path, try.fname.stump)

# 4. Function call 
# The actual function call of performance_evaluator using mapply() is just 1 line of code
try.l.perfevals.all <- mapply(FUN = performance_evaluator, load.path.folder = try.folder.path, load.fname.stump = try.fname.stump)
# The results matrix
try.l.perfevals.all # matrix

# 5. Format the output object
# Add colnames
colnames(try.l.perfevals.all) <- file.path(try.folder.path, try.fname.stump)
colnames(try.l.perfevals.all) <- try.fname.stump
# Rename
tRF_ALL_LR_FLR_MR_perfeval <- try.l.perfevals.all
```