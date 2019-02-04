# ml4calibrated450k

## Overview 
This is a companion repository for the article *"Comparative analysis of machine learning workflows to estimate class probabilities for precision cancer diagnostics on DNA methylation microarray data"* submitted to Nature Protocols.

Our comaprisons included four well-established machine learning algorithms: random forests (RF), elastic net penalized multinomial logistic regression (ELNET), support vector machines (SVM) and boosted trees (XGBOOST).

For calibration we used i) Platt scaling implemented by logistic regression (LR), Firth's penalized LR; and ii) ridge penalized multinomial regression (MR). 

All algorithms were compared on an uqinque data set of brain tumor DNA methylation reference cohort (n=2801 cases belonging to 91 classes) published in:

Capper, D., Jones, D. T. W., Sill, M. and et al. (2018a). 
"DNA methylation-based classification of central nervous system tumours. Nature, 555, 469." 
https://www.nature.com/articles/nature26000

The corresponding Github repository using a MR calibrated RF classifier and pre-processing steps is available at https://github.com/mwsill/mnp_training. 

The 450k DNA methylation array data of the reference cohort is available in the Gene Expression Omnibus under the accession number GSE109381 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109381).

The benchmarking data set based on the 10,000 most variable CpG probes can be easily generated using R scripts provided in the above repository (https://github.com/mwsill/mnp_training). 
A smaller subset of the reference DNA methylation cohort data containing only the 1000 most variable CpG probes (`betas1000.RData`) is provided for direct download at this repository. The true class label vector `y` is also directly downloadable from here. 

***

## Repo content

All algorithms were implementated and evaluated within:
1. 5 x 5 fold nested cross validation (CV) scheme  
  + R package: base R

### 1. Machine learning classifiers:
+ Random Forests (RF) 
  + vanilla RF (using default settings; vRF)
  + tuned RF (tRF)
     + Brier score (BS)
     + Misclassification error (ME)
     + Multiclass log loss (LL)
  + R package: randomForest, caret
+ Elastic net penalized multinomial logistic regression (ELNET) 
  + concurrent tuning of alpha and lambda 
  + R package: glmnet
+ Support vector machines (SVM)
  + Radial Basis Function kernels (RBF)
  + Linear kernels (LK)
  + R packages: e1071, ksvm (caret), liblineaR; GPU (NVIDIA CUDA accelerated) Rgtsvm
+ Gradient boosted decision trees (XGBOOST)
  + comperehensive tuning of multiple tuning parameters
  + R package: xgboost, caret
  
### 2. Calibration/Post-processing algortihms:
+ Platt scaling 
  + Logistic Regression (LR)
      + R package: glm (base R)
  + Firth's penalized LR (FLR) 
      + R package: brglm 
+ Ridge penalized multinomial logistic regression (MR)
      + R package: glmnet
      
### 3. Performance evaluation: 
We also provide scripts for evaluation such as:
+ Misclassification error (ME)
+ Multiclass AUCH as published by Hand and Till (2001) 
      + R package: HandTill2001
+ Brier score (BS)
+ Mutliclass log loss (mLL) 
  + using the Kaggle formulation https://web.archive.org/web/20160316134526/https://www.kaggle.com/wiki/MultiClassLogLoss.

***

## Hardware requirements 
Our scripts require (possibly highly) multicore computers with sufficient RAM. 

The given runtimes were generated using either a workstation with specs of 64 GB RAM, Intel i7 6850k CPU (6 cores/12 thread @ 3.6 GHz) or AWS instances (general purpose M.2 64 cores or compute optimized C.2 16 cores).

Runtimes for GPU (NVIDIA CUDA accelerated) SVM classifires with RBF or LK (Rgtsvm package) were generated on NVIDIA GTX 1080Ti GPUs.

***
 
## OS & Setup requirements 

We tested our R notebooks under 
+ Both CPU and GPU 
  + Ubuntu  16.04.03 LTS
+ CPU only 
  + Mac OS X El Capitan 10.11.6, OS X Mojave 10.14.2 

R 3.3.3 - 3.4

For SVM with GPU acceleration 
  + Setup guide for Rgtsvm see https://github.com/Danko-Lab/Rgtsvm
    + NVIDIA CUDA 8.0, cuDNN
    + Boost library http://www.boost.org/users/download/

***

## Installation guide 


### 1. CPU-based implementations
Please install the R packages dependencies (see above)
```
# CRAN
install.packages("foo", dependencies=T)

# Github / devtools - to directly install packages from Github
install_github()

```

### 2. GPU-accelerated SVM
For NVIDIA CUDA installation see detailed guide at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
For Boost library required for the Rgtsvm package see the user guide at http://www.boost.org/users/download/


***

### Example to perform hyperparameter tuning for the random forests algorithm

Below, we present the steps needed to perform hyperparameter tuning for the RF classifier including its calibration with MR (tRF<sub>BS | ME | LL</sub> + MR) and its final performance evaluation.

### 3. Load data sets & objects

```{r}
# Load data sets 
load("MNPbetas10Kvar.RData") 
# contains betas matrix (2801 x 10000) and y (vector of 2801) true outcome labels

# Betas1000.RData is also provided with the 1000 most variable CpG probes after unsupervised variance filtering
load("betas1000.RData")
# True outcome labels y
load("y.RData")

load("nfolds.RData")
# contains the nfolds list for nested 5 x 5-fold CV
```
  
### 4. Setup and import pre-requisite R packages.

```
# Parallel backend
library(parallel) 
library(doParallel)
# Random Forests classifier
library(randomForest)
# Caret framework for tuning randomForest hyperparameters
library(caret)

# Define number of cores for the parallel backend
# Consider leaving 1 thread for the operating system.
cores <- detectCores()-1 
```

### 5. Source R.scripts necessary for tuning and fitting the RF classifier

We use a 3-layered approach for each ML-classifier algorithm including: 
1. subfunctions 
2. training functions and finally the 
3. nested CV 

```
# 1. Subfunctions to define and perform custom grid search using the caret package
source("subfunctions_tunedRF.R")
# This script contains the funtion trainRF_caret_custom_tuner() 

# 2. Training & Hyperparameter tuning & Variable selection performed here
source("train_tunedRF.R")

# 3. Source scripts for full evaluation of tRF in the nested CV scheme 
source("nestedcv_tunedRF.R")

# Run the function that performs the task
run_nestedcv_tunedRF(y.. = y, betas.. = betas, 
                     n.cv.folds = 5, 
                     nfolds.. = nfolds,
               # nfolds is imported via the load("nfolds.RData")
                     cores = 10, 
                     seed = 1234, 
                     K.start = 1, k.start = 0,
                     out.path = "tRF/", out.fname = "CVfold", # (1)
                     mtry.min = NULL, mtry.max = NULL, length.mtry = 2, # (2)
                     ntrees.min = 1000, ntrees.max = 2000, ntree.by = 500,
                     nodesize.proc = c(0.01, 0.05, 0.1), #(3)
                     p.n.pred.var = c(100, 500, 1000, 10000)
                     )

```

### 5. Perform calibration using ridge penalized  multinomial logistic regression (MR)

```
# Source the script
source("calibration_tRF.R")
```

### 6.	Performance evaluation

Use a comprehensive panel of performance metrics: 
+ For Discrimination - derived from the ROC plot: 
  + misclassification error (ME)
  + multiclass AUC (mAUC) 
+ Overall prediction performance - strictly proper scoring rules for evaluating the difference between observed class and predicted class probabilities: 
  + Brier score (BS)
  + multiclass log loss (LL)

```
# Source the script for complete performance evaluation of tRF
source("performance_evaluation_tRF.R")
```

