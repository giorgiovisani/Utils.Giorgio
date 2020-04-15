#'@title Function to Test Lasso Stability on a given dataset (uses bootstrap samples)
#' @description Function to create bootstrap samples of our original dataset,
#' Divides each bootstrap dataset into Train and Test (keeping the same ratio for the response variable),
#' Performs normalization of the variables (transforming in the range [0,1]), if \code{params$stand} is set to \code{TRUE}
#' Otherwise, the standardization is done inside the Lasso model, using the \eqn{\frac{X-\bar{x}}{\sigma}}
#' Creates the model matrix as given by the formula (this way, we may specify interactions and how to encode categorical variables)
#' Performs Lasso using cross-validation (with number of folds as assigned in \code{params})
#' Returns the value of AUC and the list of important variables retrieved by Lasso in the bootstrap sample
#'
#'
#' @param boot_idx Indices f the sampled units in the bootstrap sample
#' @param dataset Original Dataset
#' @param params List of params (comprising \code{n_folds,stand,train_size,lasso_select})
#' @param model_formula Model formula to be used in Lasso Regression
#' @param contrast_list List of contrasts to be applied to create the design matrix
#'
#' @return The AUC value of the Lasso Model on the bootstrap dataset,
#' The list of important varaibles retrieved by Lasso (chooses lambda according to \code{params$lasso_select})
#' @examples
#'
#' @export
#'
#' @importFrom caret createDataPartition
#' @importFrom stats model.matrix coef predict
#' @importFrom BBmisc normalize
#' @importFrom dplyr if_else
#'


bootstrap_lasso = function(boot_idx,
                           dataset,
                           params,
                           model_formula,
                           contrast_list) {

  Y_var_name = all.vars(model_formula)[1]

  boot_sample = dataset[boot_idx,]
  boot_sample = boot_sample[sample(nrow(boot_sample)),]   # shuffle dataset
  train.index <- caret::createDataPartition(boot_sample[[Y_var_name]],
                                            p = params$train_size, list = FALSE)

  dati_encoded = stats::model.matrix(model_formula, data = boot_sample, contrasts=contrast_list)
  dati_encoded = dati_encoded[,colnames(dati_encoded) != "(Intercept)"]
  X_train <- dati_encoded[train.index,]
  X_test <- dati_encoded[-train.index,]

  Y_train <- as.data.frame(boot_sample)[train.index,Y_var_name]
  Y_test <- as.data.frame(boot_sample)[-train.index,Y_var_name]

  stats_train_encoded = list(max=apply(X_train,2,max),min=apply(dati_encoded,2,min))
  X_train_scaled = BBmisc::normalize(X_train,method = "range",range = c(0,1),margin = 2)
  X_test_scaled = normalize_test(X_test = X_test,stats_train = stats_train_encoded)

  weight_0_obs =  sum(Y_train == 1) / sum(Y_train == 0)
  weights = if_else(Y_train == 1,1,weight_0_obs)

  if (params$stand == "manual") {data = X_train_scaled; data_new = X_test_scaled; st = FALSE
  } else {data =X_train; data_new = X_test; st = TRUE}

  vlambda=exp(seq(-10,1,length=401))
  cvfit = cv.glmnet(data,Y_train, family="binomial",alpha=1, lambda = vlambda, weights = weights, type.multinomial="grouped",
                    type.measure = "deviance",nfolds = params$n_folds, standardize=st)

  coeff_matrix = coef(cvfit, s = params$lasso_select)
  imp_feat_idxs = nonzero(coeff_matrix)[,1]
  imp_features = rownames(coeff_matrix)[imp_feat_idxs]

  preds = predict(cvfit, newx = data_new, s = params$lasso_select,type="response")
  auc_value = auc(preds = preds, labels = as.double(Y_test)-1)
  return(list(auc_value=auc_value,imp_features=imp_features,num_features = length(imp_features)))
}
