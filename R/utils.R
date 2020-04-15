
#Utils: funzioni utili che vengono richiamate negli altri script

#' @title Media Ponderata per voti di esami universitari, rispetto ai crediti
#'
#' @description Funzione a cui passo la matrice dei voti degli esami, e un vettore
#' con il numero di crediti per ogni esame. Mi restituisce un vettore delle
#' medie voto ponderate (una per ogni studente: numero di righe della matrice \code{b})
#'
#' @param b Matrice dei voti ( righe -->studenti, colonne --> esami)
#' @param c Vettore dei crediti di ogni esame (stesso ordine delle colonne di b)
#'
#' @return Vettore delle medie ponderate, \code{length(media_ponderata)=nrows(b)}
#'
#' @export

media_ponderata = function(b,c) {
  media_pond = vector(mode = 'double', length = nrow(b))
  voto = vector(mode = 'double', length = nrow(b))
  crediti = vector(mode = 'double', length = nrow(b))
  for (i in 1:nrow(b)) {
    for (j in 1:ncol(b)) {
      if (!is.na(b[[i,j]]) & (b[[i,j]] != 999)) {
        voto[i] = voto[i] + b[[i,j]] * c[j]
        crediti[i] = crediti[i] + c[j]
      }
    }
  }
  media_pond = voto/crediti
  return(media_pond)
}

#Calculate AUC Value
#' @title AUC Value from vector of predictions and labels
#'
#' @description Works for predictions made out of classifiers.
#' It gives backe the Area Under the Curve (AUC) value
#'
#' @param preds Vector of predictions (numeric)
#' If the classifier returns a matrix or dataframe,
#'we should pass only the column relative to the predition of the label 1.
#'Predictions should be in the form of probabilities of belonging to the class 1
#' @param labels Vector of true labels (Y variable) (numeric)
#'
#' @return AUC value of the classifier underlying the predictions passed as argument
#'
#' @export
#' @importFrom PRROC roc.curve
#'
auc <- function(preds,labels){
  auc_value = roc.curve(scores.class0 = preds, weights.class0=labels,)$auc
  return(auc_value)
}

# #Exit The Script
# exit <- function() {
#   .Internal(.invokeRestart(list(NULL, NULL), NULL))
# }


#CONFUSION.MATRIX (per disegnare la matrice di confusione)
# in riga valori predicted, in colonna valori reali. Nell'ultima colonna d? la precisione dei livelli 0,1 predetti
#' @title Confusion Matrix for our classifier
#'
#' @description Giving the predictions (as vector of probabilities of being class 1),
#' and the threshold value, the function returns the Confusion Matrix
#' with the chosen threshold value to assign units to class 1 or 0 accordingly.
#' Works only for binary classifiers.
#' @param preds Vector of predictions (numeric)
#' If the classifier returns a matrix or dataframe,
#'we should pass only the column relative to the predition of the label 1.
#'Predictions should be in the form of probabilities of belonging to the class 1
#' @param labels Vector of true labels (Y variable) (numeric)
#' @param threshold Number in the range [0,1], the values in preds higher than threshold are considered predicted in class 1,
#' the others belong to class 0
#'
#' @return Confusion Matrix: a dataframe with the Two entry confusion matrix and a column of class errors
#'
#' @export

confusion.matrix <- function(preds,labels,threshold=0.5) {
  prediction <- ifelse(preds > threshold, "Mod_T", "Mod_F")
  labels = labels == 1
  confusion  <- table(prediction, labels)
  if (min(dim(confusion))<2) {
    return(confusion)
  }
  else {
    confusion  <- cbind(confusion, c(round(1 - confusion[1,1]/(confusion[1,1]+confusion[2,1]),2), round(1 - confusion[2,2]/(confusion[2,2]+confusion[1,2]),2)))
    confusion  <- as.data.frame(confusion)
    names(confusion) <- c('Real_F', 'Real_T',"class_err")
    return(confusion)
  }

}


# BEST CONFUSION MATRIX, GIVEN THE MODEL
#' @title BEST CONFUSION MATRIX, GIVEN THE MODEL
#'
#' @description Finds the best confusion matrix (hence the best threshold)
#' given the predictions of some classifier.
#' To do so, it looks for the confusion matrix
#' which maximises the sum of sensitivity and specificity.
#' Works only for binary classifiers.
#' @param preds Vector of predictions (numeric)
#' If the classifier returns a matrix or dataframe,
#'we should pass only the column relative to the predition of the label 1.
#'Predictions should be in the form of probabilities of belonging to the class 1
#' @param labels Vector of true labels (Y variable) (numeric)
#'
#' @return A confusion matrix, with a column with class errors
#' @export
#' @importFrom PRROC roc.curve

best.confusion <- function(preds,labels) {

  roc = roc.curve(scores.class0 = preds, weights.class0=labels,curve=TRUE)
  valori_roc = roc$curve
  colnames(valori_roc) = c("false_positive_rate","sensitivity","threshold")
  sum_spec_tpr = 1-valori_roc[,"false_positive_rate"]+valori_roc[,"sensitivity"]
  threshold = valori_roc[which.max(sum_spec_tpr),"threshold"]
  cat("Migliore Matrice di confusione \n\n")
  confusion.matrix(preds,labels,threshold = threshold)
}


#COND_NUM

#' @title Evaluate the Condition Number of a matrix, allowing to discard variables
#'
#' @description Calcolo il condition number di una matrice di dati.
#'  Pensata per capire se c'è collinearità nella matrice X (correlazioni lineari alte tra gruppi di variabili).
#'  Per ottenere condition number reliable devo mettere le variabili sulla stessa scala. SCRIVERE PERCHÈ
#'  Quindi la funzione standardizza le variabili, e toglie le variabili contenute nella lista \code{vars_to_discard},
#'  poi calcola il condition number della matrice risultante.
#'  Esempi di variabili da scartare per calcolare il condition number:
#' We should discard all the non numeric (character) variables (or encode them using the model matrix),
#' discard also the response variable, and the intercept (da NaN se faccio lo scale)"
#'
#' @param dataset Matrice (o anche dataframe) delle variabili X
#' @param vars_to_discard lista con nomi delle variabili da non considerare nel calcolo del condition number
#'
#' @return Print the condition number and return it as a numeric value
#' @importFrom dplyr select
#' @export

cond_num <- function(dataset,vars_to_discard=NULL){
  "Vars_to_discard should be a list of variable names.
  We should discard all the non numeric (character) variables (or encode them using the model matrix),
  discard also the response variable, adn the intercept (da NaN quando faccio lo scale)"

  if (missing(vars_to_discard)){
    matrix = scale(dataset)
    } else {
      matrix = as.data.frame(dataset) %>%
        select(-vars_to_discard) %>%
        scale()
      }
cond_num = kappa(matrix)
cat("Condition Number:\n",cond_num,"\n\n")
return(cond_num)
}


#' @title Normalize the Test Dataset, based on Train set statistics
#'
#' @description Lo faccio per non avere data leakage tra Train e Test.
#' Normalizzo le variabili, quindi le metto tutte nel range [0,1] (del Train set),
#' quindi è possibile che ci siano anche valori minori di zero e maggiori di 1 (dato che lo applico sul test set)
#'
#' @param X_test Dataframe delle variabili X del Test Set
#' @param stats_train Lista con variabili chiamate max e min. Ognuna di queste è una lista con Nome_Variabile: Valore
#'
#' @return Un Dataframe che è il dataframe X_test con tutte le variabili normalizzate
#' @export

normalize_test = function (X_test,stats_train){
  X_test_scaled = sweep(X_test,2,stats_train$min)/ t(replicate(nrow(X_test),stats_train$max-stats_train$min))
  return(X_test_scaled)
}



