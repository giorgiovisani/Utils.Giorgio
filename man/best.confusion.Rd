% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{best.confusion}
\alias{best.confusion}
\title{BEST CONFUSION MATRIX, GIVEN THE MODEL}
\usage{
best.confusion(preds, labels)
}
\arguments{
\item{preds}{Vector of predictions (numeric)
If the classifier returns a matrix or dataframe,
we should pass only the column relative to the predition of the label 1.
Predictions should be in the form of probabilities of belonging to the class 1}

\item{labels}{Vector of true labels (Y variable) (numeric)}
}
\value{
A confusion matrix, with a column with class errors
}
\description{
Finds the best confusion matrix (hence the best threshold)
given the predictions of some classifier.
To do so, it looks for the confusion matrix
which maximises the sum of sensitivity and specificity.
Works only for binary classifiers.
}
