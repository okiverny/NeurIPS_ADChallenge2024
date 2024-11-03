# NeurIPS_ADChallenge2024

Here is the code developped for [NeurIPS - Ariel Data Challenge 2024](https://www.kaggle.com/competitions/ariel-data-challenge-2024/overview) challenge on Kaggle. It contains the 17th place solution based on simple implemetation of parametric fitting. There are also other strategies here but they were not really tested on the competition.

Link to the Kaggle notebook: [kaggle/parametric-fits-17th-place](https://www.kaggle.com/code/olehkivernyk/parametric-fits-17th-place)

## 1. Calibration procedure
I used the calibration pipeline implemented in C from [this link](https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/531453).

Kaggle notebook with calibration step: [Notebook 1: calibration of train data](https://www.kaggle.com/code/olehkivernyk/neurips-starter).

## 2. Signal Analysis
Kaggle notebook: [NeurIPS_ModelTraining](https://www.kaggle.com/code/olehkivernyk/neurips-modeltraining/). This is the first study of data.
Kaggle notebook: [NeurIPS_TransitModel](https://www.kaggle.com/code/olehkivernyk/neurips-transitmodel/). This is the study of transition parameterization with `Erf` function.

### 2.1 Background subtraction