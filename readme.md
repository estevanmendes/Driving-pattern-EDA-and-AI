# Quick intro

Acceration and rotation were used to create a model that classifies the driving behaviour in two categories: Normal and Agressive. 

An Exploratory data analysis was done, followed by the delevepment of an AI model using machine learning and deep Learning 


## Code Outline

The jupyter notebook outline:

- EDA Training dataset

- EDA Testing dataset

- Data cleaning and preprocessing

- Data Loader
    - Denoising data
    - [Dataloader class](Analysis/dataloader/dataloader.py)

- ML models 
    - [ML base class](Analysis/models/ml_models.py)
    - Logistic Regression
    - SVM
    - Random Forest
    - Naive Bayes
    - XGBRegressor     
- DL models
    - [DL Base class](Analysis/models/dl_models.py)
    - MLP models

    - CNN model following [1] with recurrence plot aproach

    - Transfer learning - MobileNetV2

    - RNN models (GRU,LSTM)

- Hyperparams Finetunnig 
    - Random Grid Search CV
        - [Code](Analysis/model_selection/random_grid_search_cv.py) 
    - Grid Search CV
        - [Code](Analysis/model_selection/grid_search_cv.py)
    - Best model evaluation

## Hightlights

For data filtering was implemented 3 methods was in the data loader, gaussian filter, exponential decay, and rolling average. 

Few signal processing methods were applied in the EDA of each sample.


## Dataset

The  [dataset](https://www.kaggle.com/datasets/outofskills/driving-behavior) was retrieved from Kaggle


# Results

## Best model metrics

The best model found using accuracy, balanced accuracy, AUC, precision and recall was LSTM.

|  Metric |  Mean (%)  | Std (%)  |
|---|---|---|
| Balanced Accuracy  |70   | 1 | 
| Recall  | 67  | 1  | 
| Precision  | 53  | 4   | 
| Accuracy  | 71  | 3   | 
| AUC  | 70  | 1   | 

Since we a very concerned to detect properly the agrevisse behaviour the higher recall was prioritaized. 



## Analysis Package

A package was created contaning the dataloader codes and the machine learning and deep learning base code which speed up the trainning, validation, testing and evalution process.


Also, there are few code for hyperparams finetuning using Keras-tunner. It was implementaded a random search CV, and GridSearch CV. 

# Futher informations

## References


[1] [Shahverdy, Mohammad, et al. "Driver behavior detection and classification using deep convolutional neural networks." Expert Systems with Applications 149 (2020): 113240.](https://www.sciencedirect.com/science/article/abs/pii/S095741742030066X)

[2][ Spiegel, Stephan, Johannes-Brijnesh Jain, and Sahin Albayrak. "A recurrence plot-based distance measure." Translational Recurrences: From Mathematical Theory to Real-World Applications. Springer International Publishing, 2014.](https://link.springer.com/chapter/10.1007/978-3-319-09531-8_1)

[3] [Bland, J. Martin, and DouglasG Altman. "Statistical methods for assessing agreement between two methods of clinical measurement." The lancet 327.8476 (1986): 307-310.](https://pubmed.ncbi.nlm.nih.gov/2868172/)



