# Stroke Prediction

In this blog, we're going to explore the stroke prediction from kaggle and try to come up with a model to predict whether a person has stroke or not?

[Link to dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)

[Link to notebook](https://github.com/JpChii/ML-Projects/blob/main/end-to-end-stroke-prediction.ipynb)

## Exploratory data analysis

We're going to see how the features influence or impact `stroke` - target. First let's list down the features and then proceed with the impact with cool😎 plots.

### List of columns/features
1. `id` - unique value for each sample
2. `gender` - gender of the sample 
    - values [ Male, Female, Other ]
3. `age`
4. `hypertension` - whether the person has Blood Pressure or not
5. `heart_disease` - whether the person has heart disease or not
6. `ever_married` - whether the person was married before
    - values [ yes, No ]
7. `work_type` - what kind of sector the person works in
    - values [ Private, Self-employed, Govt_job, children, Never_worked ]
8. `Residence_type` - Where the person is located
    - values [ Urban, Rural ]
9. `avg_glucose_level` - blood sugar level of the person
10. `bmi` - body mass index, a value that indicates whether a person has approriate heigh and weight
11. `smoking_status` - person's smoking status
     - values [ formerly smoked, never smoked, smokes, Unknown ]
12. `stroke` - target parameter, whether person will have stroke or not

***Plot 1: `stroke` value counts***

We've an unbalanced dataset with lot's of samples available for false positive. Has a ratio of 95:5 for True negative: True Positive

<img src="/images/stroke/target-value-counts.png">

***Plot 2: How `gender` affects `stroke`***

**Women** are more susceptible to have stroke than**Men**, There's an outlier with one sample in **other** gender we'll drop the sample.

<img src="/images/stroke/stroke-gender.png">

***Plot 3: How whether a person is or was impacts affects `stroke`***

People who is or was married are more susceptible to have stroke. To all the single out there, at least you won't have stroke any time soon😜🤣.

<img src="/images/stroke/ever-married-stroke.png">

***Plot 4: How `residence_type` impacts `stroke`***

Urban people have a little edge compared to rural people on stroke possiblity.

<img src="/images/stroke/residence-type-stroke.png">

***Plot 5: How `smoking_type` impacts `stroke`***

*Smoking is injurious to health and environment*. Looking at the plot,smoking has caused a little percent of them a stroke. **never smoked** are affected by stroke higher than other categories, but never_smoked contributes to 37% of the dataset.

<img src="/images/stroke/smoking-type-stroke.png">

***Plot 6: How `work_type` impacts `stroke`***

People in **private** are more susceptibe to stroke compared to other categories, but *private* category contributes to is 57% dataset.

<img src="/images/stroke/work-type-stroke.png">

***Plot 7: How `hyper_tension` impacts `stroke`***

Hypertension has high possiblity of becoming a stroke.

<img src="/images/stroke/hyper-tension-stroke.png">

***Plot 8: How `age`, `bmi` impacts `stroke`***

Older people with low bmi have a possiblity of stroke.

<img src="/images/stroke/age-bmi-stroke.png">

***Plot 9: How `age`, `avg_glucose_level` impacts `stroke`***

Older people with low glucose level have a possiblity of stroke.

<img src="/images/stroke/age-glucose-stroke.png">

****Plot 10: One last EDA for this blog****

An correlation matrix between features and how they affect each others.

<img src="/images/stroke/correlation-matrix.png">

## Preparing the data

The below mentioned cztegorical features are converted to numerical fetaures.

1. gender
    - Male: 0
    - Female: 1
2. ever_married
    - Yes: 0
    - No: 1
3. work_type
    - Private: 0
    - Self-Employed: 1
    - Govt_job: 2
    - children: 3
    - Never_worked: 4
4. Residence_type
    - Urban: 0
    - Rural: 1
5. smoking_status
    - formerly smoked: 0
    - never smoked: 1
    - smoked: 2
    - Unknown: 3

## Modelling

1. The data is splitted into train and test using `train_test_split`
2. Initially We've taken the below the classifiers and modelled them on the training data
    - KNeighborsClassifier
    - LogisticRegression
    - RandomForestClassifier

```Python
# Put models in a dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier()}

def fit_and_score(models, x_train, x_test, y_train, y_test):
    """
    Model to fit the data to a model and score the model with test data
    models - dictionary of models
    x_train - training features
    x_test - test features
    y_train - training features
    y_test - test_features
    """
    np. random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        model_scores[name] = model.score(x_test, y_test)
    return model_scores
```

After training we've got the below `accuracy scores` for the models

<img src="/images/stroke/model-accuracy.png">

### Hypertuning
Let's Hypertune the above models to see if we can get better accuracy.

#### 1. KNeighborsClassifier with neigbhors range from (0,21)

<img src="/images/stroke/Neighbor-score.png">

No improvement on model after hypertuning the n_neigbors.

#### 2. RandomizedSearchCV RandomForestClassifier

The below hyperparameters provided better performance than the base model.

```{'n_estimators': 260,
 'min_samples_split': 8,
 'min_samples_leaf': 13,
 'max_depth': 10}
 ```
 
## Evaluation
 
Since this is an imbalanced dataset, Accuracy won't be enough to confirm the model's performance. There are few ways to work on imbalanced sets. 

Let's find below metrics on how the model performs on both classes - One of the ways to work on imbalanced sets.
 
#### 1. ROC Curve
 
<img src="/images/stroke/roc-curve.png">

We've an AUC of 0.84 which is not bad.

#### 2. Confusion matrix

Confusion matrix will give more clear picture on model's predictions on majority and minority classes.

<img src="/images/stroke/conf-matrix.png">
 
The model is unable to predict anything for class 1. Since machine learning tries to mimize the error at most, model is choosing the majority class which will be at 95%. The minority class is really important for stroke prediction because if we predict True Potive(Stroke) as False Postive(No Stroke) will be major pain to the patients. Let's see the `F1-score` and `mac-avg` as our last evaluation of the model.

#### 3. Classification report

The f1-score for class 1 is 0 and macro avg is 0.49 further confirming classification report, that the model is not at all predicting the minority class.

```
precision    recall  f1-score   support

           0       0.95      1.00      0.97       929
           1       0.00      0.00      0.00        53

    accuracy                           0.95       982
   macro avg       0.47      0.50      0.49       982
weighted avg       0.89      0.95      0.92       982
```

Okay now we're getting how big an impact the imbalance dataset is having on our models.

## Imbalanced dataser Experimentation

This is my first binary classification modelling experience on an imbalanced dataset. There are two way's here

1. Algorithm-based
2. Dataset-based

I went Dataset-based approach. Since am using scikit-learn for modelling, I've researched the docs and found `imblearn`. `imblearn` has oversampling, undersampling and [more methods](https://imbalanced-learn.org/stable/user_guide.html) to play around.

### 1. RandomOverSampling

This sampling technique creates new samples for the minority class. It creates new samples from current available samples.

*Resampling code*

```Python
from sklearn.datasets import make_classification
X, Y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                          n_redundant=0, n_repeated=0, n_classes=2,
                          n_clusters_per_class=1,
                          weights=[0.1, 0.9],
                          class_sep=0.8, random_state=0)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_resample(X, Y)
```

Now we've gor resampled data, let's split, train and score them to see if we get any improvement for monority class.

The model performs extremley well, which might be overfitting but this is better compored to F1-score of 0.00 for class 1.

#### 1.1 ClassificationReport

Better on all metrics accuracy, f1-score, macro avg.

```
              precision    recall  f1-score   support

           0       1.00      0.95      0.97       938
           1       0.95      1.00      0.97       855

    accuracy                           0.97      1793
   macro avg       0.97      0.98      0.97      1793
weighted avg       0.98      0.97      0.97      1793
```

#### 1.2 ConfusionMatrix

<img src="/images/stroke/conf-matrix-ros.png">

After oversampling model is capable of predicitions on minority class.

### 1.3 ROC curve

Got great auc of 0.98 😮.

<img src="/images/stroke/roc-ros.png">

### 2. SMOTE

Let's try one last oversampling technique SMOTE.

#### 2.1 ClassificationReport

Better on all metrics accuracy, f1-score, macro avg compared to base model.

```
              precision    recall  f1-score   support

           0       1.00      0.95      0.97       938
           1       0.95      1.00      0.97       855

    accuracy                           0.97      1793
   macro avg       0.97      0.98      0.97      1793
weighted avg       0.98      0.97      0.97      1793
```

#### 2.2 ConfusionMatrix

<img src="/images/stroke/conf-matrix-smote.png">

After oversampling model is capable of predicitions on minority class.

## Fina Decision

We can use the oversampled models whcih gived better performance on both classed compared to minority models.
