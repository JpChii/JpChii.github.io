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
