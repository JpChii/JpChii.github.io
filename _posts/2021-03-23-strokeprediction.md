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

We've an unbalanced dataset with lot's of samples available for false positive.

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
