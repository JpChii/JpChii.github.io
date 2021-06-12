# Guide-to-machine-learning-as-a-service

I've been creating machine learning models for sometime now. So the model's are just sitting in my storage??

What was the purpose of creating these models, obviously to be used by applications. 
How can i make my models accesible to an application? This blog is my journey on finding the answer for this question.

For all the modelling experiments and evalation, am using jupyter notebooks or google colab. But this might not be the case when we are exposing the model to be accesed by other service.
We need a standard directory strucure and `.py` files and well-defined dependecies.

I came across [Real Python article on application layouts](https://realpython.com/python-application-layouts/), The structure am gonna use to build the api as follows,

```
model_name/
|
|--api/
|
|
|-- bin/
|
|-- docs/
|   |-preprocessing.md
|   |-model-io.md
|
|-- src/
|   |-- __init__.py
|   |-- runner.py
|   |-- preprocess
|       |-- __init__.py
|       |-- preprocess.py
|
|-- README.md
|-- requirements.txt
```

* api directory will contain the routes
* bin directory is the starting point of the api
* docs has documentation on the application
* src has the bulk of functionalities with preprocessing and model prediction
* README.md is the high level introduction to the application
* requirements.txt has the dependent packages of the application
