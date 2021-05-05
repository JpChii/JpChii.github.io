# Loading and preparing Image dataset from directory using TensorFLow

We're going to checkout two ways of preparing the data here,
  1. ImageDataGenerator
  2. image_dataset_from_directory

First to use the above two libraries to load data, the images must follow a specific directory structure.

**File Structure PreRequisite**

```
# Example of file structure
directory <- top level folder
└───train <- training images
│   └───class_1
│   │   │   1008104.jpg
│   │   │   1638227.jpg
│   │   │   ...      
│   └───class_1
│       │   1000205.jpg
│       │   1647351.jpg
│       │   ...
│   
└───test <- testing images
│   └───class_1
│   │   │   1001116.jpg
│   │   │   1507019.jpg
│   │   │   ...      
│   └───class_2
│       │   100274.jpg
│       │   1653815.jpg
│       │   ...    
```

You can checkout [Daniel's preprocessing notebook](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/image_data_modification.ipynb) for preparing the data.

## ImageDataGenerator

[Definition form docs](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#used-in-the-notebooks_1) - Generate batches of tensor image data with real time augumentaion.

So What's **Data Augumentation?** - We'll cover this later in the post.

