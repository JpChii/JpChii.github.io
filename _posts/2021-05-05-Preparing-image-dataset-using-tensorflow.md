# Loading and preparing Image dataset from directory using TensorFLow

We're going to checkout two ways of preparing the data here,
  1. ImageDataGenerator
  2. image_dataset_from_directory
  3. tf.data API

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

## 1.1 ImageDataGenerator

[Definition form docs](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#used-in-the-notebooks_1) - Generate batches of tensor image data with real time augumentaion.

So What's **Data Augumentation?** - We'll cover this later in the post.

To load in the data from directory, first an ImageDataGenrator instance needs to be created.

```Python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
```

Two seperate data generator instances are created for training and test data.
```Python
# Creating DataGen instances
train_datagen_10_percent = ImageDataGenerator(rescale=1/255.)
test_datagen_1_percent = ImageDataGenerator(rescale=1/255.)
```

*`rescale=1/255.` is used to scale the images between 0 and 1 because most deep learning and machine leraning models prefer data that is scaled 0r normalized.*

Let's use `flow_from_directory()` method of `ImageDataGenerator` instance to load the data. We'll load the data for both training and test data at the same time.

```Python
# Loading in the data
train_data_10_percent = train_datagen_10_percent.flow_from_directory(directory=train_dir_10_percent,
                                                                     target_size=(224, 224),
                                                                     class_mode='categorical',
                                                                     batch_size=32,
                                                                     shuffle=True)

test_data_1_percent = test_datagen_1_percent.flow_from_directory(directory=test_dir_1_percent,
                                                                     target_size=(224, 224),
                                                                     class_mode='categorical',
                                                                     batch_size=32)
```

First Let's see the parameters passes to the `flow_from_directory()`
  1. directory - The directory from where images are picked up
  2. batch_size - The images are converted to batches of 32. If we load all images from train or test it might not fit into the memory of the machine, so training the model in batches of data is good to save computer efficiency. `32` is a good batch size
  3. target_size - Specify the shape of the image to be converted after loaded from directory
  4. class_mode - class_mode is `binary` for binary classification and `categorical` for multi-class classification. Since we're only dealing with only two classes, we've passed `binary`
  5. seed - Mentioning seed to maintain consisitency if we repeat the experiments

After checking whether `train_data` is tensor or not using `tf.is_tensor()`, it returned `False`. `flow_from_directory()` returns an array of batched images and not `Tensors`.

We can checkout a single batch using `images, labels = train_data.next()`, we get image shape - (`batch_size`, `target_size`, `target_size`, `rgb`).

**Training time**: This method of loading data takes the second lowest training time in the methods being dicussesd here. For 29 classes with 300 images per class, the training in GPU took `2mins 10s` and step duration of `71-74ms`.

## 1.2 ImageDataGenerator Data Augumentation

Data Augumentation - Is the method to tweak the images in our dataset while it's loaded in training for accomodating the real worl images or unseen data.

We can implement Data Augumentaion in ImageDataGenerator using below ImageDateGenerator,

```Pyhon
# Creating DataGen instances
train_data_10_percent_aug = ImageDataGenerator(rescale=1/255.,
                                               horizontal_flip=True,
                                               zoom_range=0.2,
                                               rotation_range=0.2,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2)

# Loding in the data
train_data_10_percent_aug = train_data_10_percent_aug.flow_from_directory(directory=train_dir_10_percent,
                                                                     target_size=(224, 224),
                                                                     class_mode='categorical',
                                                                     batch_size=32)
```

There are many options for augumenting the data, let's explain the ones covered above.
  1. horizontal_flip - Flips the image in horizontal axis
  2. zoom_range - zooms in the image
  3. rotation_range - rotates the image
  4. width_shift_range - range of width shift performed
  5. height_shift_range - range of height shift performed

All other parameters are same as in `1.ImageDataGenerator`

Advantage of using data augumentation is it will give better results compared to training without augumentaion in most cases. But `ImageDataGenerator Data Augumentaion` increases the training time, because the data is augumented in CPU and the loaded into GPU for train.

**Training time**: This method of loading data takes the second lowest training time in the methods being dicussesd here. For 29 classes with 300 images per class, the training in GPU took `8mins 7s` and step duration of `355-362ms`.
