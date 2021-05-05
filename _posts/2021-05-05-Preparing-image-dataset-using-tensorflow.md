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

To load in the data from directory, first an ImageDataGenrator instance needs to be created.

```Python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
```

Two seperate data generator instances are created for training and test data.

Let's use `flow_from_directory()` method of `ImageDataGenerator` instance to load the data. We'll load the data for both training and test data at the same time.

```Python
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                              batch_size=32,
                                              target_size=(224,224),
                                              class_mode="binary",
                                              seed=42)

test_data = test_datagen.flow_from_directory(directory=test_dir,
                                            batch_size=32,
                                            target_size=(224,224),
                                            class_mode="binary",
                                            seed=32)
```

First Let's see the parameters passes to the `flow_from_directory()`
  1. directory - The directory from where images are picked up
  2. batch_size - The images are converted to batches of 32. If we load all images from train or test it might not fit into the memory of the machine, so training the model in batches of data is good to save computer efficiency. `32` is a good batch size
  3. target_size - Specify the shape of the image to be converted after loaded from directory
  4. class_mode - class_mode is `binary` for binary classification and `categorical` for multi-class classification. Since we're only dealing with only two classes, we've passed `binary`
  5. seed - Mentioning seed to maintain consisitency if we repeat the experiments

After checking whether `train_data` is tensor or not using `tf.is_tensor()`, it returned `False`. `flow_from_directory()` returns an array of batched images and not `Tensors`.

Let's checkout a single batch using `images, labels = train_data.next()`, we get image shape - (`batch_size`, `target_size`, `target_size`, `rgb`).

### ImageDataGenerator Data Augumentation

