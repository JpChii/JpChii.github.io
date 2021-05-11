# Loading Image dataset from directory using TensorFLow

This blog discusses three ways to load data for modelling,
  1. ImageDataGenerator
  2. image_dataset_from_directory
  3. tf.data API

First to use the above methods of loading data, the images must follow below directory structure.

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
  4. class_mode - class_mode is `binary` for binary classification and `categorical` for multi-class classification.
  5. seed - Mentioning seed to maintain consisitency if we repeat the experiments

After checking whether `train_data` is tensor or not using `tf.is_tensor()`, it returned `False`. `flow_from_directory()` returns an array of batched images and not `Tensors`.

We can checkout a single batch using `images, labels = train_data.next()`, we get image shape - (`batch_size`, `target_size`, `target_size`, `rgb`).

**Training time**: This method of loading data gives the second highest training time in the methods being dicussesd here. For 29 classes with 300 images per class, the training in GPU(Tesla T4) took `2mins 9s` and step duration of `71-74ms`.

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
  2. zoom_range - Zooms in the image
  3. rotation_range - Rotates the image
  4. width_shift_range - Range of width shift performed
  5. height_shift_range - Range of height shift performed

All other parameters are same as in `1.ImageDataGenerator`

Advantage of using data augumentation is it will give better results compared to training without augumentaion in most cases. But `ImageDataGenerator Data Augumentaion` increases the training time, because the data is augumented in CPU and the loaded into GPU for train.

**Training time**: This method of loading data has highest training time in the methods being dicussesd here. For 29 classes with 300 images per class, the training in GPU(Tesla T4) took `7mins 53s` and step duration of `345-351ms`

There's another way of data augumentation using `tf.keras.experimental.preporcessing` which reduces the training time.

**Return Type**: Return type of `ImageDataGenerator.flow_from_directory()` is `numpy array`.

[Other Methods of ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#methods_2)

## 2. image_dataset_from_directory

Let's checkout how to load data using `tf.keras.preprocessing.image_dataset_from_directory`,

```Python
# Loading in the data
train_data_10_percent_idfd = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir_10_percent,
                                                                                 label_mode='categorical',
                                                                                 image_size=(224,224),
                                                                                 batch_size=32,
                                                                                 shuffle=True)

test_data_1_percent_idfd = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir_1_percent,
                                                                                 label_mode='categorical',
                                                                                 image_size=(224,224),
                                                                                 batch_size=32)
```

Checking the parameters passed to `image_dataset_from_directory`,
  1. directory - The directory from where images are picked up
  2. label_mode - This is similar to class_mode in `ImageDataGenerator`, `binary` for binary classification and `categorical` for multi-class classification
  3. image_size - Specify the shape of the image to be converted after loaded from directory
  4. batch_szie - The images are converted to batches of 32. If we load all images from train or test it might not fit into the memory of the machine, so training the model in batches of data is good to save computer efficiency. `32` is a good batch size

We can checkout the data using snippet below, we get image shape - (`batch_size`, `target_size`, `target_size`, `rgb`).

```Python
train_1s_idfd = train_data_10_percent_idfd.take(1)
for i, l in train_1s_idfd:
  print(f"Shape of the image: {i.shape,}")
  print(f"Shape of the label: {l.shape}")
  print(f"Image : {i[0]}")
  print(f"Label: {l[0]}")
```

**Training time**: This method of loading data gives the second lowest training time in the methods being dicussesd here. For 29 classes with 300 images per class, the training in GPU took `1min 55s` and step duration of `83-85ms`.

**Return Type**: Return type of `image_dataset_from_directory` is [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) `image_dataset_from_directory` which is a advantage over `ImageDataGenerator`.

# 3. tf.data API

This first two methods are naive data loading methods or input pipeline. One big consideration for any ML practitioner is to have reduced experimenatation time. Without proper input pipelines and huge amount of data(1000 images per class in 101 classes) will increase the training time massivley. 

`tf.data API` offers methods using which we can setup better perorming pipeline.

[Methods and code used are based on this documentaion](https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control)

To load data using `tf.data API`, we need functions to preprocess the image. Why this function is needed will be understodd in further reading...

#### 3.1 Create POSIX path using pathlib
```Python
from pathlib import Path
train_dir_path_10_percent = Path("/content/asl_10_percent")
test_dir_path_1_percent = Path("/content/asl_10_percent_test")
```

#### 3.2 Get the image count
```Python
image_count_train = len(list(train_dir_path_10_percent.glob('*/*.jpg')))
image_count_test = len(list(test_dir_path_1_percent.glob('*/*.jpg')))
```

#### 3.3 Getting the list of files and shuffling it
```Python
# Getting the list of files for train
train_ds = tf.data.Dataset.list_files(file_pattern=str(train_dir_path_10_percent/'*/*'), shuffle=False)
train_ds = train_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=False)

# Getting the list of files for test
test_ds = tf.data.Dataset.list_files(file_pattern=str(test_dir_path_1_percent/'*/*'), shuffle=False)
test_ds = test_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=False)
```

#### 3.4 Function to return one_hot encoded labels
```Python
def get_label(file_path):
  part = tf.strings.split(file_path, os.path.sep)

  # One hot encode the label
  one_hot = part[-1] == class_names
  return tf.one_hot(tf.argmax(one_hot), 29)
```

#### 3.5 Function to decode image
```Python
def decode_image(img):
  # Convert the compressed string to a uint8 tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image to desired shape
  return tf.image.resize(img, [img_size, img_size])
```

#### 3.6 Function returns image and one_hot encoded label using 3.4 and 3.5
```Python
def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_image(img)

  return img, label
```

*Now we're ready to load the data, let's write it and explain it later.*

```Python
# Load in the data
img_size = 224
# Train
train_data_10_percent_da = train_ds.map(map_func=process_path,
                                       num_parallel_calls=tf.data.AUTOTUNE)

train_data_10_percent_da = train_data_10_percent_da.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Test
test_data_1_percent_da = test_ds.map(map_func=process_path,
                                       num_parallel_calls=tf.data.AUTOTUNE)

test_data_1_percent_da = test_data_1_percent_da.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)
```

1. `map()` - is used to map the preprocessing function over a list of filepaths which return img and label
  a. `map_func` - pass the preprocessing function here
  b. `num_parallel_calls` - this takes care of parallel processing calls in map and we're using `tf.data.AUTOTUNE` for better parallel calls

2. Once `map()` is completed, `shuffle()`, `bactch()` are applied on top of it.
  a. `buffer_size` - Ideally, buffer size will be length of our trainig dataset. But if it's huge amount line 100000 or 1000000 it will not fit into memory. So it's better to use buffer_size of 1000 to 1500

3. `prefetch()` - this is the most important thing improving the training time. what it does is while one batching of data is in progress, it prefetches the data for next batch, reducing the loading time and in turn training time compared to other methods.

**Training time**: This method of loading data gives the lowest training time in the methods being dicussesd here. For 29 classes with 300 images per class, the training in GPU(Tesla T4) took `1min 13s` and step duration of `50ms`.

**Return Type**: Return type of `tf.data API` is [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

*At the end, it's better to use `tf.data API` for larger experiments and other methods for smaller experiments.*

**Summary:**

Data Loading methods are affecting the training metrics too, which cna be explored in the below table.

GPU: Tesla T4

| Loading Method                               | Training Time | Step Time  | loss  |accuracy | val_losss   | val_accuracy |
| -------------------------------------------- | ------------- | ---------- | ----- |-------- | ----------- | ------------ |
| ImageDataGenerator                           | 2mins 9s      |  71-74ms   |0.6653 | 0.8026  |  6.4802     |  0.2034      |
| ImageDataGenerator with Data Augumentation   | 7mins 53s     |  345-351ms |3.2899 | 0.1245  |  13.2588    |  0.0345      |
| image_dataset_from_directory                 | 1min 55s      |  83-85ms   |23.9092| 0.8178  |  437.2661   |  0.1862      |
| tf.data API                                  | 1min 13s      |  50ms      |0.0000 | 1.0000  |  13804.7002 |  0.0000      |

[Source Notebook](https://github.com/JpChii/ML-Projects/blob/main/asl_computer_vision.ipynb) - This notebook explores more than Loading data using TensorFlow, have fun reading... 😁
