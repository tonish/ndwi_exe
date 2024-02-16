# reads the original files and creates a single tf.data.Example record for each input image.
import tensorflow as tf
import numpy as np
import skimage 
import os

def load_imgs(path):
    """load images and labels paths from the split csv"""
    with open(path, 'r') as file:
        content = file.readlines()
    
    data_prefix = 'v1.1/data/flood_events/HandLabeled/S2Hand'
    label_prefix = 'v1.1/data/flood_events/HandLabeled/LabelHand'

    imgs = [os.path.join(data_prefix,i.split(',')[0].replace('S1Hand','S2Hand')) for i in content]
    labels = [os.path.join(label_prefix,i.split(',')[1].strip('\n')) for i in content]
    return imgs,labels

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features
def image_example(img_file, label,split):
    image = np.moveaxis(skimage.io.imread(img_file),0,-1)
    image_shape = image.shape
    image_string = image.astype('uint16').tobytes()

    label = np.moveaxis(skimage.io.imread(label),0,-1)
    label = label.astype('uint16').tobytes()
    
    feature = {
      'split': _bytes_feature(split),
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _bytes_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

    return tf.train.Example(features=tf.train.Features(feature=feature))

record_file = './images.tfrecords'

splits = {b'bolivia':'v1.1/splits/flood_handlabeled/flood_bolivia_data.csv',
          b'train':'v1.1/splits/flood_handlabeled/flood_train_data.csv',
          b'test':'v1.1/splits/flood_handlabeled/flood_test_data.csv',
          b'val':'v1.1/splits/flood_handlabeled/flood_valid_data.csv'}

#iterate splits and create the tfrecord file from all the images, keeping track on the splits
with tf.io.TFRecordWriter(record_file) as writer:
    for split_name, split_path in splits.items():
        #use split path to create images and labels lists
        images, labels = load_imgs(split_path)
        for filename, label in zip(images,labels):
            tf_example = image_example(filename, label,split_name)
            writer.write(tf_example.SerializeToString())

print('finished creating the file')