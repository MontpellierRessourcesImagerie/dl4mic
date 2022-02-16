from __future__ import print_function
# Keras imports
from keras import models
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger # we currently don't use any other callbacks from ModelCheckpoints
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as keras

# General import

import numpy as np
import os
import glob
from skimage import img_as_ubyte, io, transform
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from pathlib import Path
import shutil
import random
import time
import csv
import sys
from math import ceil

# Imports for QC
from PIL import Image
from scipy import signal
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from skimage.util import img_as_uint
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr


from sklearn.feature_extraction import image
from skimage import img_as_ubyte, io, transform
from skimage.util.shape import view_as_windows

# Suppressing some warnings
import warnings
warnings.filterwarnings('ignore')



def create_patches(Training_source, Training_target, patch_width, patch_height):
  """
  Function creates patches from the Training_source and Training_target images. 
  The steps parameter indicates the offset between patches and, if integer, is the same in x and y.
  Saves all created patches in two new directories in the /content folder.

  Returns: - Two paths to where the patches are now saved
  """
  DEBUG = False

  Patch_source = os.path.join('../../unet-data','img_patches')
  Patch_target = os.path.join('../../unet-data','mask_patches')
  Patch_rejected = os.path.join('../../unet-data', 'rejected')
  

  #Here we save the patches, in the /content directory as they will not usually be needed after training
  if os.path.exists(Patch_source):
    shutil.rmtree(Patch_source)
  if os.path.exists(Patch_target):
    shutil.rmtree(Patch_target)
  if os.path.exists(Patch_rejected):
    shutil.rmtree(Patch_rejected)

  os.mkdir(Patch_source)
  os.mkdir(Patch_target)
  os.mkdir(Patch_rejected) #This directory will contain the images that have too little signal.
  

  all_patches_img = np.empty([0,patch_width, patch_height])
  all_patches_mask = np.empty([0,patch_width, patch_height])

  for file in os.listdir(Training_source):

    img = io.imread(os.path.join(Training_source, file))
    mask = io.imread(os.path.join(Training_target, file),as_gray=True)

    if DEBUG:
      print(file)
      print(img.dtype)

    # Using view_as_windows with step size equal to the patch size to ensure there is no overlap
    patches_img = view_as_windows(img, (patch_width, patch_height), (patch_width, patch_height))
    patches_mask = view_as_windows(mask, (patch_width, patch_height), (patch_width, patch_height))
    #the shape of patches_img and patches_mask will be (number of patches along x, number of patches along y,patch_width,patch_height)

    all_patches_img = np.concatenate((all_patches_img, patches_img.reshape(patches_img.shape[0]*patches_img.shape[1], patch_width,patch_height)), axis = 0)
    all_patches_mask = np.concatenate((all_patches_mask, patches_mask.reshape(patches_mask.shape[0]*patches_mask.shape[1], patch_width,patch_height)), axis = 0)

  number_of_patches = all_patches_img.shape[0]
  print('number of patches: '+str(number_of_patches))

  if DEBUG:
    print(all_patches_img.shape)
    print(all_patches_img.dtype)

  for i in range(number_of_patches):
    img_save_path = os.path.join(Patch_source,'patch_'+str(i)+'.tif')
    mask_save_path = os.path.join(Patch_target,'patch_'+str(i)+'.tif')

    # if the mask conatins at least 5% of its total number pixels as mask, then go ahead and save the images
    pixel_threshold_array = sorted(all_patches_mask[i].flatten())
    if pixel_threshold_array[int(round(len(pixel_threshold_array)*0.95))]>0:
      io.imsave(img_save_path, img_as_ubyte(normalizeMinMax(all_patches_img[i])))
      io.imsave(mask_save_path, convert2Mask(normalizeMinMax(all_patches_mask[i]),0))
    else:
      io.imsave(Patch_rejected+'/patch_'+str(i)+'_image.tif', img_as_ubyte(normalizeMinMax(all_patches_img[i])))
      io.imsave(Patch_rejected+'/patch_'+str(i)+'_mask.tif', convert2Mask(normalizeMinMax(all_patches_mask[i]),0))

  return Patch_source, Patch_target


def estimatePatchSize(data_path, max_width = 512, max_height = 512):

  files = os.listdir(data_path)
  
  # Get the size of the first image found in the folder and initialise the variables to that
  n = 0 
  while os.path.isdir(os.path.join(data_path, files[n])):
    n += 1
  (height_min, width_min) = Image.open(os.path.join(data_path, files[n])).size

  # Screen the size of all dataset to find the minimum image size
  for file in files:
    if not os.path.isdir(os.path.join(data_path, file)):
      (height, width) = Image.open(os.path.join(data_path, file)).size
      if width < width_min:
        width_min = width
      if height < height_min:
        height_min = height
  
  # Find the power of patches that will fit within the smallest dataset
  width_min, height_min = (fittingPowerOfTwo(width_min), fittingPowerOfTwo(height_min))

  # Clip values at maximum permissible values
  if width_min > max_width:
    width_min = max_width

  if height_min > max_height:
    height_min = max_height
  
  return (width_min, height_min)

def fittingPowerOfTwo(number):
  n = 0
  while 2**n <= number:
    n += 1 
  return 2**(n-1)


def getClassWeights(Training_target_path):

  Mask_dir_list = os.listdir(Training_target_path)
  number_of_dataset = len(Mask_dir_list)

  class_count = np.zeros(2, dtype=int)
  for i in range(number_of_dataset):
    mask = io.imread(os.path.join(Training_target_path, Mask_dir_list[i]))
    mask = normalizeMinMax(mask)
    class_count[0] += mask.shape[0]*mask.shape[1] - mask.sum()
    class_count[1] += mask.sum()

  n_samples = class_count.sum()
  n_classes = 2

  class_weights = n_samples / (n_classes * class_count)
  return class_weights

def weighted_binary_crossentropy(class_weights):

    def _weighted_binary_crossentropy(y_true, y_pred):
        binary_crossentropy = keras.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * class_weights[1] + (1. - y_true) * class_weights[0]
        weighted_binary_crossentropy = weight_vector * binary_crossentropy

        return keras.mean(weighted_binary_crossentropy)

    return _weighted_binary_crossentropy


def save_augment(datagen,orig_img,dir_augmented_data="/content/augment"):
  """
  Saves a subset of the augmented data for visualisation, by default in /content.

  This is adapted from: https://fairyonice.github.io/Learn-about-ImageDataGenerator.html
  
  """
  try:
    os.mkdir(dir_augmented_data)
  except:
        ## if the preview folder exists, then remove
        ## the contents (pictures) in the folder
    for item in os.listdir(dir_augmented_data):
      os.remove(dir_augmented_data + "/" + item)

    ## convert the original image to array
  x = img_to_array(orig_img)
    ## reshape (Sampke, Nrow, Ncol, 3) 3 = R, G or B
    #print(x.shape)
  x = x.reshape((1,) + x.shape)
    #print(x.shape)
    ## -------------------------- ##
    ## randomly generate pictures
    ## -------------------------- ##
  i = 0
    #We will just save 5 images,
    #but this can be changed, but note the visualisation in 3. currently uses 5.
  Nplot = 5
  for batch in datagen.flow(x,batch_size=1,
                            save_to_dir=dir_augmented_data,
                            save_format='tif',
                            seed=42):
    i += 1
    if i > Nplot - 1:
      break

# Generators
def buildDoubleGenerator(image_datagen, mask_datagen, image_folder_path, mask_folder_path, subset, batch_size, target_size):
  '''
  Can generate image and mask at the same time use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
  
  datagen: ImageDataGenerator 
  subset: can take either 'training' or 'validation'
  '''
  seed = 1
  image_generator = image_datagen.flow_from_directory(
      os.path.dirname(image_folder_path),
      classes = [os.path.basename(image_folder_path)],
      class_mode = None,
      color_mode = "grayscale",
      target_size = target_size,
      batch_size = batch_size,
      subset = subset,
      interpolation = "bicubic",
      seed = seed)
  
  mask_generator = mask_datagen.flow_from_directory(
      os.path.dirname(mask_folder_path),
      classes = [os.path.basename(mask_folder_path)],
      class_mode = None,
      color_mode = "grayscale",
      target_size = target_size,
      batch_size = batch_size,
      subset = subset,
      interpolation = "nearest",
      seed = seed)
  
  this_generator = zip(image_generator, mask_generator)
  for (img,mask) in this_generator:
      # img,mask = adjustData(img,mask)
      yield (img,mask)


def prepareGenerators(image_folder_path, mask_folder_path, datagen_parameters, batch_size = 4, target_size = (512, 512)):
  image_datagen = ImageDataGenerator(**datagen_parameters, preprocessing_function = normalizePercentile)
  mask_datagen = ImageDataGenerator(**datagen_parameters, preprocessing_function = normalizeMinMax)

  train_datagen = buildDoubleGenerator(image_datagen, mask_datagen, image_folder_path, mask_folder_path, 'training', batch_size, target_size)
  validation_datagen = buildDoubleGenerator(image_datagen, mask_datagen, image_folder_path, mask_folder_path, 'validation', batch_size, target_size)

  return (train_datagen, validation_datagen)


# Normalization functions from Martin Weigert
def normalizePercentile(x, pmin=1, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):#dtype=np.float32
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x



# Simple normalization to min/max fir the Mask
def normalizeMinMax(x, dtype=np.float32):
  x = x.astype(dtype,copy=False)
  x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
  return x


# def predictionGenerator(Data_path, target_size = (256,256), as_gray = True):
#   for filename in os.listdir(Data_path):
#     if not os.path.isdir(os.path.join(Data_path, filename)):
#       img = io.imread(os.path.join(Data_path, filename), as_gray = as_gray)
#       img = normalizePercentile(img)
#       # img = img/255 # WARNING: this is expecting 8bit images
#       img = transform.resize(img,target_size, preserve_range=True, anti_aliasing=True, order = 1) # liner interpolation
#       img = np.reshape(img,img.shape+(1,))
#       img = np.reshape(img,(1,)+img.shape)
#     yield img


# def predictionResize(Data_path, predictions):
#   resized_predictions = []
#   for (i, filename) in enumerate(os.listdir(Data_path)):
#     if not os.path.isdir(os.path.join(Data_path, filename)):
#       img = Image.open(os.path.join(Data_path, filename))
#       (width, height) = img.size
#       resized_predictions.append(transform.resize(predictions[i], (height, width), preserve_range=True, anti_aliasing=True, order = 1))
#   return resized_predictions


# This is code outlines the architecture of U-net. The choice of pooling steps decides the depth of the network. 

def unet(pretrained_weights=None, input_size=(256, 256, 1), pooling_steps=4, learning_rate=1e-4, verbose=True,class_weights=np.ones(2)):
    inputs = Input(input_size)
    channels = 64
    convs = []
    dropout_steps = 2

    # Downsampling
    for p in range(0, pooling_steps + 1):
        if p == 0:
            convs.append(Conv2D(channels * pow(2, p), 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs))
        else:
            convs.append(Conv2D(channels * pow(2, p), 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool))
        convs[p] = Conv2D(channels * pow(2, p), 3, activation='relu', padding='same', kernel_initializer='he_normal')(convs[p])
        if p != pooling_steps:
            if p > pooling_steps-dropout_steps:
                convs[p] = Dropout(0.5)(convs[p])
            pool = MaxPooling2D(pool_size=(2, 2))(convs[p])
        else:
            drop = Dropout(0.5)(convs[p])

    # Upsampling
    for p in range(pooling_steps - 1, -1, -1):
        if p == pooling_steps - 1:
            up = Conv2D(channels * pow(2, p), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop))
        else:
            up = Conv2D(channels * pow(2, p), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(convs[-1]))

        merge = concatenate([convs[p], up], axis=3)
        convs.append(Conv2D(channels * pow(2, p), 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge))
        convs[-1] = Conv2D(channels * pow(2, p), 3, activation='relu', padding='same', kernel_initializer='he_normal')(convs[-1])


    conv = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(convs[-1])  # activation = 'relu'
    conv = Conv2D(1, 1, activation='sigmoid')(conv)

    model = Model(inputs=inputs, outputs=conv)
    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['acc'])
    # model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_binary_crossentropy(class_weights))

    if verbose:
        model.summary()
        
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def predict_as_tiles(Image_path, model):

  # Read the data in and normalize
  Image_raw = io.imread(Image_path, as_gray = True)
  Image_raw = normalizePercentile(Image_raw)

  # Get the patch size from the input layer of the model
  patch_size = model.layers[0].output_shape[1:3]

  # Pad the image with zeros if any of its dimensions is smaller than the patch size
  if Image_raw.shape[0] < patch_size[0] or Image_raw.shape[1] < patch_size[1]:
    Image = np.zeros((max(Image_raw.shape[0], patch_size[0]), max(Image_raw.shape[1], patch_size[1])))
    Image[0:Image_raw.shape[0], 0: Image_raw.shape[1]] = Image_raw
  else:
    Image = Image_raw

  # Calculate the number of patches in each dimension
  n_patch_in_width = ceil(Image.shape[0]/patch_size[0])
  n_patch_in_height = ceil(Image.shape[1]/patch_size[1])

  prediction = np.zeros(Image.shape)

  for x in range(n_patch_in_width):
    for y in range(n_patch_in_height):
      xi = patch_size[0]*x
      yi = patch_size[1]*y

      # If the patch exceeds the edge of the image shift it back 
      if xi+patch_size[0] >= Image.shape[0]:
        xi = Image.shape[0]-patch_size[0]

      if yi+patch_size[1] >= Image.shape[1]:
        yi = Image.shape[1]-patch_size[1]
      
      # Extract and reshape the patch
      patch = Image[xi:xi+patch_size[0], yi:yi+patch_size[1]]
      patch = np.reshape(patch,patch.shape+(1,))
      patch = np.reshape(patch,(1,)+patch.shape)

      # Get the prediction from the patch and paste it in the prediction in the right place
      predicted_patch = model.predict(patch, batch_size = 1)
      prediction[xi:xi+patch_size[0], yi:yi+patch_size[1]] = np.squeeze(predicted_patch)


  return prediction[0:Image_raw.shape[0], 0: Image_raw.shape[1]]
  



def saveResult(save_path, nparray, source_dir_list, prefix='', threshold=None):
  if not os.path.exists(save_path):
      os.makedirs(save_path)
  for (filename, image) in zip(source_dir_list, nparray):
      io.imsave(os.path.join(save_path, prefix+os.path.splitext(filename)[0]+'.tif'), img_as_ubyte(image)) # saving as unsigned 8-bit image
      
      # For masks, threshold the images and return 8 bit image
      if threshold is not None:
        mask = convert2Mask(image, threshold)
        io.imsave(os.path.join(save_path, prefix+'mask_'+os.path.splitext(filename)[0]+'.tif'), mask)


def convert2Mask(image, threshold):
  mask = img_as_ubyte(image, force_copy=True)
  mask[mask > threshold] = 255
  mask[mask <= threshold] = 0
  return mask


def getIoUvsThreshold(prediction_filepath, groud_truth_filepath):
  prediction = io.imread(prediction_filepath)
  ground_truth_image = img_as_ubyte(io.imread(groud_truth_filepath, as_gray=True), force_copy=True)

  threshold_list = []
  IoU_scores_list = []

  for threshold in range(0,256): 
    # Convert to 8-bit for calculating the IoU
    mask = img_as_ubyte(prediction, force_copy=True)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0

    # Intersection over Union metric
    intersection = np.logical_and(ground_truth_image, np.squeeze(mask))
    union = np.logical_or(ground_truth_image, np.squeeze(mask))
    iou_score = np.sum(intersection) / np.sum(union)

    threshold_list.append(threshold)
    IoU_scores_list.append(iou_score)

  for threshold in range(0, 256):
    # Convert to 8-bit for calculating the IoU
    mask = img_as_ubyte(prediction, force_copy=True)
    mask = 255-mask
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0

    # Intersection over Union metric
    intersection = np.logical_and(ground_truth_image, np.squeeze(mask))
    union = np.logical_or(ground_truth_image, np.squeeze(mask))
    iou_score = np.sum(intersection) / np.sum(union)

    threshold_list.append(threshold)
    IoU_scores_list.append(iou_score)

  return (threshold_list, IoU_scores_list)



# -------------- Other definitions -----------
W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red
prediction_prefix = ''
