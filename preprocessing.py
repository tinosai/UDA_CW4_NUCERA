# # Data Pre-processing (This code does not need to be run!)

# # Image preparation

# Please note this script does not need to be run. We will provide the pre-processed images in the file named `dataset.parquet`. However, this script will provide evidence of the pre-processing steps we have 
# taken, from the source, up to the dataset export stage.
# 
# The original dataset is available [here](http://adrianbarburesearch.blogspot.com/p/renoir-dataset.html). We have contacted the author and obtained his permission to use and share a modified copy of the data set for the purpose of this project. We have downloaded the three aligned datasets `Canon T3i`, `Canon S90`, and `Xiaomi Mi3` and saved them in a folder, please replace `ENTER_ALIGNED_PICTURES_FOLDER_HERE` with the full path of the aligned images folder on your computer.

import cv2
import glob
import tqdm
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shutil import rmtree, copyfile, copytree
np.random.seed(70103)

### clean up the environment
#### 1. remove the "data directory" and create a new one
os.makedirs("data")

#### 2. remove the images in local folder
for file in glob.glob("*.png"):
    os.remove(file)
    
#### 3. Copy the "Aligned" folders in "data"
for folder in glob.glob("ENTER_ALIGNED_PICTURES_FOLDER_HERE/*Aligned*"):
    copytree(folder, os.path.join("data",folder.split("/")[-1]))


# ## 1. Remove irrelevant images and files
# Remove all the objects which are not needed for the current analysis.
picturesToRemoveList = glob.glob("./**/*_Aligned/**/*full*") + glob.glob("./**/*_Aligned/**/Thumbs*") + glob.glob("./**/*_Aligned/**/*plot*") + glob.glob("./**/*_Aligned/*.txt") + glob.glob("./**/*_Aligned/**/Mask*.bmp") 
for file in picturesToRemoveList:
    os.remove(file)

# ## 2. Convert the images to grayscale and save to .png (original files are in .bmp)
# We list up all the images in the `Mi3_Aligned`, `S90_Aligned` and `T3i_Aligned` folders, then we convert the `bmp` format into `png` and, finally, resize all the pictures to half of their dimensions in order to have more interesting data in the image patches.
picturesList = glob.glob("./**/*_Aligned/**/*.bmp")
print(f"There are {len(picturesList)} images to convert")

for image in tqdm.tqdm(picturesList):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ### shrink images
    gray = cv2.resize(gray, (gray.shape[1]//2,gray.shape[0]//2))
    cv2.imwrite(image.replace(".bmp",".png"),gray)


# ## 3. Remove the .bmp pictures since they have been replaced.
picturesToRemoveList = glob.glob("./**/*_Aligned/**/*.bmp")
for file in picturesToRemoveList:
    os.remove(file)

# ## 4. Discard some noisy images
# We need to discard more data. Each of the subfolders contain some `Batch` folders which indicate a scene. Each scene features a *reference* (the ground truth) and some *noisy images* (input for the denoising model).
# If there is only `1` noisy picture, this is kept. If there are `2` or more noisy pictures in the `Batch` folder, then a random number is sampled from a categorical with number of categories $k$ corresponding to the number of noisy images in the folder. Then the image corresponding to the index sampled from the categorical distribution is kept.
foldersToCheck = glob.glob("./**/**/Batch*")
for folder in foldersToCheck:
    ## retrieve the noisy images from the folder
    noisy_images = glob.glob(os.path.join(folder,"*Noisy.png"))
    ## there must be at least a noisy image
    assert len(noisy_images)>=1
    ## if only one picture is noisy, keep it. Else, sample from categorical and remove the pictures not corresponding to the sampled index.
    if len(noisy_images)>1:
        idx = np.random.randint(len(noisy_images))
        ## remove the sampled image from the list of the pictures to delete. Send to a dummy. We don't need it.
        noisy_images.pop(idx);
        ## delete the other pictures
        for delImage in noisy_images:
            os.remove(delImage)

            
### THIS PART CAN BE IGNORED BUT IS KEPT FOR REPRODUCIBILITY ##############################
copyfile("./data/Mi3_Aligned/Batch_014/IMG_20160210_062948Reference.png", "reference.png")
copyfile("./data/Mi3_Aligned/Batch_014/IMG_20160210_063005Noisy.png", "input.png")
rmtree("./data/Mi3_Aligned/Batch_014")
img = cv2.imread("input.png")[:600,500:1000]
cv2.imwrite("input.png",img)
img = cv2.imread("reference.png")[:600,500:1000]
cv2.imwrite("reference.png",img)            
###########################################################################################

# ## 5. Dataset generation
print(f"We now have a total of {len(glob.glob('./**/**/**/*.png'))} images")

# There are in total 240 images, of which 120 include noise and 120 do not. These images are, however, very large. We can therefore split each large image into multiple images of smaller size and square shape (the square shape is not strictly necessary, but it allows the convolutional neural network to optimize memory handling).
# 
# The split can be done in two different ways:
# - sliding window
# - partitioning
# 
# The *sliding window* method allows to obtain more smaller images from a single large image, as the smaller patches are allowed to overlap. This however can lead to issues of data leakage of the test set into the training set. *Partitioning* allows to deal with the issue by making sure that the intersection between two different patches of the image is the empty set.
# 
# For this particular project, we decide to split each image into a `128x128` pixel patches. The remainder of the pixels at the border of the images that are too few to result in a patch are discarded.
allImages = glob.glob("./**/**/**/*.png")
size = 128 #set up the size to split in
imageDictionary = defaultdict(lambda : {"input" : 0, "target" :0})
tot_images = 0 #keep track of the total number of images
for image in tqdm.tqdm(allImages):
    ### set up the image in a dictionary
    flag = "input" if "Noisy" in image else "target"
    name = image.split("/")[-3].lower() + "_" + image.split("/")[-2].lower()
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    h , w = img.shape
    new_h, new_w = h//size * size , w//size * size    
    imageDictionary[name][flag] = img[:new_h, :new_w].reshape(h//size, size, -1, size).swapaxes(1,2).reshape(-1,size,size)
    tot_images += imageDictionary[name][flag].shape[0]

print(f"The total number of patches is {tot_images}")

idx = 50
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax[0].imshow(imageDictionary["s90_aligned_batch_011"]["input"][idx,...],cmap="gray")
ax[1].imshow(imageDictionary["s90_aligned_batch_011"]["target"][idx,...],cmap="gray")
ax[0].axis("off")
ax[1].axis("off")
plt.suptitle("Comparison of Images: Input (left) and Target (right)")
plt.tight_layout()


# ## 6. Group the images in a large matrix, subsample them, and save them to parquet

# Instantiate a large numpy matrix to store the image data. We will encode the three different cameras as:
# 
# 0. MI3
# 1. T3I
# 2. S90
# 
# These values will be saved in the first column of the numpy matrix.

numpy_matrix = np.zeros((tot_images,2*size**2+1), dtype=np.uint8)
i = 0
keys = sorted(list(imageDictionary.keys()))
camera_dict = {"mi3":0, "t3i":1, "s90":2}
for key in tqdm.tqdm(keys):
    value = imageDictionary[key]
    ### concatenate images horizontally
    concatenated_images = np.concatenate((value["input"].reshape(value["input"].shape[0],-1),value["target"].reshape(value["target"].shape[0],-1)), axis=1)
    concatenated_images = np.concatenate((np.array([camera_dict[key.split("_")[0]]]*concatenated_images.shape[0]).reshape(-1,1), concatenated_images), axis=1)
    numpy_matrix[i:i+concatenated_images.shape[0],:] = concatenated_images
    i+=concatenated_images.shape[0]


# We then discard the "uninteresting" pictures (those which are black almost everywhere). We then keep only the images whose:
# - sum of intensities is larger than 10 (remove all black pictures)
# - the standard deviation is larger than 25 (keep pictures featuring large intensity variations).
numpy_matrix = numpy_matrix[np.logical_and((numpy_matrix[:,1:1+size**2]).sum(axis=1) > 10,(numpy_matrix[:,1:1+size**2]).std(axis=1) > 25),:]


# Given the 100Mb storage limit, we will only keep 4050 patches from the original dataset. In order to obtain the best results, we first shuffle the dataset, then each of the used cameras will be allocated 1350 images.

### 1. randomly permute the numpy matrix
numpy_matrix = numpy_matrix[np.random.permutation(numpy_matrix.shape[0]),:]
### 2. allocate 1350 images for each camera
camera_each = 1350
numpy_matrix = numpy_matrix[np.concatenate((np.where(numpy_matrix[:,0]==0)[0][:camera_each],np.where(numpy_matrix[:,0]==1)[0][:camera_each],np.where(numpy_matrix[:,0]==2)[0][:camera_each])),:]


# Then we carry out a sanity check to make sure 1350 samples have been drawn from each camera.
for i in range(3):
    assert np.sum(numpy_matrix[:,0] == i) == camera_each


# And, in the end, we export the image data matrix to a `parquet` file. `Apache Parquet` allows quick data load and high compression for large datasets. The use of `parquet` with the `brotli` scheme is necessary to limit the size of the output file.
parq_table = pa.table({f"i_{i}": numpy_matrix[i,:] for i in range(numpy_matrix.shape[0])})
pa.parquet.write_table(parq_table, "dataset.parquet", compression="brotli")

# This concludes the data wrangling from the original data set. We have reduced the original dimension from 11.3Gb to 98Mb.

#### REMOVE UNNEEDED PICTURES###################
os.remove("input.png")
os.remove("reference.png")
################################################