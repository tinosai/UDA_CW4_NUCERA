# Unstructured Data Analysis: Final Coursework - Fortunato Nucera
Repository for the final project of Unstructured Data Analysis (2022)

I declare that I have worked on this assessment independently. 

## Compulsory Files Descriptions
This section includes the description of the compulsory files - requested in the assessment question sheet - to be included for the report.

* `README.md` is the current file, providing a description of the project folder.
* `dataset.parquet` is the data set file, measuring 99.1MB. This file is easily readable using the Python library `Pandas`.
* `UDA_FinalProject_NUCERA.pdf` is the report file, of length 12 pages, which contains all the details and explanations concerning the current project.
* `UDA_FinalProject_NUCERA.ipynb` is the file containing all the code to reproduce the findings of the report. I have used my personal CID as random seed. In particular, the cells which are lengthy and may require multiple runs have a heading seed setup so that reproducibility is guaranteed.

This completes the list of the compulsory files.

## Non-Compulsory Files Descriptions
Along with the compulsory files, we also provide some non-compulsory files which may aid the grader in proceeding through the notebook more quickly.

* `requirements.txt` contains the list of all the Python packages in the environment I have worked in. Dissimilarities between different runs of the notebook on different machines can be attributed to the different versions of the used packages.
* `model_03000_huberloss.pt` **This file is too large for GitHub. However, people inside the Imperial College network can download it from [this link to my college OneDrive](https://imperiallondon-my.sharepoint.com/:u:/g/personal/fn321_ic_ac_uk/EQNZwrKsj-hCrced0XMius4BAPI4SIuimGOBLjsxdoZRRA?download=1)**. This file contains the U-Net model and its weights, after 3000 epochs. It takes a long time to train this model and we therefore stored the weights for future use (for example, for fine-tuning, should this be necessary). If this file is not present in folder the notebook is in, the training routine will automatically kick in and start training the network for 3000 epochs. Mind that, depending on the GPU, this may take even more than 1 day. Training on CPU is even more time consuming and therefore deemed unfeasible for the current project. 
* `image_rep.png` is an image which we have used to produce the pictures of the denoising algorithms outputs across the notebook. Section 6 of the notebook will not run if this image is not available.
* `survey_result` is the folder containing the results of the surveys we administered to 2 surveyees. Section 5.2.2 will not run if this folder and its contents are not available. The folder contains:  

  * `survey_u0.pdf` : the survey form filled by the first surveyee.
  * `survey_u1.pdf` : the survey form filled by the second surveyee.
  * `survey_u0.csv` : the results for the first surveyee in csv format.
  * `survey_u1.csv` : the results for the second surveyee in csv format.
  * `answer_0.csv`  : the ground truth for the first surveyee, indicating which model the images in each page of `survey_u0.pdf` belong to.
  * `answer_1.csv`  : the ground truth for the second surveyee, indicating which model the images in each page of `survey_u1.pdf` belong to.

The notebook will automatically create and populate a `survey` folder, therefore we renamed this folder to `survey_result` to avoid overwriting.
  
* `preprocessing.py` is a script which includes all the commands we used to go from the original data set downloaded from [Barbu's website](http://adrianbarburesearch.blogspot.com/p/renoir-dataset.html) to `dataset.parquet`. Make sure to modify the string `ENTER_ALIGNED_PICTURES_FOLDER_HERE` with the full path of the aligned picture folders on your machine if you decide to run the preprocessing step. Mind that pre-processing is not necessary as the file `dataset.parquet` includes the already pre-processed images.

* `img_noisy.png` and `img_ref.png` are used to calculate the loss for a different camera and a different image size on a single picture, and show how U-Net may - in some cases - may exhibit a worse performance than its competitors.

## How to run the code
In order to run the code, you should:  
- clone the current repository to your local working directory.
- decide on whether you want to train the neural network from scratch. If you want to use the pre-defined weights, you can simply download them 
from the OneDrive link provided in the appropriate section and place the file in the working directory (recommended).
- run the jupyter notebook `UDA_FinalProject_NUCERA.ipynb`. Please pay attention to the extra instructions provided in `Other Information` below.

Also, even though not compulsory, the use of a GPU is recommended. We run the code on both a MacBook Pro 2021 16-inch with 64GB of memory and 10 cores and 
on a workstation in the cloud featuring 200GB of memory, 30 cores, and an nVidia A100 GPU with 40GB of dedicated memory. Use an nVidia GPU for testing if possible. 

The most time consuming sections of the code are the NLM/NMF calculation (about 1.5 hours in total) and the U-Net training (which can vary from hours to days depending on the hardware ???) .

## Other Information:
In the notebook, `UDA_FinalProject_NUCERA.ipynb`, the images are exported using the LaTex style for the labels. Change:
```{python}
plt.rcParams["text.usetex"] = True
```
to:
```{python}
plt.rcParams["text.usetex"] = False
```
If LaTex is not available on your machine. In addition, in Section 5.2.1, when the surveys are exported, the setup of a font becomes necessary.
On a Mac, we simply set:
```{python}
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Tahoma.ttf",18)
```
But you need to make sure that, on your machine, this font is properly included. For a list of fonts on your machine, simply run:
```{python}
import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
```
And replace `/System/Library/Fonts/Supplemental/Tahoma.ttf` with the full path of the font you wish to use.


This completes the `README` file.
