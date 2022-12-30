# Unstructured Data Analysis: Final Coursework - Fortunato Nucera
Repository for the final project of Unstructured Data Analysis (2022)

I declare that I have worked on this assessment independently. 

## Compulsory Files Descriptions
This section includes the description of the compulsory files - requested in the assessment question sheet - to be included for the report.

* `README.html` is the current file, providing a description of the project folder.
* `dataset.parquet` is the data set file, measuring 99.1MB. This file is easily readable using the Python library `Pandas`.
* `UDA_FinalProject_NUCERA.pdf` is the report file, of length 12 pages, which contains all the details and explanations concerning the current project.
* `UDA_FinalProject_NUCERA.ipynb` is the file containing all the code to reproduce the findings of the report. I have used my personal CID as random seed. In particular, the cells which are lengthy and may require multiple runs have a heading seed setup so that reproducibility is guaranteed.

This completes the list of the compulsory files.

## Non-Compulsory Files Descriptions
Along with the compulsory files, we also provide some non-compulsory files which may aid the grader in proceeding through the notebook more quickly.

* `requirements.txt` contains the list of all the Python packages in the environment I have worked in. Dissimilarities between different runs of the notebook on different machines can be attributed to the different versions of the used packages.
* `model_03000_huberloss.pt`. **This file is too large for GitHub. However, people inside the Imperial College network can download it from [this link to my college OneDrive](https://imperiallondon-my.sharepoint.com/:u:/g/personal/fn321_ic_ac_uk/EQNZwrKsj-hCrced0XMius4BAPI4SIuimGOBLjsxdoZRRA?download=1)**. This file contains the U-Net model and its weights, after 3000 iterations. It takes a long time to train this model and we therefore stored the weights for future use (for example, for fine-tuning, should this be necessary). If this file is not present in folder the notebook is in, the training routine will automatically kick in and start training the network for 3000 iterations. Mind that, depending on the GPU, this may take even more than 1 day. Training on CPU is even more time consuming and therefore deemed unfeasible for the current project. 
* `image_rep.png` is an image which we have used for producing pictures of the denoising algorithms outputs across the notebook. Section `6` of the notebook will not run if this image is not available.
* `survey_result` is a folder of the surveys we administered to 2 surveyees. Section 5.2.2 will not run if this folder and its contents are not available. The folder contains:  

  * `survey_u0.pdf` : the survey form filled by the first surveyee.
  * `survey_u1.pdf` : the survey form filled by the second surveyee.
  * `survey_u0.csv` : the results for the first surveyee in csv format.
  * `survey_u1.csv` : the results for the second surveyee in csv format.
  * `answer_0.csv`  : the ground truth for the first surveyee, indicating which model the images in each page of `survey_u0.pdf` belong to.
  * `answer_1.csv`  : the ground truth for the second surveyee, indicating which model the images in each page of `survey_u1.pdf` belong to.
  
* `preprocessing.py` is a script which includes all the commands we used to go from the original data set downloaded from [Barbu's website](http://adrianbarburesearch.blogspot.com/p/renoir-dataset.html) to `dataset.parquet`. Make sure to modify the string `ENTER_ALIGNED_PICTURES_FOLDER_HERE` with the full path of the aligned picture folders on your machine if you decide to run the preprocessing step.

This completes the file description.
