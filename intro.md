<details>
<summary><b>How to use this application?</b></summary>

This application require one data input from the user. To do so, click on the Browse (in left side-bar panel) and upload the csv data input file.

Note that this application can read only csv file (comma delimited file), so if you don't have csv input data file, first convert your data in csv format and then proceed. Make sure you have top row as variable names.

Once csv file is uploaded successfully, Data Partition & Model Development sections will appear below the file upload option to split data and tune model parameters.

**Data Partition**:

    1. Select target variable
    2. Select train/test partition (by default it is 70 (Train):30 (Test))
    3. Click split button and it will generate train and test set. 
 
**Model Development**

    1. After selecting the classification algorithm a list of relevant hyperparameters with default values appear.
    2. Tune parameters
    3. Select evaluation metrics
    4. Click classify button

After performing above steps it will display performance metrics on test dataset.

**Note:** Everytime you made changes to data partition or model development section do not forget to click split and classify button respectively.

</details>
