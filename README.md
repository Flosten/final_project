# Learning-based alarm systems for hypoglycaemia and hyperglycaemia prevention
## Project Description
This project focuses on developing learning-based alarm system for Type 1 Diabetes patients using the UVA/Padova simulator dataset. Accurate glucose prediction and reliable alarm generation is critical for enabling proactive diabetes management and preventing dangerous events such as hypoglycemia and hyperglycemia.

## Key Features
**Multi-Channel LSTM:** separate modeling of CGM, insulin, and meal inputs.

**Dual-Input Mechanism** both the raw sequence data and the physio layer processed data are used as the input.

**Physiological Modeling Layer:** combines raw and processed features to capture both data-driven and mechanistic patterns.

**Customized Clinical Loss:** emphasizes hypoglycemia and hyperglycemia prediction for improved clinical relevance.

**Attention + SHAP:** ensures interpretability by identifying feature contributions at both individual and group levels.

## Results
**Case Study:** the proposed model significantly outperforms baseline LSTM models, with ablation studies confirming the contribution of each module.

**Group Study:** the proposed model consistently achieves better performance than baselines, demonstrating robustness even under limited data scenarios.

## File Overview
The project is organized into the following folders and files:
- **src** Contains the functions for running this project.
  - `ablation_study.py`: Contains functions used for a series of ablation studies.
  - `Evaluation.py`: Contains functions used for evaluating the model performance
  - `Modeling.py`: Contains functions used for building, training, and evaluating the baseline model and proposed model.
  - `Preprocessing.py`: Contains functions used for preprocessing training and testing datasets for model training.
  - `Visualising.py`: Contains functions used for visualising the results.

- **Datasets**: Stores the datasets that used in this project.
  - `Testing_Data`: Stores the test dataset.
    - `ts-dt1`: Stores the dataset that used to evaluate the model performance.
    - `ts-dtI`: Stores the dataset that used to learn the insulin bolus response.
    - `ts-dtM`: Stores the dataset that used to learn the carbohydrates intake response.
  - `Training_Data`: Stores the dataset that used to training the model
    - `train_4days`: Stores the dataset for 100 patients over a period of four days.
    - `train_30days`: Stores the dataset of patient 98 over a period of thirty days.

- **env/** Includes the descriptions of the environment required to run the project
  - `environment.yml`: Defines the environment and its version
  - `requirements.txt`: Lists python packages that required to run the code 

- **figures**: Stores the plots generated during the project, including images from EDA, model training and hyperparameters tuning process as well as the final results.

- **results**: Stores the prediction results generated during the project, including baseline model, proposed model and ablation study in both case study and group study.

- **models**: Stores the pre-trained model for patient 98.

- **main.py**: The main script that contains the complete workflow code for the sentiment analysis task.

## Required Packages
- `numpy`
- `pandas`
- `torch`
- `scikit-learn`
- `tqdm`
- `matplotlib`
- `scipy`
- `captum`


## How to Run the Code
1. **Open the terminal and use cd to navigate to the root directory**
2. **Create the Conda Environment:**
   ```bash
   sudo conda env create -f env/environment.yml
3. **Check the Environment:**
   ```bash
   conda info --envs
4. **Activate the Environment:**
   ```bash
   conda activate final-project-env
5. **Install the required packages:**
   ```bash
   pip install -r env/requirements.txt
6. **Run the main script:**
   ```bash
   python main.py

## Note:
Since the dataset is quite large and the data resources are relatively important, the `Dataset` folder is left empty.  
To run the experiments, please prepare and add the required datasets yourself according to the folder structure described in the README.