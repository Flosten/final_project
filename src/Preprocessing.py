"""
This module provides functions to preprocess data for Baseline models, Proposed models,
and to perform exploratory data analysis (EDA) on the dataset.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import src.Visualising as visl


def get_data(data_type, folder_path, p_num):
    """
    Load data from .mat files based on the specified data type and participant number.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path (str): Path to the folder containing the data files.
    p_num (int): Participant number, should be between 1 and 100.

    Returns:
    dict: Loaded data from the .mat file.
    """
    # Ensure p_num is a string with leading zeros
    if 0 < p_num < 10:
        p_num = f"00{p_num}"
    elif 10 <= p_num < 100:
        p_num = f"0{p_num}"
    else:
        p_num = str(p_num)

    # get the file path based on data_type and folder_path
    if data_type == "train":
        file_path = f"./Dataset/Training_Data/{folder_path}/s#adult#{p_num}.mat"
    elif data_type == "test":
        file_path = f"./Dataset/Testing_Data/{folder_path}/s#adult#{p_num}.mat"
    else:
        raise ValueError("data_type must be either 'train' or 'test'")

    # load the data from the .mat file
    data = loadmat(file_path)

    return data


def mat_deconstruct(data):
    """
    Recursively deconstruct a MATLAB structured array into a Python dictionary.

    Parameters:
    data (any): The data to deconstruct, can be a numpy array, structured array, or other types.

    Returns:
    dict or list: Deconstructed data in a more Python-friendly format.
    """
    if isinstance(data, np.ndarray):
        if data.dtype.names:
            out = {}
            for name in data.dtype.names:
                # get the value for each field in the structured array
                value = data[name]
                try:
                    # if the value is a single-element array, extract the value
                    value = value[0]
                except IndexError:
                    pass
                out[name] = mat_deconstruct(value)
            return out

        elif data.size == 1:
            return mat_deconstruct(data.item())
        else:
            return [mat_deconstruct(el) for el in data]
    elif isinstance(data, (np.generic, np.number)):
        return data.item()
    else:
        return data


def extract_features(data, ts=5):
    """
    Extract features from the data and pretreat it for model training.

    Parameters:
    data (dict): The data loaded from the .mat file.
    ts (int): Time step for resampling, default is 5 minutes.

    Returns:
    tuple: Input data and output data dictionaries.
    """
    # input: CGM, current insulin bolus, current insulin basal, current carb intake
    # get the CGM data
    input_cgm = mat_deconstruct(data["CGM"])
    input_cgm_time = input_cgm["time"][1:]  # Remove the first value
    input_cgm_time = input_cgm_time[::ts]  # Resample to 5 min
    input_cgm_values = input_cgm["signals"]["values"]
    input_cgm_values = input_cgm_values[1:]  # Remove the first value
    input_cgm_values = input_cgm_values[::ts]  # Resample to 5 min

    # get the current carb intake
    input_carb = mat_deconstruct(data["carb_intake"])
    input_carb_time = input_carb["time"][1:]  # Remove the first value
    input_carb_values = input_carb["signals"]["values"]
    input_carb_values = input_carb_values[1:]  # Remove the first value
    # resample the carb intake from 30 sec to 1 min
    input_carb_time = input_carb_time[::2]
    input_carb_values = input_carb_values[::2]  # Resample to 1 min
    # from 1 min to 5 min
    input_carb_time = input_carb_time[::ts]
    input_carb_values = input_carb_values[::ts]  # Resample to 5 min

    # get the current insulin
    input_insulin = mat_deconstruct(data["basalBolusMem"])
    current_basal = None
    current_bolus = None
    # current insulin basal
    if ts == 5:
        current_basal = deal_insulin_basal(input_insulin["basal_reconstructed"])
    elif ts == 1:
        current_basal = deal_insulin_basal_1min(input_insulin["basal_reconstructed"])

    # get the current insulin bolus
    if ts == 5:
        current_bolus = deal_insulin_bolus(input_insulin["bolus_reconstructed"])
    elif ts == 1:
        current_bolus = deal_insulin_bolus_1min(input_insulin["bolus_reconstructed"])

    # output: current G
    output = mat_deconstruct(data["G"])
    output_time = output["time"][1:]
    output_time = output_time[::ts]  # Resample to 5 min
    output_values = output["signals"]["values"]
    output_values = output_values[1:]  # Remove the first value
    output_values = output_values[::ts]  # Resample to 5 min

    # # output: current cgm
    # output_time = input_cgm_time
    # output_values = input_cgm_values

    # Ensure all arrays are of the same length
    min_length = min(
        len(input_cgm_time),
        len(input_cgm_values),
        len(input_carb_time),
        len(input_carb_values),
        len(current_basal),
        len(current_bolus),
        len(output_time),
        len(output_values),
    )
    max_length = max(
        len(input_cgm_time),
        len(input_cgm_values),
        len(input_carb_time),
        len(input_carb_values),
        len(current_basal),
        len(current_bolus),
        len(output_time),
        len(output_values),
    )

    if min_length != max_length:
        raise ValueError("Input arrays must be of the same length")

    else:
        print(f"Input arrays are of length: {min_length}")
        # construct the input data
        input_data = {
            "time": input_cgm_time,
            "cgm": input_cgm_values,
            "carb_intake": input_carb_values,
            "current_basal": current_basal,
            "current_bolus": current_bolus,
        }
        # construct the output data
        output_data = {"time": output_time, "G": output_values}

    return input_data, output_data


def deal_insulin_basal_1min(data):
    """
    Process the insulin basal data for 1-minute intervals.

    Parameters:
    data (list or array): The raw insulin basal data.

    Returns:
    np.ndarray: Processed insulin basal data resampled to 1-minute intervals.
    """
    basal_ary = np.array(data)
    # -> pmol/min
    basal_ary = basal_ary * 100
    basal_ary = basal_ary[576:]  # Remove the first 576 values
    basal_ary = np.append(basal_ary, 0)  # Append a zero at the end

    # resample the basal data from 5min to 1min
    basal_1min = np.repeat(basal_ary, 5)
    return basal_1min


def deal_insulin_basal(data):
    """
    Process the insulin basal data for 5-minute intervals.

    Parameters:
    data (list or array): The raw insulin basal data.

    Returns:
    np.ndarray: Processed insulin basal data resampled to 5-minute intervals.
    """
    basal_ary = np.array(data)
    # -> pmol/min
    basal_ary = basal_ary * 100
    basal_ary = basal_ary[576:]  # Remove the first 576 values
    basal_ary = np.append(basal_ary, 0)  # Append a zero at the end

    return basal_ary


def deal_insulin_bolus_1min(data):
    """
    Process the insulin bolus data for 1-minute intervals.

    Parameters:
    data (list or array): The raw insulin bolus data.

    Returns:
    np.ndarray: Processed insulin bolus data resampled to 1-minute intervals.
    """
    bolus_ary = np.array(data)
    # -> pmol
    bolus_ary = bolus_ary * 6000
    bolus_ary = bolus_ary[576:]  # Remove the first 576 values
    bolus_ary = np.append(bolus_ary, 0)  # Append a zero at the end

    # resample the bolus data from 5min to 1min
    bolus_len = len(bolus_ary)
    bolus_1min = np.zeros((bolus_len, 5))
    bolus_1min[:, 0] = bolus_ary
    bolus_1min = bolus_1min.flatten()

    return bolus_1min


def deal_insulin_bolus(data):
    """
    Process the insulin bolus data for 5-minute intervals.

    Parameters:
    data (list or array): The raw insulin bolus data.

    Returns:
    np.ndarray: Processed insulin bolus data resampled to 5-minute intervals.
    """
    bolus_ary = np.array(data)
    # -> pmol
    bolus_ary = bolus_ary * (6000 / 5)  # Convert to pmol/min
    bolus_ary = bolus_ary[576:]  # Remove the first 576 values
    bolus_ary = np.append(bolus_ary, 0)  # Append a zero at the end

    return bolus_ary


def data_standardization(data):
    """
    Standardize the input data using StandardScaler.

    Parameters:
    data (list or array): The input data to be standardized.

    Returns:
    tuple: Standardized data and the fitted StandardScaler object.
    """
    # input data shape (sampels, )
    scalar = StandardScaler()
    data = np.array(data)
    data = data.reshape(-1, 1)  # Reshape to 2D array for StandardScaler
    data_scaled = scalar.fit_transform(data)
    data_scaled = data_scaled.flatten()  # Flatten back to 1D array
    return data_scaled, scalar


def data_inverse_standardization(data, scaler):
    """
    Inverse the standardization of the input data using the fitted StandardScaler.

    Parameters:
    data (list or array): The standardized input data to be inverse transformed.
    scaler (StandardScaler): The fitted StandardScaler object used for standardization.

    Returns:
    np.ndarray: Inverse transformed data.
    """
    # input data shape (samples, )
    data = np.array(data)
    data = data.reshape(-1, 1)  # Reshape to 2D array for inverse transformation
    data_inverse = scaler.inverse_transform(data)
    data_inverse = data_inverse.flatten()  # Flatten back to 1D array
    return data_inverse


def data_inverse_standardization_paper(data, scaler):
    """
    Inverse the standardization of the input data using the fitted StandardScaler
    same as the paper.

    Parameters:
    data (list or array): The standardized input data to be inverse transformed.
    scaler (list): The fitted StandardScaler object used for standardization,

    Returns:
    np.ndarray: Inverse transformed data.
    """
    # input data shape (samples, )
    mean = scaler[0][0]
    std = scaler[0][1]
    data = np.array(data)
    data = data.reshape(-1, 1)  # Reshape to 2D array for inverse transformation
    data_inverse = data * std + mean  # Inverse standardization
    data_inverse = data_inverse.flatten()  # Flatten back to 1D array

    return data_inverse


# ----------------Exploratory Data Analysis (EDA) functions-------------------
def data_eda(
    data_type_1,
    data_folder_1,
    data_type_2,
    data_folder_2,  # insulin
    data_folder_3,  # meal
    p_num,
):
    """
    Perform exploratory data analysis (EDA) on the given data.

    Parameters:
    data_type_1 (str): Type of the first data, either 'train' or 'test'.
    data_folder_1 (str): Folder path for the data (CGM, carb intake, insulin).
    data_type_2 (str): Type of the second data, either 'train' or 'test'.
    data_folder_2 (str): Folder path for the second data (insulin).
    data_folder_3 (str): Folder path for the third data (meal).
    p_num (int): Participant number, should be between 1 and 100.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Base directory: {base_dir}")
    figures_dir = os.path.join(base_dir, "figures")

    # plot the input data and output data
    visualise_data = get_data(
        data_type=data_type_1,
        folder_path=data_folder_1,
        p_num=p_num,
    )
    input_vis_data, output_vis_data = extract_features(visualise_data)

    vis_cgm = input_vis_data["cgm"][-288:]
    vis_carb_intake = input_vis_data["carb_intake"][-288:]
    vis_insulin_basal = input_vis_data["current_basal"][-288:]
    vis_insulin_bolus = input_vis_data["current_bolus"][-288:]
    vis_G = output_vis_data["G"][-288:]

    # plot and save the data
    cgm_fig, _ = visl.visualise_data(
        data=vis_cgm,
        ticks_per_day=6,
        y_label="CGM Glucose (mg/dL)",
        time_steps=5,
    )
    cgm_fig_name = f"CGM_data_patient_{p_num}.png"
    cgm_fig_path = os.path.join(figures_dir, cgm_fig_name)
    cgm_fig.savefig(cgm_fig_path)
    plt.close(cgm_fig)

    carb_fig, _ = visl.visualise_data(
        data=vis_carb_intake,
        ticks_per_day=6,
        y_label="Carbohydrate Intake (g)",
        time_steps=5,
    )
    carb_fig_name = f"Carb_Intake_data_patient_{p_num}.png"
    carb_fig_path = os.path.join(figures_dir, carb_fig_name)
    carb_fig.savefig(carb_fig_path)
    plt.close(carb_fig)

    basal_fig, _ = visl.visualise_data(
        data=vis_insulin_basal,
        ticks_per_day=6,
        y_label="Insulin Basal (pmol/min)",
        time_steps=5,
    )
    basal_fig_name = f"Insulin_Basal_data_patient_{p_num}.png"
    basal_fig_path = os.path.join(figures_dir, basal_fig_name)
    basal_fig.savefig(basal_fig_path)
    plt.close(basal_fig)

    bolus_fig, _ = visl.visualise_data(
        data=vis_insulin_bolus,
        ticks_per_day=6,
        y_label="Insulin Bolus (pmol)",
        time_steps=5,
    )
    bolus_fig_name = f"Insulin_Bolus_data_patient_{p_num}.png"
    bolus_fig_path = os.path.join(figures_dir, bolus_fig_name)
    bolus_fig.savefig(bolus_fig_path)
    plt.close(bolus_fig)

    g_fig, _ = visl.visualise_data(
        data=vis_G,
        ticks_per_day=6,
        y_label="Reference Glucose (mg/dL)",
        time_steps=5,
    )
    g_fig_name = f"G_data_patient_{p_num}.png"
    g_fig_path = os.path.join(figures_dir, g_fig_name)
    g_fig.savefig(g_fig_path)
    plt.close(g_fig)

    # plot the data for physiological signals
    data_insulin = get_data(
        data_type=data_type_2,
        folder_path=data_folder_2,
        p_num=p_num,
    )
    data_meal = get_data(
        data_type=data_type_2,
        folder_path=data_folder_3,
        p_num=p_num,
    )
    input_data_insulin, output_data_insulin = extract_features(data_insulin)
    input_data_meal, output_data_meal = extract_features(data_meal)

    # get the bolus, carb and reference G data
    bolus_insulin = input_data_insulin["current_bolus"]
    bolus_insulin_g = output_data_insulin["G"]
    carb_intake = input_data_meal["carb_intake"]
    carb_intake_g = output_data_meal["G"]

    # plot and save the figure
    bolus_insulin_fig, _ = visl.visualise_insulin_meal_response(
        data_1=bolus_insulin_g,
        data_2=bolus_insulin,
        legend_1="Reference Glucose (mg/dL)",
        legend_2="Insulin Bolus (pmol)",
        ticks_per_day=6,
        time_steps=5,
    )
    bolus_insulin_fig_name = f"Insulin_Bolus_response_patient_{p_num}.png"
    bolus_insulin_fig_path = os.path.join(figures_dir, bolus_insulin_fig_name)
    bolus_insulin_fig.savefig(bolus_insulin_fig_path)
    plt.close(bolus_insulin_fig)

    carb_intake_fig, _ = visl.visualise_insulin_meal_response(
        data_1=carb_intake_g,
        data_2=carb_intake,
        legend_1="Reference Glucose (mg/dL)",
        legend_2="Carbohydrate Intake (g)",
        ticks_per_day=6,
        time_steps=5,
    )
    carb_intake_fig_name = f"Carb_Intake_response_patient_{p_num}.png"
    carb_intake_fig_path = os.path.join(figures_dir, carb_intake_fig_name)
    carb_intake_fig.savefig(carb_intake_fig_path)
    plt.close(carb_intake_fig)


# ---------------- preprocessing functions for baseline model -------------------
def data_preprocessing_baseline(
    data_type,
    folder_path_4days,
    folder_path_30days,
    p_num,
    seq_len,
    ph,
    interval,
    scaler=None,
):
    """
    Preprocess the data for the baseline model.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path_4days (str): Folder path for the 4 days data.
    folder_path_30days (str): Folder path for the 30 days data.
    p_num (int): Participant number, between 1 and 100.
    seq_len (int): Sequence length for the model.
    ph (int): Prediction horizon.
    interval (int): Time interval in minutes for the data.
    scaler (object, optional): Predefined scaler for standardization. Defaults to None.

    Returns:
    tuple: Train and validation dataloaders, input scalers, and output scaler.
    """
    # transfer sequence length to the number of units
    seq_len = int(seq_len / interval)  # Convert sequence length to the number of units
    print(f"Sequence length in units: {seq_len}")
    # ensure ph is an integer
    ph = int(ph / interval)  # Convert ph to the number of units

    # get the data
    data_4days = get_data(data_type, folder_path_4days, p_num)
    data_30days = get_data(data_type, folder_path_30days, p_num)

    # extract features
    input_data_4days, output_data_4days = extract_features(data_4days, ts=interval)
    input_data_30days, output_data_30days = extract_features(data_30days, ts=interval)

    # split the data into train and validation sets
    train_days = 20
    val_days = 10
    input_train_30days, input_val_30days = split_train_val(
        input_data_30days, train_days, val_days, interval=interval
    )
    output_train_30days, output_val_30days = split_train_val(
        output_data_30days, train_days, val_days, interval=interval
    )

    # standardize the data for both 4 days and 30 days datasets
    input_data_4days_scaled, input_data_30days_scaled, input_scalers = (
        standardize_2_datasets(input_data_4days, input_train_30days)
    )
    output_data_4days_scaled, output_data_30days_scaled, output_scaler = (
        standardize_2_datasets(output_data_4days, output_train_30days)
    )

    # standardize the validation data
    input_val_scaled = standardize_val_data(input_val_30days, input_scalers)
    output_val_scaled = standardize_val_data(output_val_30days, output_scaler)

    # construct dataset
    # trainset
    # 4 days
    x_4days, y_4days = select_build_sequence_data(
        input_data_4days_scaled, output_data_4days_scaled, seq_len, ph
    )
    # 30days
    x_30days, y_30days = select_build_sequence_data(
        input_data_30days_scaled, output_data_30days_scaled, seq_len, ph
    )

    x_train = np.concatenate([x_4days, x_30days], axis=0)
    y_train = np.concatenate([y_4days, y_30days], axis=0)

    # valset
    x_val, y_val = select_build_sequence_data(
        input_val_scaled, output_val_scaled, seq_len, ph
    )

    # create dataloaders
    train_loader, val_loader = create_dataloaders(x_train, y_train, x_val, y_val)

    return train_loader, val_loader, input_scalers, output_scaler


def data_preprocessing_baseline_for_99(
    data_type,
    folder_path_4days,
    p_num,  # Participant number 0 ~ 100
    seq_len,
    ph,
    interval,
    scaler=None,
):
    """
    Preprocess the data for the baseline model for 99 participants.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path_4days (str): Folder path for the 4 days data.
    p_num (int): Participant number, between 1 and 100.
    seq_len (int): Sequence length for the model.
    ph (int): Prediction horizon.
    interval (int): Time interval in minutes for the data.
    scaler (object, optional): Predefined scaler for standardization. Defaults to None.

    Returns:
    tuple: Train and validation dataloaders, input scalers, and output scaler.
    """
    # transfer sequence length to the number of units
    seq_len = int(seq_len / interval)  # Convert sequence length to the number of units
    # print(f"Sequence length in units: {seq_len}")
    # ensure ph is an integer
    ph = int(ph / interval)  # Convert ph to the number of units

    # get the data
    data_4days = get_data(data_type, folder_path_4days, p_num)

    # extract features
    input_data_4days, output_data_4days = extract_features(data_4days, ts=interval)

    # split the data into train and validation sets
    train_days = 3
    val_days = 1
    input_train_4days, input_val_4days = split_train_val(
        input_data_4days, train_days, val_days, interval=interval
    )
    output_train_4days, output_val_4days = split_train_val(
        output_data_4days, train_days, val_days, interval=interval
    )

    # standardize the data for 4 days datasets
    input_data_4days_scaled, input_scalers = standardize_dataset(input_train_4days)
    output_data_4days_scaled, output_scaler = standardize_dataset(output_train_4days)

    # standardize the validation data
    input_val_scaled = standardize_val_data(input_val_4days, input_scalers)
    output_val_scaled = standardize_val_data(output_val_4days, output_scaler)

    # construct dataset
    # trainset
    # 4 days
    x_4days, y_4days = select_build_sequence_data(
        input_data_4days_scaled, output_data_4days_scaled, seq_len, ph
    )

    x_train = x_4days
    y_train = y_4days

    # valset
    x_val, y_val = select_build_sequence_data(
        input_val_scaled, output_val_scaled, seq_len, ph
    )

    # create dataloaders
    train_loader, val_loader = create_dataloaders(x_train, y_train, x_val, y_val)

    return train_loader, val_loader, input_scalers, output_scaler


def test_data_preprocessing_baseline(
    data_type, folder_path_1, p_num, seq_len, ph, interval, input_scalers, output_scaler
):
    """
    Preprocess the test data for the baseline model.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path_1 (str): Folder path for the test data.
    p_num (int): Participant number, between 1 and 100.
    seq_len (int): Sequence length for the model.
    ph (int): Prediction horizon.
    interval (int): Time interval in minutes for the data.
    input_scalers (dict): Input scalers for standardization.
    output_scaler (object): Output scaler for standardization.

    Returns:
    DataLoader: Test dataloader with preprocessed data.
    """
    # transfer sequence length to the number of units
    seq_len = int(seq_len / interval)  # Convert sequence length to the number of units
    # ensure ph is an integer
    ph = int(ph / interval)  # Convert ph to the number of units

    # get the data
    test_data_1 = get_data(data_type, folder_path_1, p_num)

    # extract features
    input_data_1, output_data_1 = extract_features(test_data_1, ts=interval)

    # standardize the data
    input_data_1_scaled = standardize_val_data(input_data_1, input_scalers)
    output_data_1_scaled = standardize_val_data(output_data_1, output_scaler)

    # construct dataset
    x_test, y_test = select_build_sequence_data(
        input_data_1_scaled, output_data_1_scaled, seq_len, ph
    )

    # create dataloader
    test_loader = create_test_dataloader(x_test, y_test)

    return test_loader


def test_data_preprocessing_baseline_for_99(
    data_type, folder_path_1, p_num, seq_len, ph, interval, input_scalers, output_scaler
):
    """
    Preprocess the test data for the baseline model for 99 participants.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path_1 (str): Folder path for the test data.
    p_num (int): Participant number, between 1 and 100.
    seq_len (int): Sequence length for the model.
    ph (int): Prediction horizon.
    interval (int): Time interval in minutes for the data.
    input_scalers (dict): Input scalers for standardization.
    output_scaler (object): Output scaler for standardization.

    Returns:
    DataLoader: Test dataloader with preprocessed data.
    """
    # transfer sequence length to the number of units
    seq_len = int(seq_len / interval)  # Convert sequence length to the number of units
    # ensure ph is an integer
    ph = int(ph / interval)  # Convert ph to the number of units

    # get the data
    test_data_1 = get_data(data_type, folder_path_1, p_num)

    # extract features
    input_data_1, output_data_1 = extract_features(test_data_1, ts=interval)

    # standardize the data
    input_data_1_scaled = standardize_val_data(input_data_1, input_scalers)
    output_data_1_scaled = standardize_val_data(output_data_1, output_scaler)

    # construct dataset
    x_test, y_test = select_build_sequence_data(
        input_data_1_scaled, output_data_1_scaled, seq_len, ph
    )

    # create dataloader
    test_loader = create_test_dataloader(x_test, y_test)

    return test_loader


def standardize_paper(data, scaler):
    """
    Standardize the input data using the provided scaler by the paper.

    Parameters:
    data (list or array): The input data to be standardized.
    scaler (list): The fitted StandardScaler object used for standardization,
                   contains mean and standard deviation from the paper.

    Returns:
    np.ndarray: Standardized data.
    """
    mean = scaler[0][0]
    std = scaler[0][1]
    data = np.array(data)
    data = (data - mean) / std  # Standardize the data
    return data


def test_data_preprocessing_baseline_paper(
    data_type, folder_path_1, p_num, seq_len, ph, interval, input_scalers, output_scaler
):
    """
    Preprocess the test data for the baseline model using the paper's standardization method.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path_1 (str): Folder path for the test data.
    p_num (int): Participant number, between 1 and 100.
    seq_len (int): Sequence length for the model.
    ph (int): Prediction horizon.
    interval (int): Time interval in minutes for the data.
    input_scalers (dict): Input scalers for standardization.
    output_scaler (list): Output scaler for standardization, contains mean and std from the paper.

    Returns:
    DataLoader: Test dataloader with preprocessed data.
    """
    # transfer sequence length to the number of units
    seq_len = int(seq_len / interval)  # Convert sequence length to the number of units
    print(f"Sequence length in units: {seq_len}")
    # ensure ph is an integer
    ph = int(ph / interval)  # Convert ph to the number of units
    print(f"Ph in units: {ph}")

    # get the data
    test_data_1 = get_data(data_type, folder_path_1, p_num)

    # extract features
    input_data_1, output_data_1 = extract_features(test_data_1, ts=interval)

    # standardize the data
    cgm_scaled = standardize_paper(input_data_1["cgm"], input_scalers["CGM"])
    carb_scaled = standardize_paper(input_data_1["carb_intake"], input_scalers["meal"])
    basal_scaled = standardize_paper(
        input_data_1["current_basal"], input_scalers["basal"]
    )
    bolus_scaled = standardize_paper(
        input_data_1["current_bolus"], input_scalers["bolus"]
    )

    output_g = standardize_paper(output_data_1["G"], output_scaler["G"])

    # construct input data
    input_data_1_scaled = {
        "cgm": cgm_scaled,
        "current_bolus": bolus_scaled,
        "current_basal": basal_scaled,
        "carb_intake": carb_scaled,
    }

    # construct output data
    output_data_1_scaled = {"G": output_g}

    # construct dataset
    x_test, y_test = select_build_sequence_data(
        input_data_1_scaled, output_data_1_scaled, seq_len, ph
    )

    # create dataloader
    test_loader = create_test_dataloader(x_test, y_test)

    return test_loader


def standardize_2_datasets(data_4days, data_30days):
    """
    This the function used to standardize two datasets (4 days and 30 days) using StandardScaler.

    Parameters:
    data_4days (dict): Dictionary containing the 4 days data.
    data_30days (dict): Dictionary containing the 30 days data.

    Returns:
    tuple: Scaled 4 days data, scaled 30 days data, and the scalers used for each key.
    """
    scalers = {}
    data_4days_scaled = {}
    data_30days_scaled = {}

    for key in data_4days.keys():
        data_4d = np.array(data_4days[key]).reshape(
            -1, 1
        )  # Reshape to 2D array for StandardScaler
        data_30d = np.array(data_30days[key]).reshape(
            -1, 1
        )  # Reshape to 2D array for StandardScaler
        all_data = np.concatenate((data_4d, data_30d), axis=0)

        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        scaler.fit(all_data)
        scalers[key] = scaler

        data_4days_scaled[key] = scaler.transform(
            data_4d
        ).flatten()  # Flatten back to 1D array
        data_30days_scaled[key] = scaler.transform(
            data_30d
        ).flatten()  # Flatten back to 1D array

    return data_4days_scaled, data_30days_scaled, scalers


def standardize_dataset(data_4days):
    """
    Standardize the input data using StandardScaler for a single dataset.

    Parameters:
    data_4days (dict): Dictionary containing the 4/30 days data.

    Returns:
    tuple: Scaled 4/30 days data and the scaler used for each key.
    """
    scalers = {}
    data_4days_scaled = {}

    for key in data_4days.keys():
        data_4d = np.array(data_4days[key]).reshape(
            -1, 1
        )  # Reshape to 2D array for StandardScaler

        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        scaler.fit(data_4d)
        scalers[key] = scaler

        data_4days_scaled[key] = scaler.transform(
            data_4d
        ).flatten()  # Flatten back to 1D array

    return data_4days_scaled, scalers


def standardize_val_data(data, scalers):
    """
    Standardize the validation data using the fitted scalers from the training data.

    Parameters:
    data (dict): Dictionary containing the validation data.
    scalers (dict): Dictionary containing the fitted scalers for each key.

    Returns:
    dict: Scaled validation data.
    """
    val_data_scaled = {}

    for key, value in data.items():
        value = np.array(value).reshape(-1, 1)
        scaler = scalers[key]
        value_scaled = scaler.transform(value)
        val_data_scaled[key] = value_scaled.flatten()

    return val_data_scaled


def select_build_sequence_data(input_data, output_data, seq_len, ph):
    """
    This function is used to select and build sequence data for the model.

    Parameters:
    input_data (dict): Dictionary containing the input data.
    output_data (dict): Dictionary containing the output data.
    seq_len (int): Sequence length for the model.
    ph (int): Prediction horizon.

    Returns:
    tuple: Input and output data in sequence format.
    """
    # input data
    # input cgm
    input_select_cgm = select_data(input_data["cgm"], seq_len, ph, "cgm")
    input_cgm_seq = build_sequence_data(input_select_cgm, seq_len)
    # input carb intake
    input_select_carb = select_data(input_data["carb_intake"], seq_len, ph, "input")
    input_carb_seq = build_sequence_data(input_select_carb, seq_len)
    # input current basal
    input_select_basal = select_data(input_data["current_basal"], seq_len, ph, "input")
    input_basal_seq = build_sequence_data(input_select_basal, seq_len)
    # input current bolus
    input_select_bolus = select_data(input_data["current_bolus"], seq_len, ph, "input")
    input_bolus_seq = build_sequence_data(input_select_bolus, seq_len)

    # output G
    output_g = select_data(output_data["G"], seq_len, ph, "output")

    print(f"Input data shape: {input_cgm_seq.shape}")
    print(f"Output data shape: {output_g.shape}")

    # # chech the output is right
    # print(f"Input CGM time 0 value: {input_select_cgm[0]}")
    # print(f"Output G time 0 value: {output_g[0]}")
    # print(f"Input CGM time 60 value: {input_select_cgm[ph]}")

    # construct x and y
    x = np.stack(
        (
            input_cgm_seq,
            input_carb_seq,
            input_basal_seq,
            input_bolus_seq,
        ),
        axis=-1,
    )

    y = output_g.reshape(-1, 1)

    return x, y


def select_build_sequence_data_paper(input_data, output_data, seq_len, ph):
    """
    This function is used to select and build sequence data for the model
    same as the paper.

    Parameters:
    input_data (dict): Dictionary containing the input data.
    output_data (dict): Dictionary containing the output data.
    seq_len (int): Sequence length for the model.
    ph (int): Prediction horizon.

    Returns:
    tuple: Input and output data in sequence format.
    """
    # input data
    # input cgm
    input_select_cgm = select_data(input_data["cgm"], seq_len, ph, "cgm")
    input_cgm_seq = build_sequence_data(input_select_cgm, seq_len)
    # input carb intake
    input_select_carb = select_data(input_data["carb_intake"], seq_len, ph, "input")
    input_carb_seq = build_sequence_data(input_select_carb, seq_len)
    # input current basal
    input_select_basal = select_data(input_data["current_basal"], seq_len, ph, "input")
    input_basal_seq = build_sequence_data(input_select_basal, seq_len)
    # input current bolus
    input_select_bolus = select_data(input_data["current_bolus"], seq_len, ph, "input")
    input_bolus_seq = build_sequence_data(input_select_bolus, seq_len)

    # output G
    output_g = select_data(output_data["G"], seq_len, ph, "output")

    print(f"Input data shape: {input_cgm_seq.shape}")
    print(f"Output data shape: {output_g.shape}")

    # # chech the output is right
    # print(f"Input CGM time 0 value: {input_select_cgm[0]}")
    # print(f"Output G time 0 value: {output_g[0]}")
    # print(f"Input CGM time 60 value: {input_select_cgm[ph]}")

    # construct x and y
    x = np.stack(
        (
            input_cgm_seq,
            input_carb_seq,
            input_basal_seq,
            input_bolus_seq,
        ),
        axis=-1,
    )

    y = output_g.reshape(-1, 1)

    return x, y


def select_data(data, seq_len, ph, data_type):
    """
    This function is used to select the data based on the data type and sequence length.

    Parameters:
    data (list or array): The input data to be selected.
    seq_len (int): Sequence length for the model.
    ph (int): Prediction horizon.
    data_type (str): Type of data, either 'cgm', 'output', or 'input'.

    Returns:
    np.ndarray: Selected data based on the data type and sequence length.
    """
    if data_type == "cgm":
        selected_data = np.array(data)
        selected_data = selected_data[:-ph]  # Remove the last ph values

    elif data_type == "output":
        selected_data = np.array(data)
        selected_data = selected_data[seq_len + ph - 1 :]

    else:
        selected_data = np.array(data)
        selected_data = selected_data[ph:]  # Remove the first ph values

    return selected_data


def build_sequence_data(data, seq_len):
    """
    This function is used to build the sequence data based on the sequence length.

    Parameters:
    data (list or array): The input data to be built into sequences.
    seq_len (int): Sequence length for the model.

    Returns:
    np.ndarray: Sequence data with shape (num_samples, seq_len).
    """
    data = np.array(data).reshape(-1)
    num_samples = len(data) - seq_len + 1

    data_seq = np.zeros((num_samples, seq_len))
    for i in range(num_samples):
        data_seq[i] = data[i : i + seq_len]

    return data_seq


def create_dataloaders(x_train, y_train, x_val, y_val, batch_size=128):
    """
    Create PyTorch DataLoaders for training and validation datasets.

    Parameters:
    x_train (np.ndarray): Training input data.
    y_train (np.ndarray): Training output data.
    x_val (np.ndarray): Validation input data.
    y_val (np.ndarray): Validation output data.
    batch_size (int): Batch size for the DataLoader.

    Returns:
    tuple: Training and validation DataLoaders.
    """
    # convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # create TensorDataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_test_dataloader(x_test, y_test, batch_size=128):
    """
    Create a PyTorch DataLoader for the test dataset.

    Parameters:
    x_test (np.ndarray): Test input data.
    y_test (np.ndarray): Test output data.
    batch_size (int): Batch size for the DataLoader.

    Returns:
    DataLoader: Test DataLoader.
    """
    # convert numpy arrays to PyTorch tensors
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # create TensorDataset
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def split_train_val(data, train_days, val_days, interval=5):
    """
    Split the data into training and validation sets based on the number of days.

    Parameters:
    data (dict): Dictionary containing the data to be split.
    train_days (int): Number of days for the training set.
    val_days (int): Number of days for the validation set.
    interval (int): Time interval in minutes for the data.

    Returns:
    tuple: Dictionaries containing the training and validation data.
    """
    # calculate the number of samples for train and validation sets
    num_train_samples = train_days * (
        24 * 60 // interval
    )  # 24 hours, 60 minutes per hour, divided by interval
    num_val_samples = val_days * (
        24 * 60 // interval
    )  # 24 hours, 60 minutes per hour, divided by interval

    split_dict_train = {}
    split_dict_val = {}

    for key, value in data.items():
        value = np.array(value)

        assert (
            len(value) >= num_train_samples + num_val_samples
        ), f"Data for {key} is not enough for {train_days} days of training and {val_days} days of validation."
        # split the data into train and validation sets
        train_data = value[:num_train_samples]
        val_data = value[num_train_samples : num_train_samples + num_val_samples]
        split_dict_train[key] = train_data
        split_dict_val[key] = val_data

    return split_dict_train, split_dict_val


# ---------------- preprocessing functions for proposed model -------------------
class MultiInputDataset(torch.utils.data.Dataset):
    """
    Custom dataset for the proposed model that handles multiple input features.

    Attributes:
    x_dict (dict): Dictionary containing input features.
    y (np.ndarray): Output labels.
    y_ori (np.ndarray): Original output labels for the proposed model.
    """

    def __init__(self, x_dict, y, y_ori):
        """
        Initialize the dataset with input features and output labels.

        Parameters:
        x_dict (dict): Dictionary containing input features.
        y (np.ndarray): Output labels.
        y_ori (np.ndarray): Original output labels for the proposed model.
        """
        self.x_dict = x_dict
        self.y = y
        self.y_ori = y_ori

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
        int: Number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Parameters:
        idx (int): Index of the item to retrieve.

        Returns:
        tuple: A tuple containing the input features, output label, and original output label.
        """
        x_item = {key: self.x_dict[key][idx] for key in self.x_dict}
        y_item = self.y[idx]
        ori_y_item = self.y_ori[idx]
        return x_item, y_item, ori_y_item


def data_preprocessing_proposed(
    data_type,
    folder_path_4days,
    folder_path_30days,
    p_num,
    seq_len_cgm,
    seq_len_carb,
    seq_len_basal,
    seq_len_bolus,
    ph,
    interval,
    batch_size=128,
    scaler=None,
):
    """
    Preprocess the data for the proposed model.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path_4days (str): Folder path for the 4 days data.
    folder_path_30days (str): Folder path for the 30 days data.
    p_num (int): Participant number, between 1 and 100.
    seq_len_cgm (int): Sequence length for CGM data.
    seq_len_carb (int): Sequence length for carbohydrate intake data.
    seq_len_basal (int): Sequence length for basal insulin data.
    seq_len_bolus (int): Sequence length for bolus insulin data.
    ph (int): Prediction horizon.
    interval (int): Time interval in minutes for the data.
    batch_size (int): Batch size for the DataLoader.
    scaler (object, optional): Predefined scaler for standardization. Defaults to None.

    Returns:
    tuple: Train and validation dataloaders, input scalers, and output scaler.
    """
    # transfer sequence length to the number of units
    seq_len_cgm = int(
        seq_len_cgm / interval
    )  # Convert sequence length to the number of units
    seq_len_carb = int(
        seq_len_carb / interval
    )  # Convert sequence length to the number of units
    seq_len_basal = int(
        seq_len_basal / interval
    )  # Convert sequence length to the number of units
    seq_len_bolus = int(
        seq_len_bolus / interval
    )  # Convert sequence length to the number of units
    # ensure ph is an integer
    ph = int(ph / interval)  # Convert ph to the number of units

    # get the data
    data_4days = get_data(data_type, folder_path_4days, p_num)
    data_30days = get_data(data_type, folder_path_30days, p_num)

    # extract features
    input_data_4days, output_data_4days = extract_features(data_4days, ts=interval)
    input_data_30days, output_data_30days = extract_features(data_30days, ts=interval)

    # split the data into train and validation sets
    train_days = 20
    val_days = 10
    input_train_30days, input_val_30days = split_train_val(
        input_data_30days, train_days, val_days, interval=interval
    )
    output_train_30days, output_val_30days = split_train_val(
        output_data_30days, train_days, val_days, interval=interval
    )

    # standardize the data for both 4 days and 30 days datasets
    input_data_4days_scaled, input_data_30days_scaled, input_scalers = (
        standardize_2_datasets(input_data_4days, input_train_30days)
    )
    output_data_4days_scaled, output_data_30days_scaled, output_scaler = (
        standardize_2_datasets(output_data_4days, output_train_30days)
    )

    # standardize the validation data
    input_val_scaled = standardize_val_data(input_val_30days, input_scalers)
    output_val_scaled = standardize_val_data(output_val_30days, output_scaler)

    # construct dataset
    # trainset
    # 4 days
    x_4days, y_4days = create_sequence_for_proposed(
        input_data_4days_scaled,
        output_data_4days_scaled,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )
    # 30days
    x_30days, y_30days = create_sequence_for_proposed(
        input_data_30days_scaled,
        output_data_30days_scaled,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )

    # we also need the original data for the proposed model
    ori_y_4days = get_ori_output(
        output_data_4days,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )
    ori_y_30days_train = get_ori_output(
        output_train_30days,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )
    ori_y_30days_val = get_ori_output(
        output_val_30days,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )
    print(f"Check original output data values: {ori_y_4days[0:5]}")

    # concatenate the data
    x_train = {
        key: np.concatenate([x_4days[key], x_30days[key]], axis=0) for key in x_4days
    }
    y_train = np.concatenate([y_4days, y_30days], axis=0)
    ori_y_train = np.concatenate([ori_y_4days, ori_y_30days_train], axis=0)

    # valset
    x_val, y_val = create_sequence_for_proposed(
        input_val_scaled,
        output_val_scaled,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )

    # create dataloaders
    train_loader, val_loader = create_dataloaders_proposed(
        x_train, y_train, x_val, y_val, ori_y_train, ori_y_30days_val, batch_size
    )

    return train_loader, val_loader, input_scalers, output_scaler


def data_preprocessing_proposed_for_99(
    data_type,
    folder_path_4days,
    p_num,
    seq_len_cgm,
    seq_len_carb,
    seq_len_basal,
    seq_len_bolus,
    ph,
    interval,
    batch_size=128,
    scaler=None,
):
    """
    Preprocess the data for the proposed model for 99 participants.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path_4days (str): Folder path for the 4 days data.
    p_num (int): Participant number, between 1 and 100.
    seq_len_cgm (int): Sequence length for CGM data.
    seq_len_carb (int): Sequence length for carbohydrate intake data.
    seq_len_basal (int): Sequence length for basal insulin data.
    seq_len_bolus (int): Sequence length for bolus insulin data.
    ph (int): Prediction horizon.
    interval (int): Time interval in minutes for the data.
    batch_size (int): Batch size for the DataLoader.
    scaler (object, optional): Predefined scaler for standardization. Defaults to None.

    Returns:
    tuple: Train and validation dataloaders, input scalers, and output scaler.
    """
    # transfer sequence length to the number of units
    seq_len_cgm = int(
        seq_len_cgm / interval
    )  # Convert sequence length to the number of units
    seq_len_carb = int(
        seq_len_carb / interval
    )  # Convert sequence length to the number of units
    seq_len_basal = int(
        seq_len_basal / interval
    )  # Convert sequence length to the number of units
    seq_len_bolus = int(
        seq_len_bolus / interval
    )  # Convert sequence length to the number of units
    # ensure ph is an integer
    ph = int(ph / interval)  # Convert ph to the number of units

    # get the data
    data_4days = get_data(data_type, folder_path_4days, p_num)

    # extract features
    input_data_4days, output_data_4days = extract_features(data_4days, ts=interval)

    # split the data into train and validation sets
    train_days = 3
    val_days = 1
    input_train_4days, input_val_4days = split_train_val(
        input_data_4days, train_days, val_days, interval=interval
    )
    output_train_4days, output_val_4days = split_train_val(
        output_data_4days, train_days, val_days, interval=interval
    )

    # standardize the data for 4 days dataset
    input_data_4days_scaled, input_scalers = standardize_dataset(input_train_4days)
    output_data_4days_scaled, output_scaler = standardize_dataset(output_train_4days)

    # standardize the validation data
    input_val_scaled = standardize_val_data(input_val_4days, input_scalers)
    output_val_scaled = standardize_val_data(output_val_4days, output_scaler)

    # construct dataset
    # trainset
    # 4 days
    x_4days, y_4days = create_sequence_for_proposed(
        input_data_4days_scaled,
        output_data_4days_scaled,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )

    # we also need the original data for the proposed model
    ori_y_4days = get_ori_output(
        output_train_4days,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )

    # print(f"Check original output data values: {ori_y_4days[0:5]}")

    # concatenate the data
    x_train = {key: np.concatenate([x_4days[key]], axis=0) for key in x_4days}
    y_train = np.concatenate([y_4days], axis=0)
    ori_y_train = np.concatenate([ori_y_4days], axis=0)

    # valset
    x_val, y_val = create_sequence_for_proposed(
        input_val_scaled,
        output_val_scaled,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )
    ori_y_4days_val = get_ori_output(
        output_val_4days,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )

    # create dataloaders
    train_loader, val_loader = create_dataloaders_proposed(
        x_train, y_train, x_val, y_val, ori_y_train, ori_y_4days_val, batch_size
    )

    return train_loader, val_loader, input_scalers, output_scaler


def test_data_preprocessing_proposed(
    data_type,
    folder_path_1,
    p_num,
    seq_len_cgm,
    seq_len_carb,
    seq_len_basal,
    seq_len_bolus,
    ph,
    interval,
    input_scalers,
    output_scaler,
    batch_size=128,
):
    """
    Preprocess the test data for the proposed model.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path_1 (str): Folder path for the test data.
    p_num (int): Participant number, between 1 and 100.
    seq_len_cgm (int): Sequence length for CGM data.
    seq_len_carb (int): Sequence length for carbohydrate intake data.
    seq_len_basal (int): Sequence length for basal insulin data.
    seq_len_bolus (int): Sequence length for bolus insulin data.
    ph (int): Prediction horizon.
    interval (int): Time interval in minutes for the data.
    input_scalers (dict): Input scalers for standardization.
    output_scaler (list): Output scaler for standardization, contains mean and std from the paper.
    batch_size (int): Batch size for the DataLoader.

    Returns:
    DataLoader: Test dataloader with preprocessed data.
    """
    # transfer sequence length to the number of units
    seq_len_cgm = int(
        seq_len_cgm / interval
    )  # Convert sequence length to the number of units
    seq_len_carb = int(
        seq_len_carb / interval
    )  # Convert sequence length to the number of units
    seq_len_basal = int(
        seq_len_basal / interval
    )  # Convert sequence length to the number of units
    seq_len_bolus = int(
        seq_len_bolus / interval
    )  # Convert sequence length to the number of units
    # ensure ph is an integer
    ph = int(ph / interval)  # Convert ph to the number of units

    # get the data
    test_data_1 = get_data(data_type, folder_path_1, p_num)

    # extract features
    input_data_1, output_data_1 = extract_features(test_data_1, ts=interval)

    # standardize the data
    input_data_1_scaled = standardize_val_data(input_data_1, input_scalers)
    output_data_1_scaled = standardize_val_data(output_data_1, output_scaler)

    # construct dataset
    x_test, y_test = create_sequence_for_proposed(
        input_data_1_scaled,
        output_data_1_scaled,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )

    # we also need the original data for the proposed model
    ori_y_test = get_ori_output(
        output_data_1,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )

    # create dataloader
    test_loader = create_test_dataloader_proposed(
        x_test, y_test, ori_y_test, batch_size
    )

    return test_loader


def test_data_preprocessing_proposed_for_99(
    data_type,
    folder_path_1,
    p_num,
    seq_len_cgm,
    seq_len_carb,
    seq_len_basal,
    seq_len_bolus,
    ph,
    interval,
    input_scalers,
    output_scaler,
    batch_size=128,
):
    """
    Preprocess the test data for the proposed model for 99 participants.

    Parameters:
    data_type (str): Type of data to load, either 'train' or 'test'.
    folder_path_1 (str): Folder path for the test data.
    p_num (int): Participant number, between 1 and 100.
    seq_len_cgm (int): Sequence length for CGM data.
    seq_len_carb (int): Sequence length for carbohydrate intake data.
    seq_len_basal (int): Sequence length for basal insulin data.
    seq_len_bolus (int): Sequence length for bolus insulin data.
    ph (int): Prediction horizon.
    interval (int): Time interval in minutes for the data.
    input_scalers (dict): Input scalers for standardization.
    output_scaler (list): Output scaler for standardization, contains mean and std from the paper.
    batch_size (int): Batch size for the DataLoader.

    Returns:
    DataLoader: Test dataloader with preprocessed data.
    """
    # transfer sequence length to the number of units
    seq_len_cgm = int(
        seq_len_cgm / interval
    )  # Convert sequence length to the number of units
    seq_len_carb = int(
        seq_len_carb / interval
    )  # Convert sequence length to the number of units
    seq_len_basal = int(
        seq_len_basal / interval
    )  # Convert sequence length to the number of units
    seq_len_bolus = int(
        seq_len_bolus / interval
    )  # Convert sequence length to the number of units
    # ensure ph is an integer
    ph = int(ph / interval)  # Convert ph to the number of units

    # get the data
    test_data_1 = get_data(data_type, folder_path_1, p_num)

    # extract features
    input_data_1, output_data_1 = extract_features(test_data_1, ts=interval)

    # standardize the data
    input_data_1_scaled = standardize_val_data(input_data_1, input_scalers)
    output_data_1_scaled = standardize_val_data(output_data_1, output_scaler)

    # construct dataset
    x_test, y_test = create_sequence_for_proposed(
        input_data_1_scaled,
        output_data_1_scaled,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )

    # we also need the original data for the proposed model
    ori_y_test = get_ori_output(
        output_data_1,
        seq_len_cgm,
        seq_len_carb,
        seq_len_basal,
        seq_len_bolus,
        ph,
    )

    # create dataloader
    test_loader = create_test_dataloader_proposed(
        x_test, y_test, ori_y_test, batch_size
    )

    return test_loader


def create_sequence_for_proposed(
    input_data, output_data, seq_len_cgm, seq_len_carb, seq_len_basal, seq_len_bolus, ph
):
    """
    This function creates sequences for the proposed model.

    Parameters:
    input_data (dict): Dictionary containing the input data.
    output_data (dict): Dictionary containing the output data.
    seq_len_cgm (int): Sequence length for CGM data.
    seq_len_carb (int): Sequence length for carbohydrate intake data.
    seq_len_basal (int): Sequence length for basal insulin data.
    seq_len_bolus (int): Sequence length for bolus insulin data.
    ph (int): Prediction horizon.

    Returns:
    tuple: Input and output data in sequence format.
    """
    # get the max sequence length
    seq_len_max = max(seq_len_cgm, seq_len_basal, seq_len_bolus, seq_len_carb)

    # input data
    # input cgm
    input_select_cgm = select_data_for_proposed(
        input_data["cgm"], seq_len_cgm, seq_len_max, ph, "cgm"
    )
    input_cgm_seq = build_sequence_for_proposed(input_select_cgm, seq_len_cgm)
    # input carb intake
    input_select_carb = select_data_for_proposed(
        input_data["carb_intake"], seq_len_carb, seq_len_max, ph, "input"
    )
    input_carb_seq = build_sequence_for_proposed(input_select_carb, seq_len_carb)
    # input current basal
    input_select_basal = select_data_for_proposed(
        input_data["current_basal"], seq_len_basal, seq_len_max, ph, "input"
    )
    input_basal_seq = build_sequence_for_proposed(input_select_basal, seq_len_basal)
    # input current bolus
    input_select_bolus = select_data_for_proposed(
        input_data["current_bolus"], seq_len_bolus, seq_len_max, ph, "input"
    )
    input_bolus_seq = build_sequence_for_proposed(input_select_bolus, seq_len_bolus)

    # output G
    output_g = select_data_for_proposed(
        output_data["G"], seq_len_cgm, seq_len_max, ph, "output"
    )

    print(f"Input data shape: {input_cgm_seq.shape}")
    print(f"Output data shape: {output_g.shape}")

    # # chech the output is right
    # print(f"Input CGM time 0 value: {input_select_cgm[0]}")
    # print(f"Output G time 0 value: {output_g[0]}")
    # print(f"Input CGM time 60 value: {input_select_cgm[ph]}")
    assert (
        input_cgm_seq.shape[0]
        == input_carb_seq.shape[0]
        == input_basal_seq.shape[0]
        == input_bolus_seq.shape[0]
        == output_g.shape[0]
    ), "Mismatch in sample lengths across input/output variables"

    # construct x and y
    # x = np.stack(
    #     (
    #         input_cgm_seq,
    #         input_carb_seq,
    #         input_basal_seq,
    #         input_bolus_seq,
    #     ),
    #     axis=-1,
    # )
    x = {
        "cgm": input_cgm_seq,
        "carb_intake": input_carb_seq,
        "current_basal": input_basal_seq,
        "current_bolus": input_bolus_seq,
    }

    y = output_g.reshape(-1, 1)

    return x, y


def select_data_for_proposed(data, seq_len, max_seq_len, ph, data_type):
    """
    This function selects and processes data for the proposed model.

    Parameters:
    data (list or array): The input data to be selected.
    seq_len (int): Sequence length for the model.
    max_seq_len (int): Maximum sequence length across all input types.
    ph (int): Prediction horizon.
    data_type (str): Type of data, either 'cgm', 'output', or 'input'.

    Returns:
    np.ndarray: Processed data based on the data type and sequence length.
    """
    data = np.array(data)

    if data_type == "cgm":
        start = max_seq_len - seq_len
        end = -ph
        return data[start:end]

    elif data_type == "output":
        return data[max_seq_len + ph - 1 :]

    else:
        start = max_seq_len - seq_len + ph
        return data[start:]


def build_sequence_for_proposed(data, seq_len):
    """
    This function builds the sequence data based on the sequence length.

    Parameters:
    data (list or array): The input data to be built into sequences.
    seq_len (int): Sequence length for the model.

    Returns:
    np.ndarray: Sequence data with shape (num_samples, seq_len).
    """
    data = np.array(data).reshape(-1)
    num_samples = len(data) - seq_len + 1

    data_seq = np.zeros((num_samples, seq_len))
    for i in range(num_samples):
        data_seq[i] = data[i : i + seq_len]

    # Ensure the sequence is of the correct length
    print(f"Data sequence shape: {data_seq.shape}")

    return data_seq


def get_ori_output(
    data,
    seq_len_cgm,
    seq_len_carb,
    seq_len_basal,
    seq_len_bolus,
    ph,
):
    """
    This function retrieves the original output data for the proposed model.

    Parameters:
    data (dict): Dictionary containing the output data.
    seq_len_cgm (int): Sequence length for CGM data.
    seq_len_carb (int): Sequence length for carbohydrate intake data.
    seq_len_basal (int): Sequence length for basal insulin data.
    seq_len_bolus (int): Sequence length for bolus insulin data.
    ph (int): Prediction horizon.

    Returns:
    np.ndarray: Original output data reshaped to a single column.
    """
    max_seq_len = max(seq_len_cgm, seq_len_basal, seq_len_bolus, seq_len_carb)
    # output G
    ori_data = select_data_for_proposed(
        data["G"], seq_len_cgm, max_seq_len, ph, "output"
    )
    ori_data = ori_data.reshape(-1, 1)
    print(f"Original output data shape: {ori_data.shape}")
    return ori_data


def create_dataloaders_proposed(
    x_dict_train, y_train, x_dict_val, y_val, y_ori_train, y_ori_val, batch_size=128
):
    """
    Create PyTorch DataLoaders for training and validation datasets for the proposed model.

    Parameters:
    x_dict_train (dict): Dictionary containing training input data.
    y_train (np.ndarray): Training output data.
    x_dict_val (dict): Dictionary containing validation input data.
    y_val (np.ndarray): Validation output data.
    y_ori_train (np.ndarray): Original training output data.
    y_ori_val (np.ndarray): Original validation output data.
    batch_size (int): Batch size for the DataLoader.

    Returns:
    tuple: Training and validation DataLoaders.
    """
    # convert numpy arrays to PyTorch tensors
    x_train_tensor = {
        key: torch.tensor(x_dict_train[key], dtype=torch.float32)
        for key in x_dict_train
    }
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_ori_train_tensor = torch.tensor(y_ori_train, dtype=torch.float32)

    x_val_tensor = {
        key: torch.tensor(x_dict_val[key], dtype=torch.float32) for key in x_dict_val
    }
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_ori_val_tensor = torch.tensor(y_ori_val, dtype=torch.float32)

    # create MultiInputDataset
    train_dataset = MultiInputDataset(
        x_train_tensor, y_train_tensor, y_ori_train_tensor
    )
    val_dataset = MultiInputDataset(x_val_tensor, y_val_tensor, y_ori_val_tensor)

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_test_dataloader_proposed(x_dict_test, y_test, y_ori, batch_size=128):
    """
    Create a PyTorch DataLoader for the test dataset for the proposed model.

    Parameters:
    x_dict_test (dict): Dictionary containing test input data.
    y_test (np.ndarray): Test output data.
    y_ori (np.ndarray): Original test output data.
    batch_size (int): Batch size for the DataLoader.

    Returns:
    DataLoader: Test DataLoader.
    """
    # convert numpy arrays to PyTorch tensors
    x_test_tensor = {
        key: torch.tensor(x_dict_test[key], dtype=torch.float32) for key in x_dict_test
    }
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_ori_tensor = torch.tensor(y_ori, dtype=torch.float32)

    # create MultiInputDataset
    test_dataset = MultiInputDataset(x_test_tensor, y_test_tensor, y_ori_tensor)

    # create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
