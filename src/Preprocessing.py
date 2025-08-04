import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def get_data(data_type, folder_path, p_num):

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
    # current insulin basal
    current_basal = deal_insulin_basal(input_insulin["basal_reconstructed"])

    # get the current insulin bolus
    current_bolus = deal_insulin_bolus(input_insulin["bolus_reconstructed"])

    # # output: current G
    # output = mat_deconstruct(data["G"])
    # output_time = output["time"][1:]
    # output_values = output["signals"]["values"]
    # output_values = output_values[1:]  # Remove the first value

    # output: current cgm
    output_time = input_cgm_time
    output_values = input_cgm_values

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


# def deal_insulin_basal(data):
#     basal_ary = np.array(data)
#     # -> pmol/min
#     basal_ary = basal_ary * 100
#     basal_ary = basal_ary[576:]  # Remove the first 576 values
#     basal_ary = np.append(basal_ary, 0)  # Append a zero at the end

#     # resample the basal data from 5min to 1min
#     basal_1min = np.repeat(basal_ary, 5)
#     return basal_1min


def deal_insulin_basal(data):
    basal_ary = np.array(data)
    # -> pmol/min
    basal_ary = basal_ary * 100
    basal_ary = basal_ary[576:]  # Remove the first 576 values
    basal_ary = np.append(basal_ary, 0)  # Append a zero at the end

    return basal_ary


# def deal_insulin_bolus(data):
#     bolus_ary = np.array(data)
#     # -> pmol
#     bolus_ary = bolus_ary * 6000
#     bolus_ary = bolus_ary[576:]  # Remove the first 576 values
#     bolus_ary = np.append(bolus_ary, 0)  # Append a zero at the end

#     # resample the bolus data from 5min to 1min
#     bolus_len = len(bolus_ary)
#     bolus_1min = np.zeros((bolus_len, 5))
#     bolus_1min[:, 0] = bolus_ary
#     bolus_1min = bolus_1min.flatten()

#     return bolus_1min


def deal_insulin_bolus(data):
    bolus_ary = np.array(data)
    # -> pmol
    bolus_ary = bolus_ary * 6000
    bolus_ary = bolus_ary[576:]  # Remove the first 576 values
    bolus_ary = np.append(bolus_ary, 0)  # Append a zero at the end

    return bolus_ary


def data_standardization(data):
    # input data shape (sampels, )
    scalar = StandardScaler()
    data = np.array(data)
    data = data.reshape(-1, 1)  # Reshape to 2D array for StandardScaler
    data_scaled = scalar.fit_transform(data)
    data_scaled = data_scaled.flatten()  # Flatten back to 1D array
    return data_scaled, scalar


def data_inverse_standardization(data, scaler):
    # input data shape (samples, )
    data = np.array(data)
    data = data.reshape(-1, 1)  # Reshape to 2D array for inverse transformation
    data_inverse = scaler.inverse_transform(data)
    data_inverse = data_inverse.flatten()  # Flatten back to 1D array
    return data_inverse


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

    # transfer sequence length to the number of units
    seq_len = int(seq_len / interval)  # Convert sequence length to the number of units
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


def test_data_preprocessing_baseline(
    data_type, folder_path_1, p_num, seq_len, ph, interval, input_scalers, output_scaler
):
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


def standardize_2_datasets(data_4days, data_30days):
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


def standardize_val_data(data, scalers):
    val_data_scaled = {}

    for key, value in data.items():
        value = np.array(value).reshape(-1, 1)
        scaler = scalers[key]
        value_scaled = scaler.transform(value)
        val_data_scaled[key] = value_scaled.flatten()

    return val_data_scaled


def select_build_sequence_data(input_data, output_data, seq_len, ph):
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

    data = np.array(data).reshape(-1)
    num_samples = len(data) - seq_len + 1

    data_seq = np.zeros((num_samples, seq_len))
    for i in range(num_samples):
        data_seq[i] = data[i : i + seq_len]

    return data_seq


def create_dataloaders(x_train, y_train, x_val, y_val, batch_size=128):

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
    # convert numpy arrays to PyTorch tensors
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # create TensorDataset
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def split_train_val(data, train_days, val_days, interval=5):
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
    def __init__(self, x_dict, y, y_ori):
        self.x_dict = x_dict
        self.y = y
        self.y_ori = y_ori

    def __len__(self):
        return len(self.y)

    # def __getitem__(self, idx):
    #     x_item = {
    #         key: torch.tensor(self.x_dict[key][idx], dtype=torch.float32)
    #         for key in self.x_dict
    #     }
    #     y_item = torch.tensor(self.y[idx], dtype=torch.float32)
    #     ori_y_item = torch.tensor(self.y_ori[idx], dtype=torch.float32)
    #     return x_item, y_item, ori_y_item
    def __getitem__(self, idx):
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
    max_seq_len = max(seq_len_cgm, seq_len_basal, seq_len_bolus, seq_len_carb)
    # output G
    ori_data = select_data_for_proposed(
        data["G"], seq_len_cgm, max_seq_len, ph, "output"
    )
    ori_data = ori_data.reshape(-1, 1)
    print(f"Original output data shape: {ori_data.shape}")
    return ori_data


# def create_dataloaders_proposed(
#     x_train, y_train, x_val, y_val, y_ori_train, y_ori_val, batch_size=128
# ):

#     # convert numpy arrays to PyTorch tensors
#     x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
#     y_ori_train_tensor = torch.tensor(y_ori_train, dtype=torch.float32)
#     x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
#     y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
#     y_ori_val_tensor = torch.tensor(y_ori_val, dtype=torch.float32)

#     # create TensorDataset
#     train_dataset = TensorDataset(x_train_tensor, y_train_tensor, y_ori_train_tensor)
#     val_dataset = TensorDataset(x_val_tensor, y_val_tensor, y_ori_val_tensor)

#     # create DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader


def create_dataloaders_proposed(
    x_dict_train, y_train, x_dict_val, y_val, y_ori_train, y_ori_val, batch_size=128
):
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


# def create_test_dataloader_proposed(x_test, y_test, y_ori, batch_size=128):
#     # convert numpy arrays to PyTorch tensors
#     x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#     y_ori_tensor = torch.tensor(y_ori, dtype=torch.float32)

#     # create TensorDataset
#     test_dataset = TensorDataset(x_test_tensor, y_test_tensor, y_ori_tensor)

#     # create DataLoader
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return test_loader


def create_test_dataloader_proposed(x_dict_test, y_test, y_ori, batch_size=128):
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
