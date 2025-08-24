import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy.ndimage import gaussian_filter1d

import src.Preprocessing as prep
import src.Visualising as vis

# from torch.nn.functional import conv1d


class PersonalizedModelB1(nn.Module):
    """
    A simple personalised LSTM model for blood glucose prediction.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int, optional): Size of the hidden layer. Default is 96.
        output_size (int, optional): Size of the output layer. Default is 1.
        num_layers (int, optional): Number of LSTM layers. Default is 1.
    """

    def __init__(self, input_size, hidden_size=96, output_size=1, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        This function defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        output = self.fc(out[:, -1, :])
        return output


def baseline_model_train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    epochs,
    device,
):
    """
    This function defines the training process for the baseline model.

    Args:
        model (PersonalizedModel_B1): The baseline model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        epochs (int): Number of training epochs.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        model (PersonalizedModel_B1): The trained model.
        fig, ax: Matplotlib figure and axes objects for visualizing the training process.
    """
    train_losses = []
    val_losses = []
    model = model.to(device)

    # training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

        # validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_dataloader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                output_val = model(x_val)
                loss_val = criterion(output_val, y_val)
                val_loss += loss_val.item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}")

    # print the learning curves
    fig, ax = vis.plot_lr_1(train_losses, val_losses)

    return model, fig, ax


def baseline_model_eval(
    model, test_dataloader, scaler, device, ticks_per_day, time_steps
):
    """
    Evaluate the baseline model on the test dataset (still in progress).

    Args:
        model (PersonalizedModel_B1): The trained baseline model.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        scaler (sklearn.preprocessing.StandardScaler): Scaler used for standardization.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        predictions (list): List of predictions for the test dataset.
        truth (list): List of ground truth values for the test dataset.
        fig, ax: Matplotlib figure and axes objects for visualizing the predictions.
    """
    model = model.to(device)
    model.eval()
    predictions = []
    truth = []

    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            predictions.append(output)
            truth.append(y)
    predictions = torch.cat(predictions, dim=0).cpu().numpy().flatten().tolist()
    truth = torch.cat(truth, dim=0).cpu().numpy().flatten().tolist()

    # inverse standardization
    predictions = prep.data_inverse_standardization(predictions, scaler)
    truth = prep.data_inverse_standardization(truth, scaler)

    # visualize the predictions
    fig, ax = vis.plot_pred_visualisation(predictions, truth, ticks_per_day, time_steps)

    # plot thresholds
    fig_theshold, ax_threshold = vis.plot_pred_threshold_visualisation(
        predictions, truth, ticks_per_day, time_steps
    )

    # apply alarm strategy
    hypo_threshold = 70
    hyper_threshold = 180
    pred_alarms = alarm_strategy(predictions, hypo_threshold, hyper_threshold)
    truth_alarms = alarm_strategy(truth, hypo_threshold, hyper_threshold)

    return (
        predictions,
        truth,
        pred_alarms,
        truth_alarms,
        fig,
        ax,
        fig_theshold,
        ax_threshold,
    )


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def paper_model_eval(model, test_dataloader, scaler, device, ticks_per_day, time_steps):

    predictions = []
    truth = []

    # before the evaluation, we need to set the model to evaluation mode
    for x, y in test_dataloader:
        x_np = to_numpy(x)  # shape (batch, time_steps, features)
        y_np = to_numpy(y)

        # make predictions
        out = model.predict(x_np, verbose=0)  # shape (batch, time_steps, features)
        predictions.append(out)
        truth.append(y_np)

    # concatenate the predictions and truth
    predictions = np.concatenate(predictions, axis=0).flatten().tolist()
    truth = np.concatenate(truth, axis=0).flatten().tolist()

    # inverse standardization
    predictions = prep.data_inverse_standardization_paper(predictions, scaler)
    truth = prep.data_inverse_standardization_paper(truth, scaler)

    # visualize the predictions
    fig, ax = vis.plot_pred_visualisation(predictions, truth, ticks_per_day, time_steps)

    # plot thresholds
    fig_theshold, ax_threshold = vis.plot_pred_threshold_visualisation(
        predictions, truth, ticks_per_day, time_steps
    )

    # apply alarm strategy
    hypo_threshold = 70
    hyper_threshold = 180
    pred_alarms = alarm_strategy(predictions, hypo_threshold, hyper_threshold)
    truth_alarms = alarm_strategy(truth, hypo_threshold, hyper_threshold)

    return (
        predictions,
        truth,
        pred_alarms,
        truth_alarms,
        fig,
        ax,
        fig_theshold,
        ax_threshold,
    )


class ChannelAttention(nn.Module):
    """
    This class implements a channel attention mechanism that computes attention weights
    based on the context and prior knowledge.

    Args:
        hidden_size (int): Size of the hidden layer.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, hidden_size * 2 + 1))

    def forward(self, context, prior_knowledge):
        """
        This function computes the attention weights and weighted context.

        Args:
            context (list): List of tensors containing context information.
            prior_knowledge (list): List of tensors containing prior knowledge.

        Returns:
            tuple: A tuple containing the weighted context and attention weights.
        """
        assert len(context) == len(prior_knowledge), "Mismatched input channels"
        batch_size = context[0].size(0)

        context_with_prior = [
            torch.cat((c, p), dim=1) for c, p in zip(context, prior_knowledge)
        ]
        context_with_prior = torch.stack(
            context_with_prior, dim=1
        )  # (batch_size, 4, hidden_size * 2 + 1)

        query = self.query.expand(
            batch_size, 1, -1
        )  # (batch_size, 1, hidden_size * 2 + 1)
        attention_score = torch.bmm(query, context_with_prior.transpose(1, 2))
        # shape: (batch_size, 1, 4)

        attention_weights = F.softmax(attention_score, dim=-1)
        # shape: (batch_size, 1, 4)

        attended_context = torch.bmm(attention_weights, context_with_prior)
        # shape: (batch_size, 1, hidden_size * 2 + 1)

        attended_context = attended_context.squeeze(1)
        # (batch_size, hidden_size * 2 + 1)

        atten_weights = attention_weights.squeeze(1)  # (batch_size, 4)
        attended_context = attended_context[:, :-1]  # Remove the prior knowledge part

        return (
            attended_context,
            atten_weights,
        )  # (weighted_context, attention_weights)


class ProposedModel(nn.Module):
    """
    Our proposed model that combines multi-channel LSTM and channel attention.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        output_size (int): Size of the output layer.
        num_layers (int, optional): Number of LSTM layers. Default is 1.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm_cgm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm_basal_insulin = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm_physio_basal_insulin = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm_bolus_insulin = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm_physio_bolus_insulin = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm_meal = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm_physio_meal = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

        # Channel attention layer
        self.channel_attention = ChannelAttention(hidden_size)

        # weights for different BG range
        self.weights = nn.Parameter(torch.tensor([2.0, 2.0, 1.0]))

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(
        self,
        cgm,
        basal_insulin,
        bolus_insulin,
        meal,
        physio_basal_insulin,
        physio_bolus_insulin,
        physio_meal,
    ):
        """
        This function defines the forward pass of the proposed model.

        Args:
            cgm (torch.Tensor): Input tensor for past CGM data (batch_size, seq_length, input_size).
            insulin (torch.Tensor): Insulin data (batch_size, seq_length, input_size).
            meal (torch.Tensor): Input tensor for meal data (batch_size, seq_length, input_size).

        Returns:
            tuple: A tuple containing the output tensor and attention weights.
        """
        out_cgm, _ = self.lstm_cgm(cgm)
        out_basal_insulin, _ = self.lstm_basal_insulin(basal_insulin)
        out_bolus_insulin, _ = self.lstm_bolus_insulin(bolus_insulin)
        out_meal, _ = self.lstm_meal(meal)
        out_physio_basal_insulin, _ = self.lstm_physio_basal_insulin(
            physio_basal_insulin
        )
        out_physio_bolus_insulin, _ = self.lstm_physio_bolus_insulin(
            physio_bolus_insulin
        )
        out_physio_meal, _ = self.lstm_physio_meal(physio_meal)

        out_cgm = out_cgm[
            :, -1, :
        ]  # Get the last time step output shape (batch_size, hidden_size * 2)
        out_basal_insulin = out_basal_insulin[:, -1, :]
        out_bolus_insulin = out_bolus_insulin[:, -1, :]
        out_meal = out_meal[:, -1, :]
        out_physio_basal_insulin = out_physio_basal_insulin[:, -1, :]
        out_physio_bolus_insulin = out_physio_bolus_insulin[:, -1, :]
        out_physio_meal = out_physio_meal[:, -1, :]

        # prior knowledge
        prior_cgm = torch.ones(cgm.size(0), 1, device=cgm.device)
        prior_insulin = basal_insulin[:, -1, :].mean(dim=1, keepdim=True)
        prior_insulin_bolus = bolus_insulin[:, -1, :].mean(dim=1, keepdim=True)
        prior_meal = meal[:, -1, :].mean(dim=1, keepdim=True)
        prior_physio_basal_insulin = physio_basal_insulin[:, -1, :].mean(
            dim=1, keepdim=True
        )
        prior_physio_bolus_insulin = physio_bolus_insulin[:, -1, :].mean(
            dim=1, keepdim=True
        )
        prior_physio_meal = physio_meal[:, -1, :].mean(dim=1, keepdim=True)

        # Concatenate the outputs
        context = [
            out_cgm,
            out_basal_insulin,
            out_bolus_insulin,
            out_meal,
            out_physio_basal_insulin,
            out_physio_bolus_insulin,
            out_physio_meal,
        ]
        prior_knowledge = [
            prior_cgm,
            prior_insulin,
            prior_insulin_bolus,
            prior_meal,
            prior_physio_basal_insulin,
            prior_physio_bolus_insulin,
            prior_physio_meal,
        ]

        # channel attention
        weighted_context, atten_weights = self.channel_attention(
            context, prior_knowledge
        )

        # fully connected layer
        output = self.fc(weighted_context)
        return output, atten_weights


def min_max_normalization(data):
    min_value = data.min(dim=1, keepdim=True)[0]
    max_value = data.max(dim=1, keepdim=True)[0]
    normalized_data = (data - min_value) / (max_value - min_value + 1e-8)
    return normalized_data


def physiological_layer(input_seq, lamda, kernel_size):
    """
    Applies the physiological layer to preprocess the insulin and meal data.
    """
    # Create a kernel based on the physiological model
    t = torch.arange(0, kernel_size).float()
    kernel = (t / lamda**2) * torch.exp(-t / lamda)
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, -1).to(input_seq.device)  # (1,1,K)

    kernel = torch.flip(kernel, dims=[-1])

    input_seq = input_seq.float()
    input_seq = input_seq.unsqueeze(1)  # Add channel dimension
    input_seq = torch.clamp(input_seq, min=0)  # Ensure non-negative values

    padding = int(kernel_size) - 1
    input_seq = F.pad(input_seq, (padding, 0), mode="constant", value=0)

    # Apply the convolution (strict convolution after kernel flip)
    output_seq = F.conv1d(input_seq, kernel)
    output_seq = output_seq.squeeze(1)  # Remove channel dimension

    output_seq = min_max_normalization(output_seq)

    return output_seq


# proposed model training function
def proposed_model_train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    epochs,
    alpha,  # loss weight for variance term
    beta,  # loss weight for regularization term
    seq_len_carb_intake,  # sequence length for carb intake
    seq_len_basal_insulin,  # sequence length for basal insulin
    seq_len_bolus_insulin,  # sequence length for bolus insulin
    tp_insulin_basal,  # peak time for basal insulin response
    tp_insulin_bolus,  # peak time for bolus insulin response
    tp_meal,  # peak time for meal response
    interval,  # time interval for the physiological model
    device,
):
    train_losses = []
    train_losses_2 = []  # for debugging and monitoring
    val_losses = []
    val_losses_2 = []  # for debugging and monitoring
    model = model.to(device)
    # physiological layer for insulin and meal
    tp_insulin_basal = tp_insulin_basal / interval  # convert peak time to time steps
    tp_insulin_bolus = tp_insulin_bolus / interval  # convert peak time to time steps
    tp_meal = tp_meal / interval  # convert peak time to time steps
    # kernel size for basal insulin and bolus insulin
    kernel_size_basal_insulin = seq_len_basal_insulin / interval
    kernel_size_bolus_insulin = seq_len_bolus_insulin / interval
    kernel_size_carb_intake = seq_len_carb_intake / interval

    # training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_loss_2 = 0.0  # for debugging and monitoring
        all_predictions = []
        all_truths = []
        for x, y_norm, y_raw in tqdm.tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            optimizer.zero_grad()
            cgm = x["cgm"]
            meal = x["carb_intake"]
            basal_insulin = x["current_basal"]
            bolus_insulin = x["current_bolus"]
            cgm = cgm.to(model.weights.device)
            basal_insulin = basal_insulin.to(model.weights.device)
            bolus_insulin = bolus_insulin.to(model.weights.device)
            meal = meal.to(model.weights.device)
            y_norm = y_norm.to(model.weights.device)
            y_raw = y_raw.to(model.weights.device)

            # Apply physiological layer to insulin and meal
            # basal_insulin = basal_insulin.squeeze(2)
            basal_insulin_processed = physiological_layer(
                basal_insulin, tp_insulin_basal, 1000
            )
            basal_insulin_preprocessed = basal_insulin_processed.unsqueeze(
                2
            )  # (batch, seq_len) -> (batch, seq_len, 1)

            # bolus_insulin = bolus_insulin.squeeze(2)
            bolus_insulin_processed = physiological_layer(
                bolus_insulin, tp_insulin_bolus, 1000
            )
            bolus_insulin_preprocessed = bolus_insulin_processed.unsqueeze(
                2
            )  # (batch, seq_len) -> (batch, seq_len, 1)

            # meal = meal.squeeze(2)  # (batch, seq_len, 1)
            meal_processed = physiological_layer(meal, tp_meal, 1000)
            meal_preprocessed = meal_processed.unsqueeze(
                2
            )  # (batch, seq_len) -> (batch, seq_len, 1)

            # preprocess the inputs cgm
            cgm = cgm.unsqueeze(2)  # (batch, seq_len) -> (batch, seq_len, 1)
            meal = meal.unsqueeze(2)  # (batch, seq_len) -> (batch, seq_len, 1)
            basal_insulin = basal_insulin.unsqueeze(2)
            bolus_insulin = bolus_insulin.unsqueeze(2)

            # forward pass
            pred, att_weights = model(
                cgm,
                basal_insulin,
                bolus_insulin,
                meal,
                basal_insulin_preprocessed,
                bolus_insulin_preprocessed,
                meal_preprocessed,
            )
            # -------------------loss function-------------------
            loss = com_loss_function(pred, y_norm, y_raw, model, alpha, beta)
            # loss = nn.MSELoss()(pred, y_norm)  # using MSE loss for training
            loss_2 = F.mse_loss(pred, y_norm)  # for debugging and monitoring
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_2 += loss_2.item()  # for debugging and monitoring
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        train_loss_2 /= len(train_dataloader)
        train_losses_2.append(train_loss_2)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train MSE: {train_loss_2:.4f}"
        )

        # validation loop
        model.eval()
        val_loss = 0.0
        val_loss_2 = 0.0  # for debugging and monitoring
        with torch.no_grad():
            for x_val, y_norm_val, y_raw_val in val_dataloader:
                # extract the inputs and targets
                cgm_val = x_val["cgm"]
                meal_val = x_val["carb_intake"]
                basal_insulin_val = x_val["current_basal"]
                bolus_insulin_val = x_val["current_bolus"]

                cgm_val = cgm_val.to(model.weights.device)
                basal_insulin_val = basal_insulin_val.to(model.weights.device)
                bolus_insulin_val = bolus_insulin_val.to(model.weights.device)
                meal_val = meal_val.to(model.weights.device)
                y_norm_val = y_norm_val.to(model.weights.device)
                y_raw_val = y_raw_val.to(model.weights.device)

                # Apply physiological layer to insulin and meal
                # basal_insulin_val = basal_insulin_val.squeeze(2)
                basal_insulin_processed_val = physiological_layer(
                    basal_insulin_val, tp_insulin_basal, 1000
                )
                basal_insulin_preprocessed_val = basal_insulin_processed_val.unsqueeze(
                    2
                )  # (batch, seq_len) -> (batch, seq_len, 1)

                # bolus_insulin_val = bolus_insulin_val.squeeze(2)
                bolus_insulin_processed_val = physiological_layer(
                    bolus_insulin_val, tp_insulin_bolus, 1000
                )
                bolus_insulin_preprocessed_val = bolus_insulin_processed_val.unsqueeze(
                    2
                )  # (batch, seq_len) -> (batch, seq_len, 1)

                # meal_val = meal_val.squeeze(2)
                meal_processed_val = physiological_layer(meal_val, tp_meal, 1000)
                meal_preprocessed_val = meal_processed_val.unsqueeze(
                    2
                )  # (batch, seq_len) -> (batch, seq_len, 1)

                # preprocess the inputs cgm
                cgm_val = cgm_val.unsqueeze(
                    2
                )  # (batch, seq_len) -> (batch, seq_len, 1)
                meal_val = meal_val.unsqueeze(2)
                basal_insulin_val = basal_insulin_val.unsqueeze(2)
                bolus_insulin_val = bolus_insulin_val.unsqueeze(2)

                # forward pass
                pred_val, _ = model(
                    cgm_val,
                    basal_insulin_val,
                    bolus_insulin_val,
                    meal_val,
                    basal_insulin_preprocessed_val,
                    bolus_insulin_preprocessed_val,
                    meal_preprocessed_val,
                )

                loss_val = com_loss_function(
                    pred_val, y_norm_val, y_raw_val, model, alpha, beta
                )
                # loss_val = nn.MSELoss()(
                #     pred_val, y_norm_val
                # )  # using MSE loss for validation
                loss_val_2 = F.mse_loss(
                    pred_val, y_norm_val
                )  # for debugging and monitoring
                val_loss += loss_val.item()
                val_loss_2 += loss_val_2.item()  # for debugging and monitoring
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        val_loss_2 /= len(val_dataloader)
        val_losses_2.append(val_loss_2)
        print(
            f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val MSE Loss: {val_loss_2:.4f}"
        )

    # print the learning curves
    fig, ax = vis.plot_lr_2(train_losses, train_losses_2, val_losses, val_losses_2)

    return model, fig, ax


def get_peak_time(
    patient_num,
    var_type,  # 'basal', 'bolus', 'meal'
):
    data_type = "test"
    data = prep.get_data(data_type, var_type, patient_num)
    input_data, output_data = prep.extract_features(data)
    idx = 0

    if var_type == "ts-dtI":  # "bolus insulin"
        idx = output_data["G"].index(min(output_data["G"]))
    elif var_type == "ts-dtM":  # "carb intake"
        idx = output_data["G"].index(max(output_data["G"]))

    peak_time = (idx - (288 / 2)) * 5  # convert to minutes

    return int(peak_time)


def proposed_model_train_for_99(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    epochs,
    alpha,
    beta,
    seq_len_carb_intake,
    seq_len_basal_insulin,
    seq_len_bolus_insulin,
    tp_insulin_basal,
    tp_insulin_bolus,
    tp_meal,
    interval,
    device,
    patience: int = 3,
    min_delta: float = 0.0,
    use_plateau_scheduler: bool = True,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 2,
    scheduler_min_lr: float = 1e-6,
):
    def _to_int_ge1(x):
        return max(int(round(float(x))), 1)

    # default device
    device = torch.device(device)
    model = model.to(device)

    # transform peak times and sequence lengths to "steps"
    tp_insulin_basal = float(tp_insulin_basal) / float(interval)
    tp_insulin_bolus = float(tp_insulin_bolus) / float(interval)
    tp_meal = float(tp_meal) / float(interval)

    # kernel size for physiological layers
    kernel_size_basal_insulin = _to_int_ge1(
        float(seq_len_basal_insulin) / float(interval)
    )
    kernel_size_bolus_insulin = _to_int_ge1(
        float(seq_len_bolus_insulin) / float(interval)
    )
    kernel_size_carb_intake = _to_int_ge1(float(seq_len_carb_intake) / float(interval))

    # record the losses
    train_losses, train_losses_2 = [], []
    val_losses, val_losses_2 = [], []

    # hyperparameters for early stopping
    best_val = float("inf")
    best_state = None
    stale_cnt = 0
    stopped_early = False

    # scheduler setup
    scheduler = None
    if use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
            verbose=True,
        )

    for epoch in range(epochs):
        # begin training
        model.train()
        running_train_loss = 0.0
        running_train_mse = 0.0

        for x, y_norm, y_raw in tqdm.tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"
        ):
            optimizer.zero_grad()

            cgm = x["cgm"].to(device)
            meal = x["carb_intake"].to(device)
            basal_insulin = x["current_basal"].to(device)
            bolus_insulin = x["current_bolus"].to(device)
            y_norm = y_norm.to(device)
            y_raw = y_raw.to(device)

            # physiological layer for insulin and meal
            bi = physiological_layer(
                basal_insulin, tp_insulin_basal, kernel_size_basal_insulin
            ).unsqueeze(2)
            boli = physiological_layer(
                bolus_insulin, tp_insulin_bolus, kernel_size_bolus_insulin
            ).unsqueeze(2)
            meal_p = physiological_layer(
                meal, tp_meal, kernel_size_carb_intake
            ).unsqueeze(2)

            # preprocess the inputs cgm
            cgm = cgm.unsqueeze(2)

            # perform forward pass
            pred, att_weights = model(cgm, bi, boli, meal_p)

            # loss calculation
            loss = com_loss_function(pred, y_norm, y_raw, model, alpha, beta)
            # loss = nn.MSELoss()(pred, y_norm)
            loss_2 = F.mse_loss(pred, y_norm)  # for debugging and monitoring

            # backward pass and optimization
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_mse += loss_2.item()

        # calculate average loss for the epoch
        epoch_train_loss = running_train_loss / max(len(train_dataloader), 1)
        epoch_train_mse = running_train_mse / max(len(train_dataloader), 1)
        train_losses.append(epoch_train_loss)
        train_losses_2.append(epoch_train_mse)
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train MSE: {epoch_train_mse:.4f}"
        )

        # -------------------- validation --------------------
        model.eval()
        running_val_loss = 0.0
        running_val_mse = 0.0

        with torch.no_grad():
            for x_val, y_norm_val, y_raw_val in val_dataloader:
                cgm_v = x_val["cgm"].to(device)
                meal_v_in = x_val["carb_intake"].to(device)
                basal_insulin_v = x_val["current_basal"].to(device)
                bolus_insulin_v = x_val["current_bolus"].to(device)
                y_norm_v = y_norm_val.to(device)
                y_raw_v = y_raw_val.to(device)

                bi_v = physiological_layer(
                    basal_insulin_v, tp_insulin_basal, kernel_size_basal_insulin
                ).unsqueeze(2)
                boli_v = physiological_layer(
                    bolus_insulin_v, tp_insulin_bolus, kernel_size_bolus_insulin
                ).unsqueeze(2)
                meal_v = physiological_layer(
                    meal_v_in, tp_meal, kernel_size_carb_intake
                ).unsqueeze(2)
                cgm_v = cgm_v.unsqueeze(2)

                pred_v, _ = model(cgm_v, bi_v, boli_v, meal_v)

                lv = com_loss_function(pred_v, y_norm_v, y_raw_v, model, alpha, beta)
                # lv = nn.MSELoss()(pred_v, y_norm_v)
                lv_2 = F.mse_loss(pred_v, y_norm_v)

                running_val_loss += lv.item()
                running_val_mse += lv_2.item()

        epoch_val_loss = running_val_loss / max(len(val_dataloader), 1)
        epoch_val_mse = running_val_mse / max(len(val_dataloader), 1)
        val_losses.append(epoch_val_loss)
        val_losses_2.append(epoch_val_mse)
        print(
            f"Epoch {epoch+1}/{epochs}, Val Loss: {epoch_val_loss:.4f}, Val MSE Loss: {epoch_val_mse:.4f}"
        )

        # scheduler step
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # -------------------- early stopping --------------------
        if_improved = epoch_val_loss < (best_val - float(min_delta))

        if if_improved:
            best_val = epoch_val_loss
            stale_cnt = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            if (epoch + 1) > 25:  # start counting after 25 epochs
                stale_cnt += 1
                if stale_cnt >= patience:
                    stopped_early = True
                    print(
                        f"Early stopping triggered at epoch {epoch+1} (best val loss: {best_val:.6f})."
                    )
                    break

    # -------------------- load best model state --------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    if not stopped_early:
        print(f"Training finished all {epochs} epochs.")
    else:
        print(
            f"Training stopped early at epoch {epoch+1} (best val loss: {best_val:.6f})."
        )

    # plot the learning curves
    fig, ax = vis.plot_lr_2(train_losses, train_losses_2, val_losses, val_losses_2)

    return model, fig, ax


def exp_smoothing(data, alpha):
    smoothed = [data[0]]  # Initialize with the first data point
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])

    return smoothed


def proposed_model_eval(
    model,
    test_dataloader,
    tp_insulin_basal,
    tp_insulin_bolus,
    tp_meal,
    seq_len_carb_intake,
    seq_len_basal_insulin,
    seq_len_bolus_insulin,
    interval,
    scaler,
    device,
    ticks_per_day,
    time_steps,
):
    """
    Evaluate the proposed model on the test dataset.

    Args:
        model (ProposedModel): The trained model.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        tp_insulin (float): Peak time for insulin response.
        tp_meal (float): Peak time for meal response.
        kernel_size (int): Time window size for the physiological model.

    Returns:
        list: List of predictions for the test dataset.
    """
    model = model.to(device)
    model.eval()
    predictions = []
    truths = []
    attention_weights = []
    # physiological layer for insulin and meal
    tp_insulin_basal = tp_insulin_basal / interval  # convert peak time to time steps
    tp_insulin_bolus = tp_insulin_bolus / interval  # convert peak time to time steps
    tp_meal = tp_meal / interval  # convert peak time to time steps

    # kernel size for basal insulin and bolus insulin
    kernel_size_basal_insulin = seq_len_basal_insulin / interval
    kernel_size_bolus_insulin = seq_len_bolus_insulin / interval
    kernel_size_carb_intake = seq_len_carb_intake / interval

    with torch.no_grad():
        for x, y_norm, y_raw in test_dataloader:
            # extract the inputs and targets
            cgm = x["cgm"]
            meal = x["carb_intake"]
            basal_insulin = x["current_basal"]
            bolus_insulin = x["current_bolus"]

            # move the inputs and targets to the device
            cgm = cgm.to(model.weights.device)
            basal_insulin = basal_insulin.to(model.weights.device)
            bolus_insulin = bolus_insulin.to(model.weights.device)
            meal = meal.to(model.weights.device)
            y_norm = y_norm.to(model.weights.device)
            y_raw = y_raw.to(model.weights.device)

            # Apply physiological layer to insulin and meal
            # basal_insulin = basal_insulin.squeeze(2)
            basal_insulin_processed = physiological_layer(
                basal_insulin, tp_insulin_basal, 1000
            )
            basal_insulin_preprocessed = basal_insulin_processed.unsqueeze(2)

            # bolus_insulin = bolus_insulin.squeeze(2)
            bolus_insulin_processed = physiological_layer(
                bolus_insulin, tp_insulin_bolus, 1000
            )
            bolus_insulin_preprocessed = bolus_insulin_processed.unsqueeze(2)

            # meal = meal.squeeze(2)
            meal_processed = physiological_layer(meal, tp_meal, 1000)
            meal_preprocessed = meal_processed.unsqueeze(2)

            # preprocess the inputs cgm
            cgm = cgm.unsqueeze(2)  # (batch, seq_len) -> (batch, seq_len, 1)
            meal = meal.unsqueeze(2)
            basal_insulin = basal_insulin.unsqueeze(2)
            bolus_insulin = bolus_insulin.unsqueeze(2)

            # forward pass
            pred, attn_weights = model(
                cgm,
                basal_insulin,
                bolus_insulin,
                meal,
                basal_insulin_preprocessed,
                bolus_insulin_preprocessed,
                meal_preprocessed,
            )
            predictions.append(pred)
            truths.append(y_norm)
            attention_weights.append(attn_weights)
    predictions = torch.cat(predictions, dim=0).cpu().numpy().flatten().tolist()
    truths = torch.cat(truths, dim=0).cpu().numpy().flatten().tolist()
    attention_weights = torch.cat(attention_weights, dim=0).cpu().numpy()

    # inverse standardization
    predictions = prep.data_inverse_standardization(predictions, scaler)
    # predicitons = exp_smoothing(predictions, alpha=0.2)  # apply exponential smoothing
    predictions = gaussian_filter1d(predictions, sigma=1)  # apply Gaussian filter
    truths = prep.data_inverse_standardization(truths, scaler)

    # visualize the predictions
    fig, ax = vis.plot_pred_visualisation(
        predictions, truths, ticks_per_day, time_steps
    )

    # plot thresholds
    fig_threshold, ax_threshold = vis.plot_pred_threshold_visualisation(
        predictions, truths, ticks_per_day, time_steps
    )

    # apply alarm strategy
    hypo_threshold = 70
    hyper_threshold = 180
    pred_alarms = alarm_strategy(predictions, hypo_threshold, hyper_threshold)
    truth_alarms = alarm_strategy(truths, hypo_threshold, hyper_threshold)

    return (
        predictions,
        truths,
        pred_alarms,
        truth_alarms,
        attention_weights,
        fig,
        ax,
        fig_threshold,
        ax_threshold,
    )


# ---- used to smooth the range weight ----
def smooth_range_weight(bg, low=70.0, high=180.0, k=0.03, w_in=1.0, w_out=2.0):

    d_low = F.relu(low - bg)
    d_high = F.relu(bg - high)
    d = d_low + d_high
    w = w_in + (w_out - w_in) * torch.tanh(k * d)
    return w


# -----------Huber Loss and Asymmetric Huber Loss Functions-----------
def huber_loss(e, delta=10.0):
    """
    e = pred - truth
    """
    abs_e = torch.abs(e)
    quad = (
        0.5
        * torch.minimum(abs_e, torch.tensor(delta, device=e.device, dtype=e.dtype)) ** 2
    )
    lin = delta * (
        abs_e
        - torch.minimum(abs_e, torch.tensor(delta, device=e.device, dtype=e.dtype))
    )
    return quad + lin


def asymmetric_huber_loss(e, delta=10.0, over_ratio=1.2, under_ratio=1.0):
    """
    e = pred - truth
    """
    base = huber_loss(e, delta=delta)
    factor = torch.where(
        e > 0,
        torch.tensor(over_ratio, device=e.device, dtype=e.dtype),
        torch.tensor(under_ratio, device=e.device, dtype=e.dtype),
    )
    return factor * base


# define the loss function for the proposed model
def com_loss_function(
    pred,
    truth,
    ori_truth,
    model,
    alpha,
    beta,
    low=70.0,
    high=180.0,
    k=0.05,  # 0.03
    w_in=1.0,
    w_out=3.0,
    lam_range=1.2,  # 1.4
):

    # smooth_range_weight
    w_t = smooth_range_weight(
        ori_truth, low=low, high=high, k=k, w_in=w_in, w_out=w_out
    )
    w_t = w_t / (w_t.mean().clamp_min(1e-6))

    # use MSE to calculate the prediction error
    err = pred - truth
    per_elem = err**2
    l_prd = (w_t * per_elem).mean()

    # calculate the weighted variance
    d_pred = pred[1:, :] - pred[:-1, :]
    d_truth = truth[1:, :] - truth[:-1, :]
    d_err = d_pred - d_truth
    w_pair = 0.5 * (w_t[1:, :] + w_t[:-1, :])
    var_err = d_err**2
    l_var = (w_pair * var_err).mean()

    # # over high
    # over_high = F.relu(pred - high) * (truth <= high)
    # range_penalty = (over_high**2).mean() * lam_range

    # over high and under low
    over_high = F.relu(pred - high) * (truth <= high)
    under_low = F.relu(low - pred) * (truth >= low)
    range_penalty = (over_high**2 + under_low**2).mean() * lam_range

    total_loss = l_prd + alpha * l_var + range_penalty
    # total_loss = l_prd + range_penalty
    return total_loss


def alarm_strategy(predictions, hypo_threshold, hyper_threshold):
    """
    Apply the alarm strategy based on the predictions.

    Args:
        predictions (list): Model predictions.
        hypo_threshold (float): Hypoglycemia threshold.
        hyper_threshold (float): Hyperglycemia threshold.

    Returns:
        list: List of alarms for each prediction.
    """
    # hypo -> 0, hyper -> 1, normal -> 2
    alarms = []
    for pred in predictions:
        if pred < hypo_threshold:
            alarms.append(0)
        elif pred > hyper_threshold:
            alarms.append(1)
        else:
            alarms.append(2)
    return alarms


def visualise_phy_layer(
    data_type,
    data_folder,
    p_num,
    figure_folder,
    peak_time_basal,
    peak_time_bolus,
    peak_time_carb,
    interval,
):
    train_plot = prep.get_data(
        data_type=data_type, folder_path=data_folder, p_num=p_num
    )

    train_plot_input, _ = prep.extract_features(train_plot)

    basal_insulin = train_plot_input["current_basal"][-288:]
    bolus_insulin = train_plot_input["current_bolus"][-288:]
    meal = train_plot_input["carb_intake"][-288:]

    basal_insulin = torch.as_tensor(basal_insulin, dtype=torch.float32)
    bolus_insulin = torch.as_tensor(bolus_insulin, dtype=torch.float32)
    meal = torch.as_tensor(meal, dtype=torch.float32)

    basal_insulin_phy_processed = physiological_layer(
        basal_insulin.unsqueeze(0), peak_time_basal / interval, 1000
    )
    fig_basal, _ = vis.visualise_insulin_meal_response(
        data_1=basal_insulin,
        data_2=basal_insulin_phy_processed[0, :],
        legend_1="Basal Insulin (pmol/min)",
        legend_2="Basal Insulin Processed",
        ticks_per_day=6,
    )
    fig_basal_name = f"basal_insulin_phy_processed_patient_{p_num}.png"
    fig_basal.savefig(os.path.join(figure_folder, fig_basal_name))
    plt.close(fig_basal)

    bolus_insulin_phy_processed = physiological_layer(
        bolus_insulin.unsqueeze(0), peak_time_bolus / interval, 1000
    )
    fig_bolus, _ = vis.visualise_insulin_meal_response(
        data_1=bolus_insulin,
        data_2=bolus_insulin_phy_processed[0, :],
        legend_1="Bolus Insulin (pmol/min)",
        legend_2="Bolus Insulin Processed",
        ticks_per_day=6,
    )
    fig_bolus_name = f"bolus_insulin_phy_processed_patient_{p_num}.png"
    fig_bolus.savefig(os.path.join(figure_folder, fig_bolus_name))
    plt.close(fig_bolus)

    meal_phy_processed = physiological_layer(
        meal.unsqueeze(0), peak_time_carb / interval, 1000
    )
    fig_meal, _ = vis.visualise_insulin_meal_response(
        data_1=meal,
        data_2=meal_phy_processed[0, :],
        legend_1="Carbohydrate Intake (g)",
        legend_2="Carbohydrate Intake Processed",
        ticks_per_day=6,
    )
    fig_meal_name = f"meal_phy_processed_patient_{p_num}.png"
    fig_meal.savefig(os.path.join(figure_folder, fig_meal_name))
    plt.close(fig_meal)
