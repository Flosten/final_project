"""
This module implements the ablation study for the proposed model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy.ndimage import gaussian_filter1d

import src.Preprocessing as prep
import src.Visualising as vis


# --------------Model for ablation study physiological modeling layer
class Abl_ChannelAttention(nn.Module):
    """
    This class implements a channel attention mechanism that computes attention weights
    based on the context and prior knowledge.

    Parameters:
        hidden_size (int): Size of the hidden layer.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, hidden_size * 2 + 1))

    def forward(self, context, prior_knowledge):
        """
        This function computes the attention weights and weighted context.

        Parameters:
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


class ProposedModel_abl_loss(nn.Module):
    """
    Our proposed model that combines multi-channel LSTM and channel attention.

    Parameters:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        output_size (int): Size of the output layer.
        num_layers (int, optional): Number of LSTM layers. Default is 1.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initializes the ProposedModel_abl_loss class.

        Parameters:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden layer.
            output_size (int): Size of the output layer.
            num_layers (int, optional): Number of LSTM layers. Default is 1.
        """
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
        self.lstm_bolus_insulin = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm_meal = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

        # Channel attention layer
        self.channel_attention = Abl_ChannelAttention(hidden_size)

        # weights for different BG range
        self.weights = nn.Parameter(torch.tensor([2.0, 2.0, 1.0]))

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, cgm, basal_insulin, bolus_insulin, meal):
        """
        This function defines the forward pass of the proposed model.

        Parameters:
            cgm (torch.Tensor): Input tensor for past CGM data.
            basal_insulin (torch.Tensor): Input tensor for basal insulin data.
            bolus_insulin (torch.Tensor): Input tensor for bolus insulin data.
            meal (torch.Tensor): Input tensor for meal data.

        Returns:
            tuple: A tuple containing the output tensor and attention weights.
        """
        out_cgm, _ = self.lstm_cgm(cgm)
        out_basal_insulin, _ = self.lstm_basal_insulin(basal_insulin)
        out_bolus_insulin, _ = self.lstm_bolus_insulin(bolus_insulin)
        out_meal, _ = self.lstm_meal(meal)

        out_cgm = out_cgm[
            :, -1, :
        ]  # Get the last time step output shape (batch_size, hidden_size * 2)
        out_basal_insulin = out_basal_insulin[:, -1, :]
        out_bolus_insulin = out_bolus_insulin[:, -1, :]
        out_meal = out_meal[:, -1, :]

        # prior knowledge
        prior_cgm = torch.ones(cgm.size(0), 1, device=cgm.device)
        prior_insulin = basal_insulin[:, -1, :].mean(dim=1, keepdim=True)
        prior_bolus_insulin = bolus_insulin[:, -1, :].mean(dim=1, keepdim=True)
        prior_meal = meal[:, -1, :].mean(dim=1, keepdim=True)

        # Concatenate the outputs
        context = [out_cgm, out_basal_insulin, out_bolus_insulin, out_meal]
        prior_knowledge = [prior_cgm, prior_insulin, prior_bolus_insulin, prior_meal]

        # channel attention
        weighted_context, atten_weights = self.channel_attention(
            context, prior_knowledge
        )

        # fully connected layer
        output = self.fc(weighted_context)
        return output, atten_weights


# ------------------------physiological layer-------------------------
def min_max_normalization(data):
    """
    Applies min-max normalization to the input data.

    Parameters:
        data (torch.Tensor): Input tensor to be normalized.

    Returns:
        torch.Tensor: Normalized tensor with values between 0 and 1.
    """
    min_value = data.min(dim=1, keepdim=True)[0]
    max_value = data.max(dim=1, keepdim=True)[0]
    normalized_data = (data - min_value) / (max_value - min_value + 1e-8)
    return normalized_data


def physiological_layer(input_seq, lamda, kernel_size):
    """
    Applies a physiological model to the input sequence using a kernel.

    Parameters:
        input_seq (torch.Tensor): Input sequence tensor of shape (batch, seq_len).
        lamda (float): Decay parameter for the kernel.
        kernel_size (int): Size of the kernel to be applied.

    Returns:
        torch.Tensor: Output sequence after applying the physiological model.
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


# ------------------------loss function-------------------------
def smooth_range_weight(bg, low=70.0, high=180.0, k=0.03, w_in=1.0, w_out=2.0):
    """
    Computes a smooth range weight based on the blood glucose (BG) values.

    Parameters:
        bg (torch.Tensor): Blood glucose values.
        low (float): hypo threshold for BG values.
        high (float): hyper threshold for BG values.
        k (float): Scaling factor for the weight.
        w_in (float): Weight for values within the range.
        w_out (float): Weight for values outside the range.

    Returns:
        torch.Tensor: Computed weight based on the BG values.
    """
    d_low = F.relu(low - bg)
    d_high = F.relu(bg - high)
    d = d_low + d_high
    w = w_in + (w_out - w_in) * torch.tanh(k * d)
    return w


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
    """
    Constructs the customized loss function for the proposed model.

    Parameters:
        pred (torch.Tensor): Model predictions.
        truth (torch.Tensor): Ground truth values.
        ori_truth (torch.Tensor): Original ground truth values.
        model (nn.Module): The proposed model.
        alpha (float): Weight for the variance term.
        beta (float): Weight for the regularization term.
        low (float): Hypoglycemia threshold.
        high (float): Hyperglycemia threshold.
        k (float): Scaling factor for the weight.
        w_in (float): Weight for values within the range.
        w_out (float): Weight for values outside the range.

    Returns:
        torch.Tensor: Computed loss value.
    """
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


# ------------------------alarm strategy-------------------------
def alarm_strategy(predictions, hypo_threshold, hyper_threshold):
    """
    Apply the alarm strategy based on the predictions.

    Parameters:
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


# -------------------------ablation study-------------------------
def proposed_model_train_loss(
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
    """
    Train the model that changes the customized loss function to MSE loss (ablation study: loss function).

    Args:
        model (ProposedModel_abl_loss): The proposed model.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.
        alpha (float): Weight for the variance term in the loss function.
        beta (float): Weight for the regularization term in the loss function.
        seq_len_carb_intake (int): Sequence length for carb intake.
        seq_len_basal_insulin (int): Sequence length for basal insulin.
        seq_len_bolus_insulin (int): Sequence length for bolus insulin.
        tp_insulin_basal (float): Peak time for basal insulin response.
        tp_insulin_bolus (float): Peak time for bolus insulin response.
        tp_meal (float): Peak time for meal response.
        interval (int): Time interval for the physiological model.
        device: Device to run the model on.

    Returns:
        model (ProposedModel_abl_loss): The trained model.
        fig (matplotlib.figure.Figure): Figure for the learning curve.
        ax (matplotlib.axes.Axes): Axes for the learning curve.
    """
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
            # loss = com_loss_function(pred, y_norm, y_raw, model, alpha, beta)
            loss = nn.MSELoss()(pred, y_norm)  # using MSE loss for training
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

                # loss_val = com_loss_function(
                #     pred_val, y_norm_val, y_raw_val, model, alpha, beta
                # )
                loss_val = nn.MSELoss()(
                    pred_val, y_norm_val
                )  # using MSE loss for validation
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
    fig, ax = vis.plot_lr_1(train_loss=train_losses, val_loss=val_losses)

    return model, fig, ax


def proposed_model_loss_eval(
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
    Evaluate the model (ablation study: loss function).

    Parameters:
        model (ProposedModel_abl_loss): The trained model.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        tp_insulin_basal (float): Peak time for basal insulin response.
        tp_insulin_bolus (float): Peak time for bolus insulin response.
        tp_meal (float): Peak time for meal response.
        seq_len_carb_intake (int): Sequence length for carb intake.
        seq_len_basal_insulin (int): Sequence length for basal insulin.
        seq_len_bolus_insulin (int): Sequence length for bolus insulin.
        interval (int): Time interval for the physiological model.
        scaler: Scaler used for standardization.
        device: Device to run the model on.
        ticks_per_day (int): Number of ticks per day in the dataset.
        time_steps (int): Total number of time steps in the dataset.

    Returns:
        tuple: A tuple containing predictions, truths, alarms, attention weights, and visualization.
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


# remove physiological modeling layer
def proposed_model_train_phy(
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
    """
    Train the model that removes the physiological modeling layer (ablation study: physio layer).

    Parameters:
        model (ProposedModel_abl_loss): The proposed model.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.
        alpha (float): Weight for the variance term in the loss function.
        beta (float): Weight for the regularization term in the loss function.
        seq_len_carb_intake (int): Sequence length for carb intake.
        seq_len_basal_insulin (int): Sequence length for basal insulin.
        seq_len_bolus_insulin (int): Sequence length for bolus insulin.
        tp_insulin_basal (float): Peak time for basal insulin response.
        tp_insulin_bolus (float): Peak time for bolus insulin response.
        tp_meal (float): Peak time for meal response.
        interval (int): Time interval for the physiological model.
        device: Device to run the model on.

    Returns:
        model (ProposedModel_abl_loss): The trained model.
        fig (matplotlib.figure.Figure): Figure for the learning curve.
        ax (matplotlib.axes.Axes): Axes for the learning curve.
    """
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

            # preprocess the inputs data
            cgm = cgm.unsqueeze(2)  # (batch, seq_len) -> (batch, seq_len, 1)
            meal = meal.unsqueeze(2)
            basal_insulin = basal_insulin.unsqueeze(2)
            bolus_insulin = bolus_insulin.unsqueeze(2)

            # forward pass
            pred, att_weights = model(
                cgm,
                basal_insulin,
                bolus_insulin,
                meal,
            )
            # -------------------------------------------------
            loss = com_loss_function(pred, y_norm, y_raw, model, alpha, beta)
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
                    basal_insulin_val,  # use the last time step
                    bolus_insulin_val,  # use the last time step
                    meal_val,  # use the last time step
                )

                loss_val = com_loss_function(
                    pred_val, y_norm_val, y_raw_val, model, alpha, beta
                )
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


def proposed_model_eval_phy(
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
    Evaluate the model that removes the physiological modeling layer (ablation study: physio layer).

    Parameters:
        model (ProposedModel_abl_loss): The trained model.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        tp_insulin_basal (float): Peak time for basal insulin response.
        tp_insulin_bolus (float): Peak time for bolus insulin response.
        tp_meal (float): Peak time for meal response.
        seq_len_carb_intake (int): Sequence length for carb intake.
        seq_len_basal_insulin (int): Sequence length for basal insulin.
        seq_len_bolus_insulin (int): Sequence length for bolus insulin.
        interval (int): Time interval for the physiological model.
        scaler: Scaler used for standardization.
        device: Device to run the model on.
        ticks_per_day (int): Number of ticks per day in the dataset.
        time_steps (int): Total number of time steps in the dataset.

    Returns:
        tuple: A tuple containing predictions, truths, alarms, attention weights, and visualization.
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
