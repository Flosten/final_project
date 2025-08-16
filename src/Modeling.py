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
        self.lstm_bolus_insulin = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm_meal = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

        # Channel attention layer
        self.channel_attention = ChannelAttention(hidden_size)

        # weights for different BG range
        self.weights = nn.Parameter(torch.tensor([2.0, 2.0, 1.0]))

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, cgm, basal_insulin, bolus_insulin, meal):
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


# # this is the one no basal insulin
# class ProposedModel(nn.Module):
#     """
#     Our proposed model that combines multi-channel LSTM and channel attention.

#     Args:
#         input_size (int): Size of the input features.
#         hidden_size (int): Size of the hidden layer.
#         output_size (int): Size of the output layer.
#         num_layers (int, optional): Number of LSTM layers. Default is 1.
#     """

#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers
#         self.lstm_cgm = nn.LSTM(
#             input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
#         )
#         self.lstm_basal_insulin = nn.LSTM(
#             input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
#         )
#         self.lstm_bolus_insulin = nn.LSTM(
#             input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
#         )
#         self.lstm_meal = nn.LSTM(
#             input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
#         )

#         # Channel attention layer
#         self.channel_attention = ChannelAttention(hidden_size)

#         # weights for different BG range
#         self.weights = nn.Parameter(torch.tensor([5.0, 4.0, 3.0, 1.0]))

#         self.fc = nn.Linear(hidden_size * 2, output_size)

#     def forward(self, cgm, bolus_insulin, meal):
#         """
#         This function defines the forward pass of the proposed model.

#         Args:
#             cgm (torch.Tensor): Input tensor for past CGM data (batch_size, seq_length, input_size).
#             insulin (torch.Tensor): Insulin data (batch_size, seq_length, input_size).
#             meal (torch.Tensor): Input tensor for meal data (batch_size, seq_length, input_size).

#         Returns:
#             tuple: A tuple containing the output tensor and attention weights.
#         """
#         out_cgm, _ = self.lstm_cgm(cgm)
#         # out_basal_insulin, _ = self.lstm_basal_insulin(basal_insulin)
#         out_bolus_insulin, _ = self.lstm_bolus_insulin(bolus_insulin)
#         out_meal, _ = self.lstm_meal(meal)

#         out_cgm = out_cgm[
#             :, -1, :
#         ]  # Get the last time step output shape (batch_size, hidden_size * 2)
#         # out_basal_insulin = out_basal_insulin[:, -1, :]
#         out_bolus_insulin = out_bolus_insulin[:, -1, :]
#         out_meal = out_meal[:, -1, :]

#         # prior knowledge
#         prior_cgm = torch.ones(cgm.size(0), 1, device=cgm.device)
#         # prior_insulin = basal_insulin[:, -1, :].mean(dim=1, keepdim=True)
#         prior_bolus_insulin = bolus_insulin[:, -1, :].mean(dim=1, keepdim=True)
#         prior_meal = meal[:, -1, :].mean(dim=1, keepdim=True)

#         # Concatenate the outputs
#         context = [out_cgm, out_bolus_insulin, out_meal]
#         prior_knowledge = [prior_cgm, prior_bolus_insulin, prior_meal]

#         # channel attention
#         weighted_context, atten_weights = self.channel_attention(
#             context, prior_knowledge
#         )

#         # fully connected layer
#         output = self.fc(weighted_context)
#         return output, atten_weights


def min_max_normalization(data):
    min_value = data.min(dim=1, keepdim=True)[0]
    max_value = data.max(dim=1, keepdim=True)[0]
    normalized_data = (data - min_value) / (max_value - min_value + 1e-8)
    return normalized_data


def physiological_layer(input_seq, lamda, kernel_size):
    """
    Applies the physiological layer to preprocess the insulin and meal data.

    Args:
        input_seq (torch.Tensor): The input sequence (insulin & meal) [batch_size, seq_length].
        lamda (float): The peak time of the physiological response (peak time/ time interval).
        kernel_size (int): The time window size for the physiological model.

    Returns:
        torch.Tensor: The processed output sequence.
    """
    # Create a kernel based on the physiological model
    t = torch.arange(0, kernel_size).float()
    kernel = (t / lamda**2) * torch.exp(-t / lamda)
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, -1).to(
        input_seq.device
    )  # Reshape to (1, 1, kernel_size)

    input_seq = input_seq.float()
    input_seq = input_seq.unsqueeze(1)  # Add channel dimension
    input_seq = torch.clamp(input_seq, min=0)  # Ensure non-negative values

    padding = int(kernel_size) - 1
    input_seq = F.pad(input_seq, (padding, 0), mode="constant", value=0)

    # Apply the convolution
    output_seq = F.conv1d(input_seq, kernel)
    output_seq = output_seq.squeeze(1)  # Remove channel dimension

    output_seq = min_max_normalization(output_seq)

    return output_seq


# def physiological_layer(input_seq, lamda, kernel_size):
#     kernel_size = int(kernel_size)
#     t = torch.arange(kernel_size - 1, -1, -1).float()  # 时间反向
#     kernel = (t / lamda**2) * torch.exp(-t / lamda)
#     kernel = kernel / kernel.sum()
#     kernel = kernel.view(1, 1, -1).to(input_seq.device)

#     input_seq = input_seq.float().unsqueeze(1)  # [B, 1, T]
#     input_seq = F.pad(
#         input_seq, (kernel_size - 1, 0), mode="constant", value=0
#     )  # 左侧填充

#     output_seq = F.conv1d(input_seq, kernel)
#     output_seq = output_seq.squeeze(1)
#     return output_seq


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
                basal_insulin, tp_insulin_basal, kernel_size_basal_insulin
            )
            basal_insulin_preprocessed = basal_insulin_processed.unsqueeze(
                2
            )  # (batch, seq_len) -> (batch, seq_len, 1)

            # bolus_insulin = bolus_insulin.squeeze(2)
            bolus_insulin_processed = physiological_layer(
                bolus_insulin, tp_insulin_bolus, kernel_size_bolus_insulin
            )
            bolus_insulin_preprocessed = bolus_insulin_processed.unsqueeze(
                2
            )  # (batch, seq_len) -> (batch, seq_len, 1)

            # meal = meal.squeeze(2)  # (batch, seq_len, 1)
            meal_processed = physiological_layer(meal, tp_meal, kernel_size_carb_intake)
            meal_preprocessed = meal_processed.unsqueeze(
                2
            )  # (batch, seq_len) -> (batch, seq_len, 1)

            # preprocess the inputs cgm
            cgm = cgm.unsqueeze(2)  # (batch, seq_len) -> (batch, seq_len, 1)

            # forward pass
            pred, att_weights = model(
                cgm,
                basal_insulin_preprocessed,  # use the last time step
                bolus_insulin_preprocessed,  # use the last time step
                meal_preprocessed,  # use the last time step
            )
            # -------------------修改一下loss 先测试作用-------------------
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
                    basal_insulin_val, tp_insulin_basal, kernel_size_basal_insulin
                )
                basal_insulin_preprocessed_val = basal_insulin_processed_val.unsqueeze(
                    2
                )  # (batch, seq_len) -> (batch, seq_len, 1)

                # bolus_insulin_val = bolus_insulin_val.squeeze(2)
                bolus_insulin_processed_val = physiological_layer(
                    bolus_insulin_val, tp_insulin_bolus, kernel_size_bolus_insulin
                )
                bolus_insulin_preprocessed_val = bolus_insulin_processed_val.unsqueeze(
                    2
                )  # (batch, seq_len) -> (batch, seq_len, 1)

                # meal_val = meal_val.squeeze(2)
                meal_processed_val = physiological_layer(
                    meal_val, tp_meal, kernel_size_carb_intake
                )
                meal_preprocessed_val = meal_processed_val.unsqueeze(
                    2
                )  # (batch, seq_len) -> (batch, seq_len, 1)

                # preprocess the inputs cgm
                cgm_val = cgm_val.unsqueeze(
                    2
                )  # (batch, seq_len) -> (batch, seq_len, 1)

                # forward pass
                pred_val, _ = model(
                    cgm_val,
                    basal_insulin_preprocessed_val,  # use the last time step
                    bolus_insulin_preprocessed_val,  # use the last time step
                    meal_preprocessed_val,  # use the last time step
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


# def proposed_model_train_no_basal(
#     model,
#     train_dataloader,
#     val_dataloader,
#     optimizer,
#     epochs,
#     alpha,  # loss weight for variance term
#     beta,  # loss weight for regularization term
#     seq_len_carb_intake,  # sequence length for carb intake
#     seq_len_basal_insulin,  # sequence length for basal insulin
#     seq_len_bolus_insulin,  # sequence length for bolus insulin
#     tp_insulin_basal,  # peak time for basal insulin response
#     tp_insulin_bolus,  # peak time for bolus insulin response
#     tp_meal,  # peak time for meal response
#     interval,  # time interval for the physiological model
#     device,
# ):
#     train_losses = []
#     train_losses_2 = []  # for debugging and monitoring
#     val_losses = []
#     val_losses_2 = []  # for debugging and monitoring
#     model = model.to(device)
#     # physiological layer for insulin and meal
#     tp_insulin_basal = tp_insulin_basal / interval  # convert peak time to time steps
#     tp_insulin_bolus = tp_insulin_bolus / interval  # convert peak time to time steps
#     tp_meal = tp_meal / interval  # convert peak time to time steps
#     # kernel size for basal insulin and bolus insulin
#     kernel_size_basal_insulin = seq_len_basal_insulin / interval
#     kernel_size_bolus_insulin = seq_len_bolus_insulin / interval
#     kernel_size_carb_intake = seq_len_carb_intake / interval

#     # training loop
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         train_loss_2 = 0.0  # for debugging and monitoring
#         all_predictions = []
#         all_truths = []
#         for x, y_norm, y_raw in tqdm.tqdm(
#             train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
#         ):
#             optimizer.zero_grad()
#             cgm = x["cgm"]
#             meal = x["carb_intake"]
#             basal_insulin = x["current_basal"]
#             bolus_insulin = x["current_bolus"]
#             cgm = cgm.to(model.weights.device)
#             basal_insulin = basal_insulin.to(model.weights.device)
#             bolus_insulin = bolus_insulin.to(model.weights.device)
#             meal = meal.to(model.weights.device)
#             y_norm = y_norm.to(model.weights.device)
#             y_raw = y_raw.to(model.weights.device)

#             # Apply physiological layer to insulin and meal
#             # basal_insulin = basal_insulin.squeeze(2)
#             basal_insulin_processed = physiological_layer(
#                 basal_insulin, tp_insulin_basal, kernel_size_basal_insulin
#             )
#             basal_insulin_preprocessed = basal_insulin_processed.unsqueeze(
#                 2
#             )  # (batch, seq_len) -> (batch, seq_len, 1)

#             # bolus_insulin = bolus_insulin.squeeze(2)
#             bolus_insulin_processed = physiological_layer(
#                 bolus_insulin, tp_insulin_bolus, kernel_size_bolus_insulin
#             )
#             bolus_insulin_preprocessed = bolus_insulin_processed.unsqueeze(
#                 2
#             )  # (batch, seq_len) -> (batch, seq_len, 1)

#             # meal = meal.squeeze(2)  # (batch, seq_len, 1)
#             meal_processed = physiological_layer(meal, tp_meal, kernel_size_carb_intake)
#             meal_preprocessed = meal_processed.unsqueeze(
#                 2
#             )  # (batch, seq_len) -> (batch, seq_len, 1)

#             # preprocess the inputs cgm
#             cgm = cgm.unsqueeze(2)  # (batch, seq_len) -> (batch, seq_len, 1)

#             # forward pass
#             pred, att_weights = model(
#                 cgm,
#                 # basal_insulin_preprocessed,  # use the last time step
#                 bolus_insulin_preprocessed,  # use the last time step
#                 meal_preprocessed,  # use the last time step
#             )
#             # -------------------修改一下loss 先测试作用-------------------
#             # loss = com_loss_function(pred, y_norm, y_raw, model, alpha, beta)
#             loss = nn.MSELoss()(pred, y_norm)  # using MSE loss for training
#             loss_2 = F.mse_loss(pred, y_norm)  # for debugging and monitoring
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             train_loss_2 += loss_2.item()  # for debugging and monitoring
#         train_loss /= len(train_dataloader)
#         train_losses.append(train_loss)
#         train_loss_2 /= len(train_dataloader)
#         train_losses_2.append(train_loss_2)

#         print(
#             f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train MSE: {train_loss_2:.4f}"
#         )

#         # validation loop
#         model.eval()
#         val_loss = 0.0
#         val_loss_2 = 0.0  # for debugging and monitoring
#         with torch.no_grad():
#             for x_val, y_norm_val, y_raw_val in val_dataloader:
#                 # extract the inputs and targets
#                 cgm_val = x_val["cgm"]
#                 meal_val = x_val["carb_intake"]
#                 basal_insulin_val = x_val["current_basal"]
#                 bolus_insulin_val = x_val["current_bolus"]

#                 cgm_val = cgm_val.to(model.weights.device)
#                 basal_insulin_val = basal_insulin_val.to(model.weights.device)
#                 bolus_insulin_val = bolus_insulin_val.to(model.weights.device)
#                 meal_val = meal_val.to(model.weights.device)
#                 y_norm_val = y_norm_val.to(model.weights.device)
#                 y_raw_val = y_raw_val.to(model.weights.device)

#                 # Apply physiological layer to insulin and meal
#                 # basal_insulin_val = basal_insulin_val.squeeze(2)
#                 basal_insulin_processed_val = physiological_layer(
#                     basal_insulin_val, tp_insulin_basal, kernel_size_basal_insulin
#                 )
#                 basal_insulin_preprocessed_val = basal_insulin_processed_val.unsqueeze(
#                     2
#                 )  # (batch, seq_len) -> (batch, seq_len, 1)

#                 # bolus_insulin_val = bolus_insulin_val.squeeze(2)
#                 bolus_insulin_processed_val = physiological_layer(
#                     bolus_insulin_val, tp_insulin_bolus, kernel_size_bolus_insulin
#                 )
#                 bolus_insulin_preprocessed_val = bolus_insulin_processed_val.unsqueeze(
#                     2
#                 )  # (batch, seq_len) -> (batch, seq_len, 1)

#                 # meal_val = meal_val.squeeze(2)
#                 meal_processed_val = physiological_layer(
#                     meal_val, tp_meal, kernel_size_carb_intake
#                 )
#                 meal_preprocessed_val = meal_processed_val.unsqueeze(
#                     2
#                 )  # (batch, seq_len) -> (batch, seq_len, 1)

#                 # preprocess the inputs cgm
#                 cgm_val = cgm_val.unsqueeze(
#                     2
#                 )  # (batch, seq_len) -> (batch, seq_len, 1)

#                 # forward pass
#                 pred_val, _ = model(
#                     cgm_val,
#                     # basal_insulin_preprocessed_val,  # use the last time step
#                     bolus_insulin_preprocessed_val,  # use the last time step
#                     meal_preprocessed_val,  # use the last time step
#                 )

#                 # loss_val = com_loss_function(
#                 #     pred_val, y_norm_val, y_raw_val, model, alpha, beta
#                 # )
#                 loss_val = nn.MSELoss()(
#                     pred_val, y_norm_val
#                 )  # using MSE loss for validation
#                 loss_val_2 = F.mse_loss(
#                     pred_val, y_norm_val
#                 )  # for debugging and monitoring
#                 val_loss += loss_val.item()
#                 val_loss_2 += loss_val_2.item()  # for debugging and monitoring
#         val_loss /= len(val_dataloader)
#         val_losses.append(val_loss)
#         val_loss_2 /= len(val_dataloader)
#         val_losses_2.append(val_loss_2)
#         print(
#             f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val MSE Loss: {val_loss_2:.4f}"
#         )

#     # print the learning curves
#     fig, ax = vis.plot_lr_2(train_losses, train_losses_2, val_losses, val_losses_2)

#     return model, fig, ax


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
                basal_insulin, tp_insulin_basal, kernel_size_basal_insulin
            )
            basal_insulin_preprocessed = basal_insulin_processed.unsqueeze(2)

            # bolus_insulin = bolus_insulin.squeeze(2)
            bolus_insulin_processed = physiological_layer(
                bolus_insulin, tp_insulin_bolus, kernel_size_bolus_insulin
            )
            bolus_insulin_preprocessed = bolus_insulin_processed.unsqueeze(2)

            # meal = meal.squeeze(2)
            meal_processed = physiological_layer(meal, tp_meal, kernel_size_carb_intake)
            meal_preprocessed = meal_processed.unsqueeze(2)

            # preprocess the inputs cgm
            cgm = cgm.unsqueeze(2)  # (batch, seq_len) -> (batch, seq_len, 1)

            # forward pass
            pred, attn_weights = model(
                cgm,
                basal_insulin_preprocessed,  # use the last time step
                bolus_insulin_preprocessed,  # use the last time step
                meal_preprocessed,  # use the last time step
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
    predictions = gaussian_filter1d(predictions, sigma=2)  # apply Gaussian filter
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


# def proposed_model_eval_no_basal(
#     model,
#     test_dataloader,
#     tp_insulin_basal,
#     tp_insulin_bolus,
#     tp_meal,
#     seq_len_carb_intake,
#     seq_len_basal_insulin,
#     seq_len_bolus_insulin,
#     interval,
#     scaler,
#     device,
#     ticks_per_day,
#     time_steps,
# ):
#     """
#     Evaluate the proposed model on the test dataset.

#     Args:
#         model (ProposedModel): The trained model.
#         test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
#         tp_insulin (float): Peak time for insulin response.
#         tp_meal (float): Peak time for meal response.
#         kernel_size (int): Time window size for the physiological model.

#     Returns:
#         list: List of predictions for the test dataset.
#     """
#     model = model.to(device)
#     model.eval()
#     predictions = []
#     truths = []
#     attention_weights = []
#     # physiological layer for insulin and meal
#     tp_insulin_basal = tp_insulin_basal / interval  # convert peak time to time steps
#     tp_insulin_bolus = tp_insulin_bolus / interval  # convert peak time to time steps
#     tp_meal = tp_meal / interval  # convert peak time to time steps

#     # kernel size for basal insulin and bolus insulin
#     kernel_size_basal_insulin = seq_len_basal_insulin / interval
#     kernel_size_bolus_insulin = seq_len_bolus_insulin / interval
#     kernel_size_carb_intake = seq_len_carb_intake / interval

#     with torch.no_grad():
#         for x, y_norm, y_raw in test_dataloader:
#             # extract the inputs and targets
#             cgm = x["cgm"]
#             meal = x["carb_intake"]
#             basal_insulin = x["current_basal"]
#             bolus_insulin = x["current_bolus"]

#             # move the inputs and targets to the device
#             cgm = cgm.to(model.weights.device)
#             basal_insulin = basal_insulin.to(model.weights.device)
#             bolus_insulin = bolus_insulin.to(model.weights.device)
#             meal = meal.to(model.weights.device)
#             y_norm = y_norm.to(model.weights.device)
#             y_raw = y_raw.to(model.weights.device)

#             # Apply physiological layer to insulin and meal
#             # basal_insulin = basal_insulin.squeeze(2)
#             basal_insulin_processed = physiological_layer(
#                 basal_insulin, tp_insulin_basal, kernel_size_basal_insulin
#             )
#             basal_insulin_preprocessed = basal_insulin_processed.unsqueeze(2)

#             # bolus_insulin = bolus_insulin.squeeze(2)
#             bolus_insulin_processed = physiological_layer(
#                 bolus_insulin, tp_insulin_bolus, kernel_size_bolus_insulin
#             )
#             bolus_insulin_preprocessed = bolus_insulin_processed.unsqueeze(2)

#             # meal = meal.squeeze(2)
#             meal_processed = physiological_layer(meal, tp_meal, kernel_size_carb_intake)
#             meal_preprocessed = meal_processed.unsqueeze(2)

#             # preprocess the inputs cgm
#             cgm = cgm.unsqueeze(2)  # (batch, seq_len) -> (batch, seq_len, 1)

#             # forward pass
#             pred, attn_weights = model(
#                 cgm,
#                 # basal_insulin_preprocessed,  # use the last time step
#                 bolus_insulin_preprocessed,  # use the last time step
#                 meal_preprocessed,  # use the last time step
#             )
#             predictions.append(pred)
#             truths.append(y_norm)
#             attention_weights.append(attn_weights)
#     predictions = torch.cat(predictions, dim=0).cpu().numpy().flatten().tolist()
#     truths = torch.cat(truths, dim=0).cpu().numpy().flatten().tolist()
#     attention_weights = torch.cat(attention_weights, dim=0).cpu().numpy()

#     # inverse standardization
#     predictions = prep.data_inverse_standardization(predictions, scaler)
#     truths = prep.data_inverse_standardization(truths, scaler)

#     # visualize the predictions
#     fig, ax = vis.plot_pred_visualisation(
#         predictions, truths, ticks_per_day, time_steps
#     )

#     # plot thresholds
#     fig_threshold, ax_threshold = vis.plot_pred_threshold_visualisation(
#         predictions, truths, ticks_per_day, time_steps
#     )

#     # apply alarm strategy
#     hypo_threshold = 70
#     hyper_threshold = 180
#     pred_alarms = alarm_strategy(predictions, hypo_threshold, hyper_threshold)
#     truth_alarms = alarm_strategy(truths, hypo_threshold, hyper_threshold)

#     return (
#         predictions,
#         truths,
#         pred_alarms,
#         truth_alarms,
#         attention_weights,
#         fig,
#         ax,
#         fig_threshold,
#         ax_threshold,
#     )


# def identify_ranges(bg_values, weights):
#     """
#     Identify the ranges of blood glucose values and apply weights.

#     Args:
#         bg_values (torch.Tensor): Blood glucose values (batch_size, 1).
#         weights (torch.Tensor): Weights for different ranges.

#     Returns:
#         torch.Tensor: Weights corresponding to the identified ranges (batch_size, 1).
#     """

#     # Identify ranges based on the bg_values and thresholds
#     con0 = (bg_values < 70).long()
#     con1 = (bg_values > 180).long()
#     con2 = ((bg_values >= 70) & (bg_values <= 180)).long()

#     # sum the weighted conditions
#     ranges = con0 * 0 + con1 * 1 + con2 * 2
#     # Apply weights to the ranges
#     get_weights = weights[ranges]

#     return get_weights


# def com_loss_function(pred, truth, ori_truth, model, alpha=0.5, beta=0.2):
#     """
#     Custom loss function for the proposed model.

#     Args:
#         pred (torch.Tensor): Model predictions (batch_size, 1).
#         truth (torch.Tensor): Ground truth values (batch_size, 1).
#         model (ProposedModel): The model.
#         alpha (float): Weight for the variance term.
#         beta (float): Weight for the regularization term.

#     Returns:
#         torch.Tensor: Computed loss value.
#     """
#     # set a fixed weight for the ranges
#     fixed_weights = torch.tensor([2.0, 1.2, 1.5], device=model.weights.device)

#     # w_t = identify_ranges(bg_values=ori_truth, weights=model.weights)  # (batch_size, 1)
#     w_t = identify_ranges(bg_values=ori_truth, weights=fixed_weights).unsqueeze(
#         1
#     )  # (batch_size, 1)

#     # calculate the weighted mean squared error
#     l_prd = (w_t * (pred - truth) ** 2).mean()  # (batch_size, 1)

#     # calculate the weightes variance
#     delta_pred = pred[1:, :] - pred[:-1, :]  # (batch_size-1, 1)
#     delta_truth = truth[1:, :] - truth[:-1, :]  # (batch_size-1, 1)
#     l_var = (w_t[1:, :] * (delta_pred - delta_truth) ** 2).mean()  # (batch_size, 1)

#     # calculate the regularization term
#     w_prior = torch.tensor([2.0, 1.2, 1.5], device=model.weights.device)
#     l_reg = F.mse_loss(model.weights, w_prior)

#     # calculate the final loss
#     total_loss = l_prd + alpha * l_var + beta * l_reg

#     return total_loss


# # ---- 平滑权重：随偏离[low, high]的距离连续增加 ----
# def smooth_range_weight(bg, low=70.0, high=180.0, k=0.03, w_in=1.0, w_out=2.0):
#     """
#     bg: (batch, 1)
#     返回: (batch, 1) in [w_in, w_out)
#     """
#     d_low = F.relu(low - bg)
#     d_high = F.relu(bg - high)
#     d = d_low + d_high
#     w = w_in + (w_out - w_in) * torch.tanh(k * d)
#     return w


# # ---- Huber 与 非对称 Huber ----
# def huber_loss(e, delta=10.0):
#     """
#     e = pred - truth
#     """
#     abs_e = torch.abs(e)
#     quad = (
#         0.5
#         * torch.minimum(abs_e, torch.tensor(delta, device=e.device, dtype=e.dtype)) ** 2
#     )
#     lin = delta * (
#         abs_e
#         - torch.minimum(abs_e, torch.tensor(delta, device=e.device, dtype=e.dtype))
#     )
#     return quad + lin


# def asymmetric_huber_loss(e, delta=10.0, over_ratio=1.2, under_ratio=1.0):
#     """
#     e = pred - truth
#     """
#     base = huber_loss(e, delta=delta)
#     factor = torch.where(
#         e > 0,
#         torch.tensor(over_ratio, device=e.device, dtype=e.dtype),
#         torch.tensor(under_ratio, device=e.device, dtype=e.dtype),
#     )
#     return factor * base


# # ---- 如果你仍需要 identify_ranges，可保留；这里不再使用它来加权 ----


# def com_loss_function(
#     pred,
#     truth,
#     ori_truth,
#     model,
#     alpha,
#     beta,
#     # 平滑加权超参
#     low=70.0,
#     high=180.0,
#     k=0.03,
#     w_in=1.0,
#     w_out=2.2,
#     # Huber/非对称超参
#     use_asymmetric=True,
#     delta=10.0,
#     over_ratio=1.4,
#     under_ratio=1.0,
#     lam_range=0.8,
# ):
#     """
#     pred/truth/ori_truth: (batch, 1)
#     alpha: 变化一致性项权重
#     beta:  正则项权重
#     """

#     # 1) 平滑权重（按真值 ori_truth 加权），并做批内归一化以保持梯度尺度
#     w_t = smooth_range_weight(
#         ori_truth, low=low, high=high, k=k, w_in=w_in, w_out=w_out
#     )
#     w_t = w_t / (w_t.mean().clamp_min(1e-6))

#     # 2) 预测误差：Huber 或 非对称 Huber（替代原 MSE）
#     err = pred - truth
#     if use_asymmetric:
#         per_elem = asymmetric_huber_loss(
#             err, delta=delta, over_ratio=over_ratio, under_ratio=under_ratio
#         )
#     else:
#         per_elem = huber_loss(err, delta=delta)
#     l_prd = (w_t * per_elem).mean()

#     # 3) 变化一致性：同样用（较小 δ 的）Huber，更稳；权重取相邻两步的平均
#     d_pred = pred[1:, :] - pred[:-1, :]
#     d_truth = truth[1:, :] - truth[:-1, :]
#     d_err = d_pred - d_truth
#     w_pair = 0.5 * (w_t[1:, :] + w_t[:-1, :])
#     l_var = (w_pair * huber_loss(d_err, delta=max(5.0, delta / 2))).mean()

#     # 4) 权重正则（保留你的写法）
#     w_prior = torch.tensor(
#         [2.0, 1.2, 1.5], device=model.weights.device, dtype=model.weights.dtype
#     )
#     l_reg = F.mse_loss(model.weights, w_prior)

#     # over high
#     over_high = F.relu(pred - high) * (truth <= high)
#     range_penalty = (over_high**2).mean() * lam_range

#     total_loss = l_prd + alpha * l_var + beta * l_reg + range_penalty
#     # total_loss = l_prd + range_penalty
#     return total_loss


# ---- 平滑权重：随偏离[low, high]的距离连续增加 ----
def smooth_range_weight(bg, low=70.0, high=180.0, k=0.03, w_in=1.0, w_out=2.0):
    """
    bg: (batch, 1)
    返回: (batch, 1) in [w_in, w_out)
    """
    d_low = F.relu(low - bg)
    d_high = F.relu(bg - high)
    d = d_low + d_high
    w = w_in + (w_out - w_in) * torch.tanh(k * d)
    return w


# ---- Huber 与 非对称 Huber ----
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


# ---- 如果你仍需要 identify_ranges，可保留；这里不再使用它来加权 ----


def com_loss_function(
    pred,
    truth,
    ori_truth,
    model,
    alpha,
    beta,
    # 平滑加权超参
    low=70.0,
    high=180.0,
    k=0.03,
    w_in=1.0,
    w_out=2.5,
    lam_range=1.4,
):
    """
    pred/truth/ori_truth: (batch, 1)
    alpha: 变化一致性项权重
    beta:  正则项权重
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

    # over high
    over_high = F.relu(pred - high) * (truth <= high)
    range_penalty = (over_high**2).mean() * lam_range

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
