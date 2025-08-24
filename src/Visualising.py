"""
This module provides functions to visualise various types of data,
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def create_xticks(data, time_steps, ticks_per_day):
    """
    Create x-ticks for the plot based on the data length and ticks per day.

    Parameters:
        data (list): List of data points to be visualised.
        time_steps (int): Number of time steps per tick.
        ticks_per_day (int): Number of ticks per day.

    Returns:
        xticks (np.ndarray): Array of x-tick positions.
        xtick_labels (list): List of formatted x-tick labels.
    """
    num = len(data)

    tick_interval = max(1, 1440 // max(1, ticks_per_day))  # 1440 minutes in a day
    sample_interval = max(
        1, tick_interval // max(1, time_steps)
    )  # Ensure at least one sample per tick

    xticks = np.arange(0, num, sample_interval)  # x-ticks at every sample_interval

    xtick_labels = []
    for i in xticks:
        minutes = (i * time_steps) % 1440
        hours = minutes // 60
        minute = minutes % 60
        xtick_labels.append(f"{hours:02d}:{minute:02d}")

    return xticks, xtick_labels


def visualise_data(data, ticks_per_day, y_label, time_steps=1):
    """
    Visualise the data using a line plot.

    Parameters:
        data (list): List of data points to be visualised.
        ticks_per_day (int): Number of ticks per day for x-axis.
        y_label (str): Label for the y-axis.
        time_steps (int): Number of time steps per tick.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data, color="#3167DB", linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)

    # Create x-ticks based on the data length and ticks per day
    xticks, xtick_labels = create_xticks(data, time_steps, ticks_per_day)
    ax.set_xticks(xticks, xtick_labels)
    # ax.set_title("Data Visualisation")
    # ax.legend()

    return fig, ax


def visualise_insulin_meal_response(
    data_1, data_2, legend_1, legend_2, ticks_per_day, time_steps=5
):
    """
    Visualise the insulin and meal response data using a line plot.

    Parameters:
        data_1 (list): List of insulin response data points.
        data_2 (list): List of meal response data points.
        legend_1 (str): Legend label for insulin response.
        legend_2 (str): Legend label for meal response.
        ticks_per_day (int): Number of ticks per day for x-axis.
        time_steps (int): Number of time steps per tick.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(data_1, label=legend_1, color="#D72E2F", linewidth=1.5, alpha=0.8)
    ax1.set_ylabel(legend_1, color="#D72E2F")
    ax1.tick_params(axis="y", labelcolor="#D72E2F")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(data_2, label=legend_2, color="#4363bf", linewidth=1.5, alpha=1.0)
    ax2.set_ylabel(legend_2, color="#4363b8", alpha=1.0)
    ax2.tick_params(axis="y", labelcolor="#4363b8", width=1.5)

    ax1.set_xlabel("Time")

    # Create x-ticks based on the data length and ticks per day
    xticks, xtick_labels = create_xticks(data_1, time_steps, ticks_per_day)
    ax1.set_xticks(xticks, xtick_labels)

    return fig, ax1


def visualise_preds_comparison(
    pred1,
    pred2,
    truth,
    ticks_per_day,
    label1,
    label2,
    time_steps=5,
):
    """
    Visualise the predicted values from model 1 and model 2 against the true values.

    Parameters:
        pred1 (list): List of predicted values from the model 1.
        pred2 (list): List of predicted values from the  model 2.
        truth (list): List of true values.
        ticks_per_day (int): Number of ticks per day for x-axis.
        label1 (str): Label for the model 1 predictions.
        label2 (str): Label for the  model 2 predictions.
        time_steps (int): Number of time steps per tick.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    base_len = len(pred1)
    prop_len = len(pred2)
    truth_len = len(truth)

    min_len = min(base_len, prop_len, truth_len)

    baseline_pred = pred1[-min_len:]
    proposed_pred = pred2[-min_len:]
    truth = truth[-min_len:]

    min_x = min(min(baseline_pred), min(proposed_pred), min(truth))
    max_x = max(max(baseline_pred), max(proposed_pred), max(truth))
    y_min = min_x - 15
    y_max = max_x + 30

    start_idx = base_len - min_len
    time = np.arange(start_idx, start_idx + min_len)
    xticks, xtick_labels = create_xticks(time, time_steps, ticks_per_day)

    ax.plot(
        baseline_pred,
        label=label1,
        color="#4C9AFF",
        linestyle="-.",
        linewidth=1.2,
        alpha=0.9,
    )

    ax.plot(
        proposed_pred,
        label=label2,
        color="#AB47BC",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )

    ax.plot(
        truth,
        label="True Values",
        color="#f58231",
        linestyle="-",
        linewidth=1.3,
        alpha=0.9,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("BG Level (mg/dL)")
    ax.legend(framealpha=0.5)
    ax.set_ylim(y_min, y_max)  # Set y-axis limits based on data range
    ax.set_xticks(xticks, xtick_labels)

    return fig, ax


def visualise_preds_comparison_threshold(
    pred1,
    pred2,
    truth,
    ticks_per_day,
    label1,
    label2,
    time_steps=5,
    threshold_1=180,
    threshold_2=70,
):
    """
    Visualise the predicted values from  model 1 and model 2 against the true values,
    including hyper and hypo threshold lines.

    Parameters:
        pred1 (list): List of predicted values from the model 1.
        pred2 (list): List of predicted values from the model 2.
        truth (list): List of true values.
        ticks_per_day (int): Number of ticks per day for x-axis.
        label1 (str): Label for the model 1 predictions.
        label2 (str): Label for the model 2 predictions.
        time_steps (int): Number of time steps per tick.
        threshold_1 (int): Hyper threshold value for BG level.
        threshold_2 (int): Hypo threshold value for BG level.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    base_len = len(pred1)
    prop_len = len(pred2)
    truth_len = len(truth)

    min_len = min(base_len, prop_len, truth_len)

    baseline_pred = pred1[-min_len:]
    proposed_pred = pred2[-min_len:]
    truth = truth[-min_len:]

    min_x = min(min(baseline_pred), min(proposed_pred), min(truth))
    max_x = max(max(baseline_pred), max(proposed_pred), max(truth))
    y_min = min_x - 15
    y_max = max_x + 30

    start_idx = base_len - min_len
    time = np.arange(start_idx, start_idx + min_len)
    xticks, xtick_labels = create_xticks(time, time_steps, ticks_per_day)

    # plot threshold lines
    ax.axhline(
        y=threshold_1,
        color="#800000",
        linestyle="--",
        label="Hyper Threshold (180 mg/dL)",
        linewidth=1.2,
    )
    ax.axhline(
        y=threshold_2,
        color="#800000",
        linestyle="-.",
        label="Hypo Threshold (70 mg/dL)",
        linewidth=1.2,
    )

    # plot predictions and truth values
    ax.plot(
        baseline_pred,
        label=label1,
        color="#4C9AFF",
        linestyle="-.",
        linewidth=1.2,
        alpha=0.9,
    )

    ax.plot(
        proposed_pred,
        label=label2,
        color="#AB47BC",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )

    ax.plot(
        truth,
        label="True Values",
        color="#f58231",
        linestyle="-",
        linewidth=1.3,
        alpha=0.9,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("BG Level (mg/dL)")
    ax.legend(framealpha=0.5, ncol=3)
    ax.set_ylim(y_min, y_max)  # Set y-axis limits based on data range
    ax.set_xticks(xticks, xtick_labels)

    return fig, ax


def plot_lr_1(train_loss, val_loss):
    """
    Plot training and validation loss over epochs.

    Parameters:
        train_loss (list): List of training loss values per epoch.
        val_loss (list): List of validation loss values per epoch.

    Returns:
    fig, ax: Matplotlib figure and axes objects.
    """
    xticks_num = 10

    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(train_loss) + 1)

    epochs_num = len(train_loss)
    num = epochs_num // xticks_num
    if num < 1:
        num = 1
    xticks = np.arange(0, epochs_num, num)

    ax.plot(epochs, train_loss, label="Training Loss", color="blue")
    ax.plot(epochs, val_loss, label="Validation Loss", color="orange")

    ax.set_xticks(xticks)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    ax.legend()

    return fig, ax


def plot_lr_2(train_loss_1, train_loss_2, val_loss_1, val_loss_2):
    """
    Plot training and validation loss for two different loss functions.

    Parameters:
        train_loss_1 (list): List of training loss values for the custom loss function.
        train_loss_2 (list): List of training loss values for the MSE loss function.
        val_loss_1 (list): List of validation loss values for the custom loss function.
        val_loss_2 (list): List of validation loss values for the MSE loss function.

    Returns:
    fig, ax: Matplotlib figure and axes objects.
    """
    xticks_num = 10

    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(train_loss_1) + 1)

    epochs_num = len(train_loss_1)
    num = epochs_num // xticks_num
    if num < 1:
        num = 1
    xticks = np.arange(0, epochs_num, num)

    ax.plot(epochs, train_loss_1, label="Custom Loss (Train)", color="#e41a1c")
    ax.plot(epochs, train_loss_2, label="MSE Loss (Train)", color="#FDAE6B")
    ax.plot(epochs, val_loss_1, label="Custom Loss (Val)", color="#4C9AFF")
    ax.plot(epochs, val_loss_2, label="MSE Loss (Val)", color="#A463F2")

    ax.set_xticks(xticks)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    ax.legend()

    return fig, ax


def plot_pred_visualisation(pred, truth, ticks_per_day, time_steps):
    """
    Plot the predicted BG levels against the true BG levels.

    Parameters:
        pred (list): List of predicted BG levels.
        truth (list): List of true BG levels.
        ticks_per_day (int): Number of ticks per day for x-axis.
        time_steps (int): Number of time steps per tick.

    Returns:
    fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    min_x = min(min(pred), min(truth))
    max_x = max(max(pred), max(truth))
    y_min = min_x - 15
    y_max = max_x + 15

    ax.plot(
        pred,
        label="Prediction",
        color="#4C9AFF",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.plot(
        truth, label="Truth", color="#AF32AF", linestyle="-", linewidth=1.2, alpha=0.85
    )  # "#A21FAB" is a dark magenta color

    ax.set_xlabel("Time")
    ax.set_ylabel("BG Level (mg/dL)")

    ax.legend(framealpha=0.5)
    # ax.set_ylim(20, 240)  # Set y-axis limits for better visibility
    ax.set_ylim(y_min, y_max)  # Set y-axis limits based on data range

    # Create x-ticks based on the data length and ticks per day
    xticks, xtick_labels = create_xticks(pred, time_steps, ticks_per_day)
    ax.set_xticks(xticks, xtick_labels, rotation=45)

    return fig, ax


def plot_pred_threshold_visualisation(
    pred, truth, ticks_per_day, time_steps, threshold_1=180, threshold_2=70
):
    """
    Plot the predicted BG levels against the true BG levels.

    Parameters:
        pred (list): List of predicted BG levels.
        truth (list): List of true BG levels.
        ticks_per_day (int): Number of ticks per day for x-axis.
        time_steps (int): Number of time steps per tick.
        threshold_1 (int): Hyper threshold value for BG level.
        threshold_2 (int): Hypo threshold value for BG level.

    Returns:
    fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    min_x = min(min(pred), min(truth))
    max_x = max(max(pred), max(truth))
    y_min = min_x - 15
    y_max = max_x + 15

    # Add threshold lines
    ax.axhline(
        y=threshold_1,
        color="red",
        linestyle="--",
        label="Hyper Threshold (180 mg/dL)",
    )
    ax.axhline(
        y=threshold_2,
        color="red",
        linestyle="-.",
        label="Hypo Threshold (70 mg/dL)",
    )

    ax.plot(
        pred,
        label="Prediction",
        color="#4C9AFF",  # "#56B4E9"
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.plot(
        truth,
        label="Truth",
        color="#AF32AF",  # "#AF32AF"
        linestyle="-",
        linewidth=1.2,
        alpha=0.85,
    )  # "#A21FAB" is a dark magenta color

    ax.set_xlabel("Time")
    ax.set_ylabel("BG Level (mg/dL)")

    ax.legend(framealpha=0.5)
    # ax.set_ylim(15, 260)  # Set y-axis limits for better visibility
    ax.set_ylim(y_min, y_max)  # Set y-axis limits based on data range

    # Create x-ticks based on the data length and ticks per day
    xticks, xtick_labels = create_xticks(pred, time_steps, ticks_per_day)
    ax.set_xticks(xticks, xtick_labels, rotation=45)

    return fig, ax


def compute_q(arr):
    """
    Compute the mean, 25th percentile (Q1), and 75th percentile (Q3) of the input array.

    Parameters:
        arr (np.ndarray): Input array with shape (n_samples, n_features).

    Returns:
        mean (np.ndarray): Mean of the input array along the first axis.
        q_low (np.ndarray): 25th percentile of the input array along the first axis.
        q_high (np.ndarray): 75th percentile of the input array along the first axis.
    """
    mean = np.nanmean(arr, axis=0)
    q_low = np.nanpercentile(arr, 25, axis=0)
    q_high = np.nanpercentile(arr, 75, axis=0)

    return mean, q_low, q_high


def plot_errors(
    baseline_error_list,
    proposed_error_list2,
    label1,
    label2,
    time_steps=5,
    ticks_per_day=2,
):
    """
    Plot the absolute errors of baseline and proposed models over time for group studies.

    Parameters:
        baseline_error_list (list): List of absolute errors for the model 1.
        proposed_error_list2 (list): List of absolute errors for the model 2.
        label1 (str): Label for the model 1.
        label2 (str): Label for the model 2.
        time_steps (int): Number of time steps per tick.
        ticks_per_day (int): Number of ticks per day.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    baseline_arr_full = np.vstack(baseline_error_list)
    proposed_arr_full = np.vstack(proposed_error_list2)

    len_1 = baseline_arr_full.shape[1]
    len_2 = proposed_arr_full.shape[1]

    min_len = min(len_1, len_2)

    baseline_arr = baseline_arr_full[:, -min_len:]
    proposed_arr = proposed_arr_full[:, -min_len:]

    baseline_mean, baseline_q_low, baseline_q_high = compute_q(baseline_arr)
    proposed_mean, proposed_q_low, proposed_q_high = compute_q(proposed_arr)

    start_idx = len_1 - min_len
    time = np.arange(start_idx, start_idx + min_len)
    xticks, xtick_labels = create_xticks(time, time_steps, ticks_per_day)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(time, baseline_mean, lw=1, color="#A463F2", label=label1, alpha=0.9)
    ax.fill_between(
        time,
        baseline_q_low,
        baseline_q_high,
        color="#A463F2",
        alpha=0.2,
    )
    ax.plot(time, proposed_mean, lw=1, color="#1E90EF", label=label2, alpha=0.9)
    ax.fill_between(
        time,
        proposed_q_low,
        proposed_q_high,
        color="#1E90EF",
        alpha=0.2,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Absolute Error (mg/dL)")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.legend()

    return fig, ax


def plot_interpretability(
    cgm_attr,
    carb_attr,
    basal_attr,
    bolus_attr,
    cgm_values,
    carb_values,
    basal_values,
    bolus_values,
    feature_names=["CGM", "Carb Intake", "Insulin Basal", "Insulin Bolus"],
):
    """
    Plot the input feature attribution values against their corresponding input feature values.

    Parameters:
        cgm_attr (np.ndarray): CGM input feature attribution values.
        carb_attr (np.ndarray): Carb intake input feature attribution values.
        basal_attr (np.ndarray): Insulin basal input feature attribution values.
        bolus_attr (np.ndarray): Insulin bolus input feature attribution values.
        cgm_values (np.ndarray): CGM input feature values.
        carb_values (np.ndarray): Carb intake input feature values.
        basal_values (np.ndarray): Insulin basal input feature values.
        bolus_values (np.ndarray): Insulin bolus input feature values.
        feature_names (list): List of feature names for the y-axis labels.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    define_cmap = LinearSegmentedColormap.from_list(
        "attr_cmap", ["#008DFF", "#0E5BDC", "#A11FAB", "#EB0079", "#FF0055"], N=256
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    var_attr = [
        cgm_attr.reshape(-1),
        carb_attr.reshape(-1),
        basal_attr.reshape(-1),
        bolus_attr.reshape(-1),
    ]
    var_values = [
        cgm_values.reshape(-1),
        carb_values.reshape(-1),
        basal_values.reshape(-1),
        bolus_values.reshape(-1),
    ]

    for i, (attr, values) in enumerate(zip(var_attr, var_values)):
        y = np.random.normal(loc=i, scale=0.05, size=attr.shape)
        sc = ax.scatter(
            attr,
            y,
            c=values,
            cmap=define_cmap,
            s=10,
            alpha=0.9,
            label=feature_names[i],
        )

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Input Feature Attribution")
    # ax.set_title("SHAP Attribution per Input Channel")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7, label="Zero Line")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Input Feature Value")
    cbar.set_ticks([values.min(), values.max()])
    cbar.set_ticklabels(["Low", "High"])

    return fig, ax


def plot_shap_violin(group_shap):
    """
    Plot SHAP values using a violin plot for each input feature.

    Parameters:
        group_shap (dict): Dictionary containing SHAP values for each input feature.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    var_order = ["CGM", "Carb Intake", "Insulin Basal", "Insulin Bolus"]
    colors = ["#E41A1C", "#FDAE6B", "#4C9AEF", "#A463F2"]

    data = [np.asarray(group_shap[k], dtype=float) for k in var_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(
        data,
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.8,
    )

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        # pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if hasattr(parts[partname], "set_edgecolor"):
            parts[partname].set_color(colors)
        elif isinstance(parts[partname], (list, tuple)):
            for line, c in zip(parts[partname], colors):
                line.set_color(c)

    ax.set_xticks(np.arange(1, len(var_order) + 1))
    ax.set_xticklabels(var_order)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45, color="gray")
    ax.set_ylabel("SHAP Attribution Value")
    ax.set_xlabel("Input Features")

    return fig, ax


def plot_shap_boxplot(group_shap):
    """
    Plot SHAP values using a boxplot for each input feature.

    Parameters:
        group_shap (dict): Dictionary containing SHAP values for each input feature.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    var_order = ["CGM", "Carb Intake", "Insulin Basal", "Insulin Bolus"]
    colors = ["#E41A1C", "#FDAE6B", "#4C9AEF", "#A463F2"]

    data = [np.asarray(group_shap[k], dtype=float) for k in var_order]

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the boxplot
    bp = ax.boxplot(
        data,
        patch_artist=True,  # set to True to fill the boxes with color
        widths=0.5,
        showmeans=False,  # show the mean value
        meanline=False,  # do not draw a line for the mean value
        showfliers=False,  # do not show outliers
    )

    # set colors for the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    # set colors for the whiskers, caps, and medians
    for median, color in zip(bp["medians"], colors):
        median.set_color("black")
        median.set_linewidth(1)

    for whisker, cap in zip(bp["whiskers"], bp["caps"]):
        whisker.set_color("gray")
        cap.set_color("gray")

    ax.set_xticks(np.arange(1, len(var_order) + 1))
    ax.set_xticklabels(var_order)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45, color="gray")
    ax.set_ylabel("SHAP Attribution Value")
    ax.set_xlabel("Input Features")

    return fig, ax
    return fig, ax
    return fig, ax
