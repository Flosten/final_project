import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def create_xticks(data, time_steps, ticks_per_day):
    """
    Create x-ticks for the plot based on the data length and ticks per day.

    Args:
        data (list): List of data points to be visualised.
        time_steps (int): Number of time steps per tick.
        ticks_per_day (int): Number of ticks per day.

    Returns:
        xticks (np.ndarray): Array of x-tick positions.
        xtick_labels (list): List of formatted x-tick labels.
    """

    num = len(data)

    tick_interval = 1440 // ticks_per_day
    sample_interval = tick_interval // time_steps

    xticks = np.arange(0, num, sample_interval)  # x-ticks at every sample_interval

    xtick_labels = []
    for i in xticks:
        minutes = (i * time_steps) % 1440
        hours = minutes // 60
        minute = minutes % 60
        xtick_labels.append(f"{hours:02d}:{minute:02d}")

    return xticks, xtick_labels


def visualise_data(data, ticks_per_day, time_steps=1):
    """
    Visualise the data using a line plot.

    Args:
        data (list): List of data points to be visualised.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data, color="#3167DB", linewidth=1.5, alpha=1)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    # Create x-ticks based on the data length and ticks per day
    xticks, xtick_labels = create_xticks(data, time_steps, ticks_per_day)
    ax.set_xticks(xticks, xtick_labels, rotation=45)
    # ax.set_title("Data Visualisation")
    # ax.legend()

    return fig, ax


def visualise_insulin_meal_response(
    data_1, data_2, legend_1, legend_2, ticks_per_day, time_steps=5
):
    """
    Visualise the insulin and meal response data using a line plot.

    Args:
        data_1 (list): List of insulin response data points.
        data_2 (list): List of meal response data points.
        legend_1 (str): Legend label for insulin response.
        legend_2 (str): Legend label for meal response.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data_1, label=legend_1, color="blue", linewidth=1.5, alpha=0.8)
    ax.plot(data_2, label=legend_2, color="orange", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()

    # Create x-ticks based on the data length and ticks per day
    xticks, xtick_labels = create_xticks(data_1, time_steps, ticks_per_day)
    ax.set_xticks(xticks, xtick_labels, rotation=45)

    return fig, ax


def plot_lr_1(train_loss, val_loss):
    """
    Plot training and validation loss over epochs.

    Args:
        train_loss (list): List of training loss values per epoch.
        val_loss (list): List of validation loss values per epoch.

    Returns:
    fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, label="Training Loss", color="blue")
    ax.plot(epochs, val_loss, label="Validation Loss", color="orange")

    ax.set_xticks(epochs)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    ax.legend()

    return fig, ax


def plot_lr_2(train_loss_1, train_loss_2, val_loss_1, val_loss_2):
    """
    Plot training and validation loss for two different loss functions.

    Args:
        train_loss_1 (list): List of training loss values for the custom loss function.
        train_loss_2 (list): List of training loss values for the MSE loss function.
        val_loss_1 (list): List of validation loss values for the custom loss function.
        val_loss_2 (list): List of validation loss values for the MSE loss function.

    Returns:
    fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(train_loss_1) + 1)

    ax.plot(epochs, train_loss_1, label="Custom Loss (Train)", color="blue")
    ax.plot(epochs, train_loss_2, label="MSE Loss (Train)", color="green")
    ax.plot(epochs, val_loss_1, label="Custom Loss (Val)", color="orange")
    ax.plot(epochs, val_loss_2, label="MSE Loss (Val)", color="red")

    ax.set_xticks(epochs)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    ax.legend()

    return fig, ax


def plot_pred_visualisation(pred, truth, ticks_per_day, time_steps):
    """
    Plot the predicted BG levels against the true BG levels.

    Args:
        pred (list): List of predicted BG levels.
        truth (list): List of true BG levels.
        ticks_per_day (int): Number of ticks per day for x-axis.
        time_steps (int): Number of time steps per tick.

    Returns:
    fig, ax: Matplotlib figure and axes objects.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        pred,
        label="Prediction",
        color="#56B4E9",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.plot(
        truth, label="Truth", color="#AF32AF", linestyle="-", linewidth=1.2, alpha=0.8
    )  # "#A21FAB" is a dark magenta color

    ax.set_xlabel("Time")
    ax.set_ylabel("BG Level")

    ax.legend(framealpha=0.5)
    ax.set_ylim(20, 240)  # Set y-axis limits for better visibility

    # Create x-ticks based on the data length and ticks per day
    xticks, xtick_labels = create_xticks(pred, time_steps, ticks_per_day)
    ax.set_xticks(xticks, xtick_labels, rotation=45)

    return fig, ax


def plot_pred_threshold_visualisation(
    pred, truth, ticks_per_day, time_steps, threshold_1=180, threshold_2=70
):
    """
    Plot the predicted BG levels against the true BG levels.

    Args:
        pred (list): List of predicted BG levels.
        truth (list): List of true BG levels.

    Returns:
    fig, ax: Matplotlib figure and axes objects.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

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
        linestyle="--",
        label="Hypo Threshold (70 mg/dL)",
    )

    ax.plot(
        pred,
        label="Prediction",
        color="#56B4E9",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.plot(
        truth, label="Truth", color="#AF32AF", linestyle="-", linewidth=1.2, alpha=0.8
    )  # "#A21FAB" is a dark magenta color

    ax.set_xlabel("Time")
    ax.set_ylabel("BG Level")

    ax.legend(framealpha=0.5)
    ax.set_ylim(15, 260)  # Set y-axis limits for better visibility

    # Create x-ticks based on the data length and ticks per day
    xticks, xtick_labels = create_xticks(pred, time_steps, ticks_per_day)
    ax.set_xticks(xticks, xtick_labels, rotation=45)

    return fig, ax


def plot_alarm_system(pred, truth, alarms):
    pass


# def plot_interpretability(
#     cgm_attr,
#     insulin_attr,
#     meal_attr,
#     cgm_values,
#     insulin_values,
#     meal_values,
#     feature_names=["CGM", "Insulin", "Meal"],
# ):
#     define_cmap = LinearSegmentedColormap.from_list(
#         "attr_cmap", ["#008DFF", "#0E5BDC", "#A11FAB", "#EB0079", "#FF0055"], N=256
#     )

#     fig, ax = plt.subplots(figsize=(8, 5))
#     var_attr = [cgm_attr.reshape(-1), insulin_attr.reshape(-1), meal_attr.reshape(-1)]
#     var_values = [
#         cgm_values.reshape(-1),
#         insulin_values.reshape(-1),
#         meal_values.reshape(-1),
#     ]

#     for i, (attr, values) in enumerate(zip(var_attr, var_values)):
#         y = np.random.normal(loc=i, scale=0.05, size=attr.shape)
#         sc = ax.scatter(
#             attr,
#             y,
#             c=values,
#             cmap=define_cmap,
#             s=10,
#             alpha=0.92,
#             label=feature_names[i],
#         )

#     ax.set_yticks(range(len(feature_names)))
#     ax.set_yticklabels(feature_names)
#     ax.set_xlabel("Input Feature Attribution")
#     ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7, label="Zero Line")

#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label("Input Feature Value")
#     cbar.set_ticks([values.min(), values.max()])
#     cbar.set_ticklabels(["Low", "High"])

#     return fig, ax


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
