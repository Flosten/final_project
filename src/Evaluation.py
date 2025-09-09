"""
Evaluation metrics and interpretability methods for the project.
"""

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
import tqdm
from captum.attr import ShapleyValueSampling
from scipy.signal import find_peaks
from sklearn.metrics import f1_score, mean_squared_error

import src.Visualising as vis


class WrapperModel(torch.nn.Module):
    """
    A wrapper for the model to handle the input format for the 4-channel model.
    """

    def __init__(self, model):
        """
        Initialize the wrapper with the model.

        Parameters:
            model (torch.nn.Module): The model to wrap.
        """
        super().__init__()
        self.model = model

    def forward(self, cgm, carb, basal, bolus):
        """
        Forward pass through the model.

        Parameters:
            cgm (torch.Tensor): CGM values.
            carb (torch.Tensor): Carb intake values.
            basal (torch.Tensor): Basal insulin values.
            bolus (torch.Tensor): Bolus insulin values.

        Returns:
            torch.Tensor: Model predictions.
        """
        x = torch.cat([cgm, carb, basal, bolus], dim=-1)
        return self.model(x)


# prediction (RMSE, UD, DD)
# RMSE
def calculate_rmse(predictions, targets):
    """
    Calculate Root Mean Squared Error (RMSE) between predictions and targets.

    Parameters:
        predictions (list): Predicted values.
        targets (list): True values.

    Returns:
        float: RMSE value.
    """
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)

    return rmse


def calculate_threshold_rmse(
    predictions, targets, hyper_threshold=180, hypo_threshold=70
):
    """
    Calculate RMSE for hyperglycemia and hypoglycemia separately based on thresholds.

    Parameters:
        predictions (list): Predicted values.
        targets (list): True values.
        hyper_threshold (float): Threshold for hyperglycemia.
        hypo_threshold (float): Threshold for hypoglycemia.

    Returns:
        tuple: RMSE for hyperglycemia and hypoglycemia.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    hyper_mask = targets > hyper_threshold
    hypo_mask = targets < hypo_threshold
    thres_mask = hyper_mask | hypo_mask

    if len(predictions[hyper_mask]) == 0 or len(targets[hyper_mask]) == 0:
        hyper_rmse = np.nan
    else:
        hyper_rmse = calculate_rmse(predictions[hyper_mask], targets[hyper_mask])

    if len(predictions[hypo_mask]) == 0 or len(targets[hypo_mask]) == 0:
        hypo_rmse = np.nan
    else:
        hypo_rmse = calculate_rmse(predictions[hypo_mask], targets[hypo_mask])

    if len(predictions[thres_mask]) == 0 or len(targets[thres_mask]) == 0:
        thres_rmse = np.nan
    else:
        thres_rmse = calculate_rmse(predictions[thres_mask], targets[thres_mask])

    return hyper_rmse, hypo_rmse, thres_rmse


# upward dalay and downward delay (UD, DD)
def get_ranges(truth):
    """
    Get the upward and downward ranges from the truth BG values.

    Parameters:
        truth (list): True BG values.

    Returns:
        tuple: Upward range and downward range.
    """
    peaks = find_peaks(truth, height=0)[0]
    valys = find_peaks(-np.array(truth))[0]

    # combine peaks and valleys and sort them
    whole_ranges = sorted(
        [(p, "peaks") for p in peaks] + [(v, "valleys") for v in valys]
    )
    trend_ranges = []

    for i in range(len(whole_ranges) - 1):
        t1, type1 = whole_ranges[i]
        t2, type2 = whole_ranges[i + 1]

        # upward range
        if type1 == "valleys" and type2 == "peaks":
            tn = t1
            tp = t2
            tp75 = tn + (int((tp - tn) * 0.75))
            trend_ranges.append((tn, tp75, "upward"))

        # downward range
        elif type1 == "peaks" and type2 == "valleys":
            tp = t1
            tn = t2
            tn75 = tp + (int((tn - tp) * 0.75))
            trend_ranges.append((tp, tn75, "downward"))

        else:
            continue

    return trend_ranges


def calculate_delay_j(prediction, truth, t1, t2, ph):
    """
    Calculate the delay in prediction for a given trend range.

    Parameters:
        prediction (list): Predicted BG values.
        truth (list): True BG values.
        t1 (int): Start index of the trend range.
        t2 (int): End index of the trend range.
        ph (int): Prediction horizon.

    Returns:
        int: Delay in prediction.
    """
    t_range = np.arange(t1, t2 + 1)
    min_mse = float("inf")

    best_delay = 0

    for delay in range(ph + 1):
        errors = []

        for t in t_range:
            # pred_index = t - ph + delay
            pred_index = t + delay
            if 0 <= pred_index < len(prediction):
                error = prediction[pred_index] - truth[t]
                errors.append(error**2)
        if len(errors) > 0:
            mse = np.mean(errors)
            if mse < min_mse:
                min_mse = mse
                best_delay = delay

    return best_delay


def calculate_ud_dd(prediction, truth, ph):
    """
    Calculate the average upward delay (UD) and downward delay (DD)
    for the given predictions and truth values.

    Parameters:
        prediction (list): Predicted BG values.
        truth (list): True BG values.
        ph (int): Prediction horizon.

    Returns:
        tuple: Average upward delay (UD), average downward delay (DD),
        list of upward delays, list of downward delays.
    """
    # get the upward and downward ranges
    trend_ranges = get_ranges(truth)
    ud_list = []
    dd_list = []

    # calculate the upward delay (UD) and downward delay (DD)
    for tr in trend_ranges:
        t1, t2, trend_type = tr
        delay = calculate_delay_j(prediction, truth, t1, t2, ph)

        if trend_type == "upward":
            ud_list.append(delay)
        elif trend_type == "downward":
            dd_list.append(delay)

    # calculate the average upward delay (UD) and downward delay (DD)
    ud = np.mean(ud_list) if ud_list else None
    dd = np.mean(dd_list) if dd_list else None

    return ud, dd, ud_list, dd_list


def calculate_fit(preds, truths):
    """
    Calculate the fit of predictions to the truth values.

    Parameters:
        preds (list): Predicted values.
        truths (list): True values.

    Returns:
        float: Fit value as a percentage.
    """
    y_pred = np.array(preds).flatten()
    y_true = np.array(truths).flatten()

    numerator = np.sum(np.linalg.norm(y_pred - y_true))
    demominator = np.sum(np.linalg.norm(y_true - np.mean(y_true)))

    if demominator == 0:
        return np.nan

    fit = 100 * (1 - numerator / demominator)
    return fit


# alarm system (F1 score)
def extract_event_starts(truth, label):
    """
    Extract the start indices of dangerous events in the truth array for a given label.

    Parameters:
        truth (list): True labels array.
        label (int): The label for which to extract event starts.

    Returns:
        list: List of start indices for the specified label.
    """
    event_starts = []
    prev = -1
    for i, val in enumerate(truth):
        if val == label and prev != label:
            event_starts.append(i)
        prev = val
    return event_starts


def evaluate_alarm_multiclass(alarm, truth, dws, dwe, step=1, ph=60):
    """
    Evaluate the alarm system performance using F1 score, precision, and recall.

    Parameters:
        alarm (list): Predicted alarm labels.
        truth (list): True labels.
        dws (int): Downward window size in seconds.
        dwe (int): Downward window extension in seconds.
        step (int): Step size for the evaluation.
        ph (int): Prediction horizon in seconds.

    Returns:
        dict: Dictionary containing TP, FP, FN, precision, recall, and F1 score for hypo and hyper alarms.
    """
    assert len(alarm) == len(truth), "Alarm and truth arrays must have the same length."
    alarm = np.array(alarm)
    truth = np.array(truth)
    num = len(alarm)

    # Convert dws and dwe to steps
    dws_steps = dws // step
    dwe_steps = dwe // step

    results = {}

    for label, label_name in zip([0, 1], ["hypo", "hyper"]):
        tp = 0
        fp = 0
        fn = 0
        used_alarms = set()

        # alarm_updated = extract_pred_event(alarm, label)
        truth_alarm_start = extract_event_starts(truth, label)
        print(f"Truth starts for {label_name}: {truth_alarm_start}")
        pred_alarm_start = extract_event_starts(alarm, label)
        print(f"Predicted starts for {label_name}: {pred_alarm_start}")

        # find true positives (TP) and false negatives (FN)
        for k_h in truth_alarm_start:
            start = max(0, k_h - dws_steps + ph)
            end = min(num - 1, k_h - dwe_steps + ph)
            hit = False
            for k in range(start, end + 1):
                if k in used_alarms:
                    continue
                if k in pred_alarm_start:
                    used_alarms.add(k)
                    hit = True
                    tp += 1
                    break
            if not hit:
                fn += 1

        # find false positives (FP)
        for k in pred_alarm_start:
            if k in used_alarms:
                continue
            future = min(k + dws_steps + 1, num)
            if np.sum(truth[k:future] == label) == 0:
                fp += 1

        # calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results[label_name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        }

    # calculate overall F1 score
    overall_tp = results["hypo"]["TP"] + results["hyper"]["TP"]
    overall_fp = results["hypo"]["FP"] + results["hyper"]["FP"]
    overall_fn = results["hypo"]["FN"] + results["hyper"]["FN"]
    overall_precision = (
        overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    )
    overall_recall = (
        overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    )
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )
    results["Overall"] = {
        "TP": overall_tp,
        "FP": overall_fp,
        "FN": overall_fn,
        "Precision": round(overall_precision, 4),
        "Recall": round(overall_recall, 4),
        "F1": round(overall_f1, 4),
    }

    return results


# def describe_results(results):

#     data = np.array(results)
#     data = data[~np.isnan(data)]

#     # normality test
#     _, p_value = stats.shapiro(data)

#     results_description = {}
#     results_description["p_value"] = p_value

#     if p_value > 0.05:  # normal distribution
#         results_description["distribution"] = "Normal"
#         results_description["mean"] = np.mean(data)
#         results_description["std"] = np.std(data, ddof=1)

#     else:  # non-normal distribution
#         results_description["distribution"] = "Non-normal"
#         results_description["median"] = np.median(data)
#         q25, q75 = np.percentile(data, [25, 75])
#         results_description["q25"] = q25
#         results_description["q75"] = q75

#     return results_description


# def describe_results(results):
#     """
#     Describe the results for the group study.

#     Parameters:
#         results (list): List of results to describe.

#     Returns:
#         dict: Dictionary containing the description of the results.
#     """
#     data = np.array(results)
#     data = data[~np.isnan(data)]

#     # normality test
#     _, p_value = stats.shapiro(data)

#     results_description = {}
#     results_description["p_value"] = p_value

#     # only use mean and std for description
#     # results_description["distribution"] = "Normal"
#     results_description["mean"] = np.mean(data)
#     results_description["std"] = np.std(data, ddof=1)

#     return results_description


def describe_results(results):
    """
    Describe the results for the group study.

    Parameters:
        results (list): List of results to describe.

    Returns:
        dict: Dictionary containing the description of the results.
    """
    if isinstance(results[0], dict):
        summary = {}
        for group in results[0].keys():
            summary[group] = {}
            for metric in results[0][group].keys():
                values = [
                    r[group][metric] for r in results if r[group][metric] is not None
                ]
                arr = np.array(values, dtype=float)
                arr = arr[~np.isnan(arr)]
                summary[group][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)),
                }
    else:
        arr = np.array(results, dtype=float)
        arr = arr[~np.isnan(arr)]
        summary = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
        }

    return summary


# interpretability (SHAP) proposed model and ablation study (loss function)
def combine_directional(attr_ori, attr_physio, eps=1e-8):
    """
    Combine origin + physio SHAP attributions using directional weighting.

    Parameters:
        attr_ori (np.ndarray): Original SHAP attribution values.
        attr_physio (np.ndarray): Physiological SHAP attribution values.
        eps (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Combined SHAP attribution values.
    """
    numerator = attr_ori * np.abs(attr_ori) + attr_physio * np.abs(attr_physio)
    denominator = np.abs(attr_ori) + np.abs(attr_physio) + eps
    combined = numerator / denominator
    return combined


def calculate_shap_proposed(
    model,
    dataloader,
    tp_insulin_basal,
    tp_insulin_bolus,
    tp_meal,
    seq_len_basal_insulin,
    seq_len_bolus_insulin,
    seq_len_carb_intake,
    interval,
    max_samples=10,
    n_samples=5,  # 15
):
    """
    Calculate SHAP values for the proposed model with physiological layer.

    Parameters:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        tp_insulin_basal (float): Peak time for insulin basal.
        tp_insulin_bolus (float): Peak time for insulin bolus.
        tp_meal (float): Peak time for carb intake.
        seq_len_basal_insulin (int): Sequence length for basal insulin.
        seq_len_bolus_insulin (int): Sequence length for bolus insulin.
        seq_len_carb_intake (int): Sequence length for carb intake.
        interval (int): Time interval in seconds.
        max_samples (int): Maximum number of samples to process.
        n_samples (int): Number of samples for SHAP calculation.

    Returns:
        tuple: Figure, axes, and SHAP values per patient.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # transform peak time to time interval
    tp_insulin_basal /= interval
    tp_insulin_bolus /= interval
    tp_meal /= interval

    kernel_size_basal = seq_len_basal_insulin / interval
    kernel_size_bolus = seq_len_bolus_insulin / interval
    kernel_size_meal = seq_len_carb_intake / interval

    (
        attr_cgm,
        attr_basal,
        attr_bolus,
        attr_meal,
        attr_basal_phy,
        attr_bolus_phy,
        attr_meal_phy,
    ) = ([], [], [], [], [], [], [])
    cgm_values, basal_values, bolus_values, meal_values = [], [], [], []
    count = 0

    sv = ShapleyValueSampling(
        lambda cgm, basal_ori, bolus_ori, meal_ori, basal, bolus, meal: model(
            cgm, basal_ori, bolus_ori, meal_ori, basal, bolus, meal
        )[0]
    )

    for x, _, _ in dataloader:
        if count >= max_samples:
            break

        # data to device
        cgm = x["cgm"].to(device)  # (batch, seq_len)
        basal = x["current_basal"].to(device)
        bolus = x["current_bolus"].to(device)
        meal = x["carb_intake"].to(device)

        # physiological layer
        basal_processed = physiological_layer(basal, tp_insulin_basal, 1000).unsqueeze(
            2
        )
        bolus_processed = physiological_layer(bolus, tp_insulin_bolus, 1000).unsqueeze(
            2
        )
        meal_processed = physiological_layer(meal, tp_meal, 1000).unsqueeze(2)

        # preprocess data
        cgm = cgm.unsqueeze(2)
        basal = basal.unsqueeze(2)
        bolus = bolus.unsqueeze(2)
        meal = meal.unsqueeze(2)
        basal_last = basal_processed
        bolus_last = bolus_processed
        meal_last = meal_processed

        # append input values
        cgm_values.append(cgm.detach().cpu().numpy())
        basal_values.append(basal_last.detach().cpu().numpy())
        bolus_values.append(bolus_last.detach().cpu().numpy())
        meal_values.append(meal_last.detach().cpu().numpy())

        # baseline
        baseline = (
            torch.zeros_like(cgm),
            torch.zeros_like(basal),
            torch.zeros_like(bolus),
            torch.zeros_like(meal),
            torch.zeros_like(basal_last),
            torch.zeros_like(bolus_last),
            torch.zeros_like(meal_last),
        )

        # attribution calculation
        attribution = sv.attribute(
            inputs=(cgm, basal, bolus, meal, basal_last, bolus_last, meal_last),
            n_samples=n_samples,
            baselines=baseline,
        )

        # print(f"Attribution shapes: {[a.shape for a in attribution]}")

        # attr_cgm.append(attribution[0].detach().cpu().numpy())
        # attr_basal.append(attribution[1].detach().cpu().numpy())
        # attr_bolus.append(attribution[2].detach().cpu().numpy())
        # attr_meal.append(attribution[3].detach().cpu().numpy())

        attr_cgm_batch = attribution[0].detach().cpu().numpy()
        mean_cgm = attr_cgm_batch.sum(axis=2, keepdims=True)
        cgm_shape = (mean_cgm.shape[0], 1, attribution[0].shape[2], 1)
        attr_cgm.append(np.broadcast_to(mean_cgm, cgm_shape))

        attr_basal_batch = attribution[1].detach().cpu().numpy()
        mean_basal = attr_basal_batch.sum(axis=2, keepdims=True)
        basal_shape = (mean_basal.shape[0], 1, attribution[1].shape[2], 1)
        attr_basal.append(np.broadcast_to(mean_basal, basal_shape))

        attr_bolus_batch = attribution[2].detach().cpu().numpy()
        mean_bolus = attr_bolus_batch.sum(axis=2, keepdims=True)
        bolus_shape = (mean_bolus.shape[0], 1, attribution[2].shape[2], 1)
        attr_bolus.append(np.broadcast_to(mean_bolus, bolus_shape))

        attr_meal_batch = attribution[3].detach().cpu().numpy()
        mean_meal = attr_meal_batch.sum(axis=2, keepdims=True)
        meal_shape = (mean_meal.shape[0], 1, attribution[3].shape[2], 1)
        attr_meal.append(np.broadcast_to(mean_meal, meal_shape))

        # attribution for preprocessed input values
        attr_basal_phy_batch = attribution[4].detach().cpu().numpy()
        mean_basal_phy = attr_basal_phy_batch.sum(axis=2, keepdims=True)
        basal_phy_shape = (mean_basal_phy.shape[0], 1, attribution[4].shape[2], 1)
        attr_basal_phy.append(np.broadcast_to(mean_basal_phy, basal_phy_shape))

        attr_bolus_phy_batch = attribution[5].detach().cpu().numpy()
        mean_bolus_phy = attr_bolus_phy_batch.sum(axis=2, keepdims=True)
        bolus_phy_shape = (mean_bolus_phy.shape[0], 1, attribution[5].shape[2], 1)
        attr_bolus_phy.append(np.broadcast_to(mean_bolus_phy, bolus_phy_shape))

        attr_meal_phy_batch = attribution[6].detach().cpu().numpy()
        mean_meal_phy = attr_meal_phy_batch.sum(axis=2, keepdims=True)
        meal_phy_shape = (mean_meal_phy.shape[0], 1, attribution[6].shape[2], 1)
        attr_meal_phy.append(np.broadcast_to(mean_meal_phy, meal_phy_shape))

        count += 1

    # concatenate the attribution values and input values
    attr_cgm = np.concatenate(attr_cgm, axis=0)
    attr_basal = np.concatenate(attr_basal, axis=0)
    attr_bolus = np.concatenate(attr_bolus, axis=0)
    attr_meal = np.concatenate(attr_meal, axis=0)
    attr_basal_phy = np.concatenate(attr_basal_phy, axis=0)
    attr_bolus_phy = np.concatenate(attr_bolus_phy, axis=0)
    attr_meal_phy = np.concatenate(attr_meal_phy, axis=0)

    # print(
    #     f"Attribution shapes: {attr_cgm.shape}, {attr_basal.shape}, {attr_bolus.shape}, {attr_meal.shape}"
    # )

    # total insulin basal, insulin bolus, and carb intake attribution values
    attr_basal = combine_directional(attr_basal, attr_basal_phy)
    attr_bolus = combine_directional(
        attr_bolus, attr_bolus_phy
    )  # no need to combine, only physio layer
    attr_meal = combine_directional(attr_meal, attr_meal_phy)

    cgm_values = np.concatenate(cgm_values, axis=0)
    basal_values = np.concatenate(basal_values, axis=0)
    bolus_values = np.concatenate(bolus_values, axis=0)
    meal_values = np.concatenate(meal_values, axis=0)
    print(
        f"Input shapes: {cgm_values.shape}, {basal_values.shape}, {bolus_values.shape}, {meal_values.shape}"
    )

    # average the attribution values across sequence length
    fig, ax = vis.plot_interpretability(
        attr_cgm[:, :, -1, :],  # only take the last frame
        attr_meal[:, :, -1, :],
        attr_basal[:, :, -1, :],
        attr_bolus[:, :, -1, :],
        cgm_values=cgm_values[:, -1, :],  # only take the last frame
        carb_values=meal_values[:, -1, :],
        basal_values=basal_values[:, -1, :],
        bolus_values=bolus_values[:, -1, :],
    )

    # record the attribution values
    shap_per_patient = {
        "CGM": attr_cgm[:, :, -1, :].reshape(-1),
        "Carb Intake": attr_meal[:, :, -1, :].reshape(-1),
        "Insulin Basal": attr_basal[:, :, -1, :].reshape(-1),
        "Insulin Bolus": attr_bolus[:, :, -1, :].reshape(-1),
    }

    return fig, ax, shap_per_patient


# interpretability (SHAP) proposed model without physiological layer
def calculate_shap_proposed_no_phy(
    model,
    dataloader,
    tp_insulin_basal,
    tp_insulin_bolus,
    tp_meal,
    seq_len_basal_insulin,
    seq_len_bolus_insulin,
    seq_len_carb_intake,
    interval,
    max_samples=10,
    n_samples=5,  # 15
):
    """
    Calculate SHAP values for the proposed model without physiological layer (ablation study).

    Parameters:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        tp_insulin_basal (float): Peak time for insulin basal.
        tp_insulin_bolus (float): Peak time for insulin bolus.
        tp_meal (float): Peak time for carb intake.
        seq_len_basal_insulin (int): Sequence length for basal insulin.
        seq_len_bolus_insulin (int): Sequence length for bolus insulin.
        seq_len_carb_intake (int): Sequence length for carb intake.
        interval (int): Time interval in seconds.
        max_samples (int): Maximum number of samples to process.
        n_samples (int): Number of samples for SHAP calculation.

    Returns:
        tuple: Figure, axes, and SHAP values per patient.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # transform peak time to time interval
    tp_insulin_basal /= interval
    tp_insulin_bolus /= interval
    tp_meal /= interval

    kernel_size_basal = seq_len_basal_insulin / interval
    kernel_size_bolus = seq_len_bolus_insulin / interval
    kernel_size_meal = seq_len_carb_intake / interval

    attr_cgm, attr_basal, attr_bolus, attr_meal = [], [], [], []
    cgm_values, basal_values, bolus_values, meal_values = [], [], [], []
    count = 0

    sv = ShapleyValueSampling(
        lambda cgm, basal, bolus, meal: model(cgm, basal, bolus, meal)[0]
    )

    for x, _, _ in dataloader:
        if count >= max_samples:
            break

        # data to device
        cgm = x["cgm"].to(device)  # (batch, seq_len)
        basal = x["current_basal"].to(device)
        bolus = x["current_bolus"].to(device)
        meal = x["carb_intake"].to(device)

        # preprocess data
        cgm = cgm.unsqueeze(2)
        basal_last = basal.unsqueeze(2)
        bolus_last = bolus.unsqueeze(2)
        meal_last = meal.unsqueeze(2)

        # append input values
        cgm_values.append(cgm.detach().cpu().numpy())
        basal_values.append(basal_last.detach().cpu().numpy())
        bolus_values.append(bolus_last.detach().cpu().numpy())
        meal_values.append(meal_last.detach().cpu().numpy())

        # baseline
        baseline = (
            torch.zeros_like(cgm),
            torch.zeros_like(basal_last),
            torch.zeros_like(bolus_last),
            torch.zeros_like(meal_last),
        )

        # attribution calculation
        attribution = sv.attribute(
            inputs=(cgm, basal_last, bolus_last, meal_last),
            n_samples=n_samples,
            baselines=baseline,
        )

        # print(f"Attribution shapes: {[a.shape for a in attribution]}")

        # attr_cgm.append(attribution[0].detach().cpu().numpy())
        # attr_basal.append(attribution[1].detach().cpu().numpy())
        # attr_bolus.append(attribution[2].detach().cpu().numpy())
        # attr_meal.append(attribution[3].detach().cpu().numpy())

        attr_cgm_batch = attribution[0].detach().cpu().numpy()
        mean_cgm = attr_cgm_batch.sum(axis=2, keepdims=True)
        cgm_shape = (mean_cgm.shape[0], 1, attribution[0].shape[2], 1)
        attr_cgm.append(np.broadcast_to(mean_cgm, cgm_shape))

        attr_basal_batch = attribution[1].detach().cpu().numpy()
        mean_basal = attr_basal_batch.sum(axis=2, keepdims=True)
        basal_shape = (mean_basal.shape[0], 1, attribution[1].shape[2], 1)
        attr_basal.append(np.broadcast_to(mean_basal, basal_shape))

        attr_bolus_batch = attribution[2].detach().cpu().numpy()
        mean_bolus = attr_bolus_batch.sum(axis=2, keepdims=True)
        bolus_shape = (mean_bolus.shape[0], 1, attribution[2].shape[2], 1)
        attr_bolus.append(np.broadcast_to(mean_bolus, bolus_shape))

        attr_meal_batch = attribution[3].detach().cpu().numpy()
        mean_meal = attr_meal_batch.sum(axis=2, keepdims=True)
        meal_shape = (mean_meal.shape[0], 1, attribution[3].shape[2], 1)
        attr_meal.append(np.broadcast_to(mean_meal, meal_shape))

        count += 1

    # concatenate the attribution values and input values
    attr_cgm = np.concatenate(attr_cgm, axis=0)
    attr_basal = np.concatenate(attr_basal, axis=0)
    attr_bolus = np.concatenate(attr_bolus, axis=0)
    attr_meal = np.concatenate(attr_meal, axis=0)
    print(
        f"Attribution shapes: {attr_cgm.shape}, {attr_basal.shape}, {attr_bolus.shape}, {attr_meal.shape}"
    )

    cgm_values = np.concatenate(cgm_values, axis=0).sum(axis=1, keepdims=True)
    basal_values = np.concatenate(basal_values, axis=0).sum(axis=1, keepdims=True)
    bolus_values = np.concatenate(bolus_values, axis=0).sum(axis=1, keepdims=True)
    meal_values = np.concatenate(meal_values, axis=0).sum(axis=1, keepdims=True)
    print(
        f"Input shapes: {cgm_values.shape}, {basal_values.shape}, {bolus_values.shape}, {meal_values.shape}"
    )

    # average the attribution values across sequence length
    fig, ax = vis.plot_interpretability(
        attr_cgm[:, :, -1, :],  # only take the last frame
        attr_meal[:, :, -1, :],
        attr_basal[:, :, -1, :],
        attr_bolus[:, :, -1, :],
        cgm_values=cgm_values[:, -1, :],  # only take the last frame
        carb_values=meal_values[:, -1, :],
        basal_values=basal_values[:, -1, :],
        bolus_values=bolus_values[:, -1, :],
    )

    # record the attribution values
    shap_per_patient = {
        "CGM": attr_cgm[:, :, -1, :].reshape(-1),
        "Carb Intake": attr_meal[:, :, -1, :].reshape(-1),
        "Insulin Basal": attr_basal[:, :, -1, :].reshape(-1),
        "Insulin Bolus": attr_bolus[:, :, -1, :].reshape(-1),
    }

    return fig, ax, shap_per_patient


# interpretability (SHAP) proposed model without physiological layer
def calculate_shap_proposed_no_dual_input(
    model,
    dataloader,
    tp_insulin_basal,
    tp_insulin_bolus,
    tp_meal,
    seq_len_basal_insulin,
    seq_len_bolus_insulin,
    seq_len_carb_intake,
    interval,
    max_samples=10,
    n_samples=5,  # 15
):
    """
    Calculate SHAP values for the proposed model without dual input (ablation study).

    Parameters:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        tp_insulin_basal (float): Peak time for insulin basal.
        tp_insulin_bolus (float): Peak time for insulin bolus.
        tp_meal (float): Peak time for carb intake.
        seq_len_basal_insulin (int): Sequence length for basal insulin.
        seq_len_bolus_insulin (int): Sequence length for bolus insulin.
        seq_len_carb_intake (int): Sequence length for carb intake.
        interval (int): Time interval in seconds.
        max_samples (int): Maximum number of samples to process.
        n_samples (int): Number of samples for SHAP calculation.

    Returns:
        tuple: Figure, axes, and SHAP values per patient.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # transform peak time to time interval
    tp_insulin_basal /= interval
    tp_insulin_bolus /= interval
    tp_meal /= interval

    kernel_size_basal = seq_len_basal_insulin / interval
    kernel_size_bolus = seq_len_bolus_insulin / interval
    kernel_size_meal = seq_len_carb_intake / interval

    attr_cgm, attr_basal, attr_bolus, attr_meal = [], [], [], []
    cgm_values, basal_values, bolus_values, meal_values = [], [], [], []
    count = 0

    sv = ShapleyValueSampling(
        lambda cgm, basal, bolus, meal: model(cgm, basal, bolus, meal)[0]
    )

    for x, _, _ in dataloader:
        if count >= max_samples:
            break

        # data to device
        cgm = x["cgm"].to(device)  # (batch, seq_len)
        basal = x["current_basal"].to(device)
        bolus = x["current_bolus"].to(device)
        meal = x["carb_intake"].to(device)

        # physiological layer
        basal_processed = physiological_layer(basal, tp_insulin_basal, 1000).unsqueeze(
            2
        )
        bolus_processed = physiological_layer(bolus, tp_insulin_bolus, 1000).unsqueeze(
            2
        )
        meal_processed = physiological_layer(meal, tp_meal, 1000).unsqueeze(2)

        # preprocess data
        cgm = cgm.unsqueeze(2)
        basal_last = basal_processed
        bolus_last = bolus_processed
        meal_last = meal_processed

        # append input values
        cgm_values.append(cgm.detach().cpu().numpy())
        basal_values.append(basal_last.detach().cpu().numpy())
        bolus_values.append(bolus_last.detach().cpu().numpy())
        meal_values.append(meal_last.detach().cpu().numpy())

        # baseline
        baseline = (
            torch.zeros_like(cgm),
            torch.zeros_like(basal_last),
            torch.zeros_like(bolus_last),
            torch.zeros_like(meal_last),
        )

        # attribution calculation
        attribution = sv.attribute(
            inputs=(cgm, basal_last, bolus_last, meal_last),
            n_samples=n_samples,
            baselines=baseline,
        )

        # print(f"Attribution shapes: {[a.shape for a in attribution]}")

        # attr_cgm.append(attribution[0].detach().cpu().numpy())
        # attr_basal.append(attribution[1].detach().cpu().numpy())
        # attr_bolus.append(attribution[2].detach().cpu().numpy())
        # attr_meal.append(attribution[3].detach().cpu().numpy())

        attr_cgm_batch = attribution[0].detach().cpu().numpy()
        mean_cgm = attr_cgm_batch.sum(axis=2, keepdims=True)
        cgm_shape = (mean_cgm.shape[0], 1, attribution[0].shape[2], 1)
        attr_cgm.append(np.broadcast_to(mean_cgm, cgm_shape))

        attr_basal_batch = attribution[1].detach().cpu().numpy()
        mean_basal = attr_basal_batch.sum(axis=2, keepdims=True)
        basal_shape = (mean_basal.shape[0], 1, attribution[1].shape[2], 1)
        attr_basal.append(np.broadcast_to(mean_basal, basal_shape))

        attr_bolus_batch = attribution[2].detach().cpu().numpy()
        mean_bolus = attr_bolus_batch.sum(axis=2, keepdims=True)
        bolus_shape = (mean_bolus.shape[0], 1, attribution[2].shape[2], 1)
        attr_bolus.append(np.broadcast_to(mean_bolus, bolus_shape))

        attr_meal_batch = attribution[3].detach().cpu().numpy()
        mean_meal = attr_meal_batch.sum(axis=2, keepdims=True)
        meal_shape = (mean_meal.shape[0], 1, attribution[3].shape[2], 1)
        attr_meal.append(np.broadcast_to(mean_meal, meal_shape))

        count += 1

    # concatenate the attribution values and input values
    attr_cgm = np.concatenate(attr_cgm, axis=0)
    attr_basal = np.concatenate(attr_basal, axis=0)
    attr_bolus = np.concatenate(attr_bolus, axis=0)
    attr_meal = np.concatenate(attr_meal, axis=0)
    print(
        f"Attribution shapes: {attr_cgm.shape}, {attr_basal.shape}, {attr_bolus.shape}, {attr_meal.shape}"
    )

    cgm_values = np.concatenate(cgm_values, axis=0)
    basal_values = np.concatenate(basal_values, axis=0)
    bolus_values = np.concatenate(bolus_values, axis=0)
    meal_values = np.concatenate(meal_values, axis=0)
    print(
        f"Input shapes: {cgm_values.shape}, {basal_values.shape}, {bolus_values.shape}, {meal_values.shape}"
    )

    # average the attribution values across sequence length
    fig, ax = vis.plot_interpretability(
        attr_cgm[:, :, -1, :],  # only take the last frame
        attr_meal[:, :, -1, :],
        attr_basal[:, :, -1, :],
        attr_bolus[:, :, -1, :],
        cgm_values=cgm_values[:, -1, :],  # only take the last frame
        carb_values=meal_values[:, -1, :],
        basal_values=basal_values[:, -1, :],
        bolus_values=bolus_values[:, -1, :],
    )

    # record the attribution values
    shap_per_patient = {
        "CGM": attr_cgm[:, :, -1, :].reshape(-1),
        "Carb Intake": attr_meal[:, :, -1, :].reshape(-1),
        "Insulin Basal": attr_basal[:, :, -1, :].reshape(-1),
        "Insulin Bolus": attr_bolus[:, :, -1, :].reshape(-1),
    }

    return fig, ax, shap_per_patient


# interpretability baseline
def calculate_shap_for_4channels(
    model,
    dataloader,
    max_samples=10,
    n_samples=5,
):
    """
    Calculate SHAP values for Baseline model with 4 channels,
    including CGM, Carb Intake, Insulin Basal, and Insulin Bolus.

    Parameters:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        max_samples (int): Maximum number of samples to process.
        n_samples (int): Number of samples for SHAP calculation.

    Returns:
        tuple: Figure, axes, and SHAP values per patient.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WrapperModel(model).to(device)  # wrap the model
    model.eval()

    attr_cgm, attr_carb, attr_basal, attr_bolus = [], [], [], []
    cgm_values, carb_values, basal_values, bolus_values = [], [], [], []

    count = 0
    sv = ShapleyValueSampling(model)

    for x, _ in tqdm.tqdm(dataloader, desc="Calculating SHAP"):
        if count >= max_samples:
            break

        # send data to device
        cgm = x[:, :, 0].unsqueeze(-1).to(device)
        carb = x[:, :, 1].unsqueeze(-1).to(device)
        basal = x[:, :, 2].unsqueeze(-1).to(device)
        bolus = x[:, :, 3].unsqueeze(-1).to(device)

        baseline = (
            torch.zeros_like(cgm),
            torch.zeros_like(carb),
            torch.zeros_like(basal),
            torch.zeros_like(bolus),
        )

        # calculate SHAP values
        attribution = sv.attribute(
            inputs=(cgm, carb, basal, bolus),
            n_samples=n_samples,
            baselines=baseline,
        )

        # append attribution values and input values
        # attr_cgm.append(attribution[0].detach().cpu().numpy())
        # attr_carb.append(attribution[1].detach().cpu().numpy())
        # attr_basal.append(attribution[2].detach().cpu().numpy())
        # attr_bolus.append(attribution[3].detach().cpu().numpy())

        attr_cgm_batch = attribution[0].detach().cpu().numpy()
        mean_cgm = attr_cgm_batch.sum(axis=2, keepdims=True)
        cgm_shape = (mean_cgm.shape[0], 1, attribution[0].shape[2], 1)
        attr_cgm.append(np.broadcast_to(mean_cgm, cgm_shape))

        attr_basal_batch = attribution[1].detach().cpu().numpy()
        mean_basal = attr_basal_batch.sum(axis=2, keepdims=True)
        basal_shape = (mean_basal.shape[0], 1, attribution[1].shape[2], 1)
        attr_basal.append(np.broadcast_to(mean_basal, basal_shape))

        attr_bolus_batch = attribution[2].detach().cpu().numpy()
        mean_bolus = attr_bolus_batch.sum(axis=2, keepdims=True)
        bolus_shape = (mean_bolus.shape[0], 1, attribution[2].shape[2], 1)
        attr_bolus.append(np.broadcast_to(mean_bolus, bolus_shape))

        attr_meal_batch = attribution[3].detach().cpu().numpy()
        mean_meal = attr_meal_batch.sum(axis=2, keepdims=True)
        meal_shape = (mean_meal.shape[0], 1, attribution[3].shape[2], 1)
        attr_carb.append(np.broadcast_to(mean_meal, meal_shape))

        cgm_values.append(cgm.detach().cpu().numpy())
        carb_values.append(carb.detach().cpu().numpy())
        basal_values.append(basal.detach().cpu().numpy())
        bolus_values.append(bolus.detach().cpu().numpy())

        count += 1

    # concatenate the attribution values and input values
    attr_cgm = np.concatenate(attr_cgm, axis=0)
    attr_carb = np.concatenate(attr_carb, axis=0)
    attr_basal = np.concatenate(attr_basal, axis=0)
    attr_bolus = np.concatenate(attr_bolus, axis=0)

    cgm_values = np.concatenate(cgm_values, axis=0).sum(axis=1, keepdims=True)
    carb_values = np.concatenate(carb_values, axis=0).sum(axis=1, keepdims=True)
    basal_values = np.concatenate(basal_values, axis=0).sum(axis=1, keepdims=True)
    bolus_values = np.concatenate(bolus_values, axis=0).sum(axis=1, keepdims=True)

    # average the attribution values across sequence length
    fig, ax = vis.plot_interpretability(
        attr_cgm[:, :, -1, :],  # only take the last frame
        attr_carb[:, :, -1, :],
        attr_basal[:, :, -1, :],
        attr_bolus[:, :, -1, :],
        cgm_values[:, -1, :],  # only take the last frame
        carb_values[:, -1, :],
        basal_values[:, -1, :],
        bolus_values[:, -1, :],  # only take the last frame
    )

    # record the attribution values
    shap_per_patient = {
        "CGM": attr_cgm[:, :, -1, :].reshape(-1),
        "Carb Intake": attr_carb[:, :, -1, :].reshape(-1),
        "Insulin Basal": attr_basal[:, :, -1, :].reshape(-1),
        "Insulin Bolus": attr_bolus[:, :, -1, :].reshape(-1),
    }

    return fig, ax, shap_per_patient


def min_max_normalization(data):
    """
    Normalize the input data using min-max normalization.

    Parameters:
        data (torch.Tensor): Input data to normalize.

    Returns:
        torch.Tensor: Normalized data.
    """
    min_value = data.min(dim=1, keepdim=True)[0]
    max_value = data.max(dim=1, keepdim=True)[0]
    normalized_data = (data - min_value) / (max_value - min_value + 1e-8)
    return normalized_data


def physiological_layer(input_seq, lamda, kernel_size):
    """
    Apply a physiological layer to the input sequence using a kernel
    based on a physiological model.

    Parameters:
        input_seq (torch.Tensor): Input sequence of shape (batch_size, seq_len).
        lamda (float): Lambda parameter for the physiological model.
        kernel_size (int): Size of the kernel to be used.

    Returns:
        torch.Tensor: Output sequence after applying the physiological layer.
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
