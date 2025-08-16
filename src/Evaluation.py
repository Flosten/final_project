import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from captum.attr import IntegratedGradients, ShapleyValueSampling
from scipy.signal import find_peaks
from sklearn.metrics import f1_score, mean_squared_error

import src.Visualising as vis


class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, cgm, carb, basal, bolus):
        x = torch.cat([cgm, carb, basal, bolus], dim=-1)
        return self.model(x)


# prediction (RMSE, UD, DD)
# RMSE
def calculate_rmse(predictions, targets):
    """
    Calculate Root Mean Squared Error (RMSE) between predictions and targets.

    Args:
        predictions (list): Predicted values.
        targets (list): True values.

    Returns:
        float: RMSE value.
    """
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)

    return rmse


# upward dalay and downward delay (UD, DD)
def get_ranges(truth):
    """
    Get the upward and downward ranges from the truth BG values.

    Args:
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

    Args:
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

    Args:
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


# alarm system (F1 score)
def extract_event_starts(truth, label):
    event_starts = []
    prev = -1
    for i, val in enumerate(truth):
        if val == label and prev != label:
            event_starts.append(i)
        prev = val
    return event_starts


# def extract_pred_event(prediction, label):
#     new_event_starts = []
#     prev = -1
#     for val in prediction:
#         if val == label and prev != label:
#             new_event_starts.append(label)
#             prev = val
#         elif val == label and prev == label:
#             new_event_starts.append(2)
#             prev = val
#         else:
#             new_event_starts.append(val)
#             prev = val
#     return new_event_starts


def evaluate_alarm_multiclass(alarm, truth, dws, dwe, step=1, ph=60):

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
    max_samples=30,
    n_samples=15,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 时间转换为步长
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

        # === 数据读取与处理 ===
        cgm = x["cgm"].to(device)  # (batch, seq_len)
        basal = x["current_basal"].to(device)
        bolus = x["current_bolus"].to(device)
        meal = x["carb_intake"].to(device)

        # === physiological layer 处理 ===
        basal_processed = physiological_layer(
            basal, tp_insulin_basal, kernel_size_basal
        ).unsqueeze(2)
        bolus_processed = physiological_layer(
            bolus, tp_insulin_bolus, kernel_size_bolus
        ).unsqueeze(2)
        meal_processed = physiological_layer(meal, tp_meal, kernel_size_meal).unsqueeze(
            2
        )

        # CGM 保持整段序列，扩展维度 (batch, seq_len, 1)
        cgm = cgm.unsqueeze(2)

        # 只取 insulin/meal 的最后一帧（保持 3D）
        basal_last = basal_processed
        bolus_last = bolus_processed
        meal_last = meal_processed

        # 保存输入值用于后续可视化
        cgm_values.append(cgm.detach().cpu().numpy())  # 全序列
        basal_values.append(basal_last.detach().cpu().numpy())  # 最后一步
        bolus_values.append(bolus_last.detach().cpu().numpy())
        meal_values.append(meal_last.detach().cpu().numpy())

        # 构造 baseline（与 inputs 同形状）
        baseline = (
            torch.zeros_like(cgm),
            torch.zeros_like(basal_last),
            torch.zeros_like(bolus_last),
            torch.zeros_like(meal_last),
        )

        # attribution 计算
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

        count += cgm.shape[0]

    # 拼接输出
    attr_cgm = np.concatenate(attr_cgm, axis=0)
    attr_basal = np.concatenate(attr_basal, axis=0)
    attr_bolus = np.concatenate(attr_bolus, axis=0)
    attr_meal = np.concatenate(attr_meal, axis=0)

    cgm_values = np.concatenate(cgm_values, axis=0)
    basal_values = np.concatenate(basal_values, axis=0)
    bolus_values = np.concatenate(bolus_values, axis=0)
    meal_values = np.concatenate(meal_values, axis=0)

    # 可视化
    fig, ax = vis.plot_interpretability(
        attr_cgm,
        attr_meal,
        attr_basal,
        attr_bolus,
        cgm_values=cgm_values,
        carb_values=meal_values,
        basal_values=basal_values,
        bolus_values=bolus_values,
    )

    return fig, ax


def calculate_shap_proposed_no_attention(
    model,
    dataloader,
    tp_insulin_basal,
    tp_insulin_bolus,
    tp_meal,
    seq_len_basal_insulin,
    seq_len_bolus_insulin,
    seq_len_carb_intake,
    interval,
    max_samples=30,
    n_samples=15,
    target=None,  # 如果模型输出是多维，建议传具体解释维度，例如 0
):
    import numpy as np
    import torch
    from captum.attr import ShapleyValueSampling

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 时间->步长
    tp_insulin_basal = tp_insulin_basal / interval
    tp_insulin_bolus = tp_insulin_bolus / interval
    tp_meal = tp_meal / interval

    # 窗口转整数
    kernel_size_basal = int(round(seq_len_basal_insulin / interval))
    kernel_size_bolus = int(round(seq_len_bolus_insulin / interval))
    kernel_size_meal = int(round(seq_len_carb_intake / interval))

    # 容器（保留整段，用于方案B可视化）
    attr_cgm_list, attr_basal_list, attr_bolus_list, attr_meal_list = [], [], [], []
    cgm_vals_list, basal_vals_list, bolus_vals_list, meal_vals_list = [], [], [], []

    counted = 0

    # 正确的前向函数：返回整个 batch 输出；target 用 attribute 时指定
    sv = ShapleyValueSampling(
        lambda cgm, basal, bolus, meal: model(cgm, basal, bolus, meal)
    )

    for x, _, _ in dataloader:
        if counted >= max_samples:
            break

        # === 数据读取（整段序列） ===
        cgm = x["cgm"].to(device)  # (B, L)
        basal = x["current_basal"].to(device)  # (B, L)
        bolus = x["current_bolus"].to(device)  # (B, L)
        meal = x["carb_intake"].to(device)  # (B, L)

        # 形状扩展到 (B, L, 1)，与训练保持一致
        cgm = cgm.unsqueeze(2)  # (B, L, 1)
        basal = physiological_layer(
            basal, tp_insulin_basal, kernel_size_basal
        ).unsqueeze(
            2
        )  # (B, L, 1)
        bolus = physiological_layer(
            bolus, tp_insulin_bolus, kernel_size_bolus
        ).unsqueeze(
            2
        )  # (B, L, 1)
        meal = physiological_layer(meal, tp_meal, kernel_size_meal).unsqueeze(
            2
        )  # (B, L, 1)

        basal = basal[:, -1:, :]
        bolus = bolus[:, -1:, :]
        meal = meal[:, -1:, :]

        # 保存输入（整段，方案B用）
        cgm_vals_list.append(cgm.detach().cpu().numpy())
        basal_vals_list.append(basal.detach().cpu().numpy())
        bolus_vals_list.append(bolus.detach().cpu().numpy())
        meal_vals_list.append(meal.detach().cpu().numpy())

        # baseline（同形状）
        baseline = (
            torch.zeros_like(cgm),
            torch.zeros_like(basal),
            torch.zeros_like(bolus),
            torch.zeros_like(meal),
        )

        # 归因：指定 target（如果需要）
        if target is None:
            attribution = sv.attribute(
                inputs=(cgm, basal, bolus, meal),
                n_samples=n_samples,
                baselines=baseline,
            )
        else:
            attribution = sv.attribute(
                inputs=(cgm, basal, bolus, meal),
                n_samples=n_samples,
                baselines=baseline,
                target=target,
            )

        attr_cgm_list.append(attribution[0].detach().cpu().numpy())
        attr_basal_list.append(attribution[1].detach().cpu().numpy())
        attr_bolus_list.append(attribution[2].detach().cpu().numpy())
        attr_meal_list.append(attribution[3].detach().cpu().numpy())

        counted += cgm.shape[0]

    # 拼接 batch 维；得到形状 (N, L, 1)
    import numpy as np

    attr_cgm = np.concatenate(attr_cgm_list, axis=0)
    attr_basal = np.concatenate(attr_basal_list, axis=0)
    attr_bolus = np.concatenate(attr_bolus_list, axis=0)
    attr_meal = np.concatenate(attr_meal_list, axis=0)

    cgm_values = np.concatenate(cgm_vals_list, axis=0)
    basal_values = np.concatenate(basal_vals_list, axis=0)
    bolus_values = np.concatenate(bolus_vals_list, axis=0)
    meal_values = np.concatenate(meal_vals_list, axis=0)

    # 直接把“整段（N,L,1）”交给可视化（方案B 会在里面展平为 N*L 个点）
    fig, ax = vis.plot_interpretability(
        attr_cgm,
        attr_meal,
        attr_basal,
        attr_bolus,
        cgm_values=cgm_values,
        carb_values=meal_values,
        basal_values=basal_values,
        bolus_values=bolus_values,
    )

    return fig, ax


def calculate_shap_proposed_with_phy(
    model,
    dataloader,
    tp_insulin_basal,
    tp_insulin_bolus,
    tp_meal,
    seq_len_basal_insulin,
    seq_len_bolus_insulin,
    seq_len_carb_intake,
    interval,
    max_samples=30,
    n_samples=15,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 时间转换为步长
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

        # === 数据读取与处理 ===
        cgm = x["cgm"].to(device)  # (batch, seq_len)
        basal = x["current_basal"].to(device)
        bolus = x["current_bolus"].to(device)
        meal = x["carb_intake"].to(device)

        # === physiological layer 处理 ===
        basal_processed = physiological_layer(
            basal, tp_insulin_basal, kernel_size_basal
        ).unsqueeze(2)
        bolus_processed = physiological_layer(
            bolus, tp_insulin_bolus, kernel_size_bolus
        ).unsqueeze(2)
        meal_processed = physiological_layer(meal, tp_meal, kernel_size_meal).unsqueeze(
            2
        )

        # CGM 保持整段序列，扩展维度 (batch, seq_len, 1)
        cgm = cgm.unsqueeze(2)

        # 只取 insulin/meal 的最后一帧（保持 3D）
        basal_last = basal_processed[:, -1:, :]
        bolus_last = bolus_processed[:, -1:, :]
        meal_last = meal_processed[:, -1:, :]

        # 保存输入值用于后续可视化
        cgm_values.append(cgm.detach().cpu().numpy())  # 全序列
        basal_values.append(basal_last.detach().cpu().numpy())  # 最后一步
        bolus_values.append(bolus_last.detach().cpu().numpy())
        meal_values.append(meal_last.detach().cpu().numpy())

        # 构造 baseline（与 inputs 同形状）
        baseline = (
            torch.zeros_like(cgm),
            torch.zeros_like(basal_last),
            torch.zeros_like(bolus_last),
            torch.zeros_like(meal_last),
        )

        # attribution 计算
        attribution = sv.attribute(
            inputs=(cgm, basal_last, bolus_last, meal_last),
            n_samples=n_samples,
            baselines=baseline,
        )

        attr_cgm.append(attribution[0].detach().cpu().numpy())
        attr_basal.append(attribution[1].detach().cpu().numpy())
        attr_bolus.append(attribution[2].detach().cpu().numpy())
        attr_meal.append(attribution[3].detach().cpu().numpy())

        count += cgm.shape[0]

    # 拼接输出
    attr_cgm = np.concatenate(attr_cgm, axis=0)
    attr_basal = np.concatenate(attr_basal, axis=0)
    attr_bolus = np.concatenate(attr_bolus, axis=0)
    attr_meal = np.concatenate(attr_meal, axis=0)

    cgm_values = np.concatenate(cgm_values, axis=0)
    basal_values = np.concatenate(basal_values, axis=0)
    bolus_values = np.concatenate(bolus_values, axis=0)
    meal_values = np.concatenate(meal_values, axis=0)

    # 可视化
    fig, ax = vis.plot_interpretability(
        attr_cgm,
        attr_meal,
        attr_basal,
        attr_bolus,
        cgm_values=cgm_values,
        carb_values=meal_values,
        basal_values=basal_values,
        bolus_values=bolus_values,
    )


# interpretability (SHAP values / integrated gradients)
def calculate_shap_for_4channels(
    model,
    dataloader,
    max_samples=30,
    n_samples=15,
):
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
        attr_cgm.append(attribution[0].detach().cpu().numpy())
        attr_carb.append(attribution[1].detach().cpu().numpy())
        attr_basal.append(attribution[2].detach().cpu().numpy())
        attr_bolus.append(attribution[3].detach().cpu().numpy())

        cgm_values.append(cgm.detach().cpu().numpy())
        carb_values.append(carb.detach().cpu().numpy())
        basal_values.append(basal.detach().cpu().numpy())
        bolus_values.append(bolus.detach().cpu().numpy())

        count += cgm.shape[0]

    # concatenate the attribution values and input values
    attr_cgm = np.concatenate(attr_cgm, axis=0)
    attr_carb = np.concatenate(attr_carb, axis=0)
    attr_basal = np.concatenate(attr_basal, axis=0)
    attr_bolus = np.concatenate(attr_bolus, axis=0)

    cgm_values = np.concatenate(cgm_values, axis=0)
    carb_values = np.concatenate(carb_values, axis=0)
    basal_values = np.concatenate(basal_values, axis=0)
    bolus_values = np.concatenate(bolus_values, axis=0)

    # average the attribution values across sequence length
    fig, ax = vis.plot_interpretability(
        attr_cgm,
        attr_carb,
        attr_basal,
        attr_bolus,
        cgm_values,
        carb_values,
        basal_values,
        bolus_values,
    )

    return fig, ax


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
