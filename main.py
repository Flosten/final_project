"""
This is the main module for the personalized baseline and proposed model,
as well as ablation study for the proposed model.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import src.ablation_study as abla
import src.Evaluation as eva
import src.Modeling as mdl
import src.Preprocessing as prep
import src.Visualising as vis


def baseline_modeling(patient_num: int):
    """
    This function trains and evaluates the baseline model for the given patient number.

    Parameters:
        patient_num (int): The patient number for which the model is trained and evaluated.

    Returns:
        tuple: A tuple containing the following elements:
            - baseline_preds (np.ndarray): Predictions made by the baseline model.
            - baseline_truths (np.ndarray): Ground truth values for the test set.
            - baseline_rmse_all (float): RMSE for all predictions.
            - baseline_hyper_rmse (float): RMSE for hyperglycemic values.
            - baseline_hypo_rmse (float): RMSE for hypoglycemic values.
            - baseline_thres_rmse (float): RMSE for values above the threshold.
            - baseline_ud (float): Upward delay of the predictions.
            - baseline_dd (float): Downward delay of the predictions.
            - baseline_fit (float): FIT score of the predictions.
            - baseline_f1_score (float): F1 score of the alarm predictions.
            - baseline_error (np.ndarray): Error between predictions and truths.
            - baseline_shap_perp: SHAP values for model interpretability.
    """
    # set hyperparameters
    train_type = "train"
    test_type = "test"
    folder_4days_path = "train_4days"
    folder_30days_path = "train_30days"
    folder_test_path = "ts-dt1"
    ph = 60
    baseline_seq_len = 300
    interval = 5
    baseline_input_size = 4
    baseline_hidden_size = 416
    baseline_output_size = 1
    baseline_lr = 0.008
    baseline_epochs = 20
    ticks = 2
    dws = 70
    dwe = 10

    # patient 98 and other patients have different datasets
    if patient_num == 98:
        # preprocess data
        baseline_train, baseline_val, baseline_input_scaler, baseline_output_scaler = (
            prep.data_preprocessing_baseline(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                folder_path_30days=folder_30days_path,
                p_num=patient_num,
                seq_len=baseline_seq_len,
                ph=ph,
                interval=interval,
                scaler=None,
            )
        )

        # create the model
        baseline_criterion = nn.MSELoss()
        baseline_model = mdl.PersonalizedModelB1(
            input_size=baseline_input_size,
            hidden_size=baseline_hidden_size,
            output_size=baseline_output_size,
        )
        baseline_optimizer = Adam(baseline_model.parameters(), lr=baseline_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # train the model
        # baseline_model, _, _ = mdl.baseline_model_train(
        #     model=baseline_model,
        #     train_dataloader=baseline_train,
        #     val_dataloader=baseline_val,
        #     optimizer=baseline_optimizer,
        #     criterion=baseline_criterion,
        #     epochs=baseline_epochs,
        #     device=device,
        # )

        # load the pretrained model
        baseline_model.load_state_dict(
            torch.load(
                os.path.join("models", f"baseline_model_patient_{patient_num}.pth")
            )
        )

        # evaluate the model on the test set
        # preprocess test data
        baseline_test = prep.test_data_preprocessing_baseline(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len=baseline_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=baseline_input_scaler,
            output_scaler=baseline_output_scaler,
        )
        # evaluate the model
        (
            baseline_preds,
            baseline_truths,
            baseline_pred_alarms,
            baseline_truth_alarms,
            eval_fig,
            _,
            eval_threshold_fig,
            _,
        ) = mdl.baseline_model_eval(
            model=baseline_model,
            test_dataloader=baseline_test,
            scaler=baseline_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )

        # # save the model
        # torch.save(
        #     baseline_model.state_dict(),
        #     os.path.join("models", f"baseline_model_patient_{patient_num}.pth"),
        # )

        # save the evaluation figures
        eval_fig_name = f"baseline_predictions_patient_{patient_num}.png"
        eval_fig.savefig(os.path.join("figures", eval_fig_name))
        plt.close(eval_fig)

        eval_threshold_fig_name = (
            f"baseline_predictions_threshold_patient_{patient_num}.png"
        )
        eval_threshold_fig.savefig(os.path.join("figures", eval_threshold_fig_name))
        plt.close(eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        baseline_rmse_all = eva.calculate_rmse(baseline_preds, baseline_truths)

        # RMSE for values above the threshold
        baseline_hyper_rmse, baseline_hypo_rmse, baseline_thres_rmse = (
            eva.calculate_threshold_rmse(baseline_preds, baseline_truths)
        )

        # Calculate upward delay and downward delay
        baseline_ud, baseline_dd, _, _ = eva.calculate_ud_dd(
            prediction=baseline_preds,
            truth=baseline_truths,
            ph=ph,
        )

        # calculate FIT
        baseline_fit = eva.calculate_fit(baseline_preds, baseline_truths)

        # calculate f1 score
        baseline_f1_score = eva.evaluate_alarm_multiclass(
            alarm=baseline_pred_alarms,
            truth=baseline_truth_alarms,
            dws=dws,
            dwe=dwe,
        )

        # model interpretability
        baseline_interp_fig, _, baseline_shap_perp = eva.calculate_shap_for_4channels(
            model=baseline_model,
            dataloader=baseline_test,
            max_samples=10,
        )
        baseline_interp_fig_name = (
            f"baseline_interpretability_patient_{patient_num}.png"
        )
        baseline_interp_fig.savefig(os.path.join("figures", baseline_interp_fig_name))
        plt.close(baseline_interp_fig)

        # error between predictions and truths
        baseline_error = abs(baseline_preds - baseline_truths)

    else:
        baseline_train, baseline_val, baseline_input_scaler, baseline_output_scaler = (
            prep.data_preprocessing_baseline_for_99(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                p_num=patient_num,
                seq_len=baseline_seq_len,
                ph=ph,
                interval=interval,
                scaler=None,
            )
        )

        # create the model
        baseline_criterion = nn.MSELoss()
        baseline_model = mdl.PersonalizedModelB1(
            input_size=baseline_input_size,
            hidden_size=baseline_hidden_size,
            output_size=baseline_output_size,
        )
        baseline_optimizer = Adam(baseline_model.parameters(), lr=baseline_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train the model
        baseline_model, _, _ = mdl.baseline_model_train(
            model=baseline_model,
            train_dataloader=baseline_train,
            val_dataloader=baseline_val,
            optimizer=baseline_optimizer,
            criterion=baseline_criterion,
            epochs=baseline_epochs,
            device=device,
        )

        # # save the model
        # torch.save(
        #     baseline_model.state_dict(),
        #     os.path.join("models", f"baseline_model_patient_{patient_num}.pth"),
        # )

        # evaluate the model on the test set
        # preprocess test data
        baseline_test = prep.test_data_preprocessing_baseline_for_99(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len=baseline_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=baseline_input_scaler,
            output_scaler=baseline_output_scaler,
        )
        # evaluate the model
        (
            baseline_preds,
            baseline_truths,
            baseline_pred_alarms,
            baseline_truth_alarms,
            eval_fig,
            _,
            eval_threshold_fig,
            _,
        ) = mdl.baseline_model_eval(
            model=baseline_model,
            test_dataloader=baseline_test,
            scaler=baseline_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )

        # # save the evaluation figures
        # eval_fig_name = f"baseline_predictions_patient_{patient_num}.png"
        # eval_fig.savefig(os.path.join("figures", eval_fig_name))
        # eval_threshold_fig_name = (
        #     f"baseline_predictions_threshold_patient_{patient_num}.png"
        # )
        # eval_threshold_fig.savefig(os.path.join("figures", eval_threshold_fig_name))

        plt.close(eval_fig)
        plt.close(eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        baseline_rmse_all = eva.calculate_rmse(baseline_preds, baseline_truths)

        # RMSE for values above the threshold
        baseline_hyper_rmse, baseline_hypo_rmse, baseline_thres_rmse = (
            eva.calculate_threshold_rmse(baseline_preds, baseline_truths)
        )

        # Calculate upward delay and downward delay
        baseline_ud, baseline_dd, _, _ = eva.calculate_ud_dd(
            prediction=baseline_preds,
            truth=baseline_truths,
            ph=ph,
        )

        # calculate FIT
        baseline_fit = eva.calculate_fit(baseline_preds, baseline_truths)

        # calculate f1 score
        baseline_f1_score = eva.evaluate_alarm_multiclass(
            alarm=baseline_pred_alarms,
            truth=baseline_truth_alarms,
            dws=dws,
            dwe=dwe,
        )

        # model interpretability
        baseline_interp_fig, _, baseline_shap_perp = eva.calculate_shap_for_4channels(
            model=baseline_model,
            dataloader=baseline_test,
            max_samples=1,
        )
        # baseline_interp_fig_name = (
        #     f"baseline_interpretability_patient_{patient_num}.png"
        # )
        # baseline_interp_fig.savefig(os.path.join("figures", baseline_interp_fig_name))
        plt.close(baseline_interp_fig)

        # error between predictions and truths
        baseline_error = abs(baseline_preds - baseline_truths)

    # delete the model to free up memory
    del baseline_model, baseline_optimizer

    # return the results
    return (
        baseline_preds,
        baseline_truths,
        baseline_rmse_all,
        baseline_hyper_rmse,
        baseline_hypo_rmse,
        baseline_thres_rmse,
        baseline_ud,
        baseline_dd,
        baseline_fit,
        baseline_f1_score,
        baseline_error,
        baseline_shap_perp,
    )


def proposed_modeling(patient_num: int):
    """
    This function trains and evaluates the proposed model for the given patient number.

    Parameters:
        patient_num (int): The patient number for which the model is trained and evaluated.

    Returns:
        tuple: A tuple containing the following elements:
            - proposed_preds (np.ndarray): Predictions made by the proposed model.
            - proposed_truths (np.ndarray): Ground truth values for the test set.
            - attention_weights (np.ndarray): Attention weights from the model.
            - proposed_rmse_all (float): RMSE for all predictions.
            - proposed_hyper_rmse (float): RMSE for hyperglycemic values.
            - proposed_hypo_rmse (float): RMSE for hypoglycemic values.
            - proposed_thres_rmse (float): RMSE for values above the threshold.
            - proposed_ud (float): Upward delay of the predictions.
            - proposed_dd (float): Downward delay of the predictions.
            - proposed_fit (float): FIT score of the predictions.
            - proposed_f1_score (float): F1 score of the alarm predictions.
            - proposed_error (np.ndarray): Error between predictions and truths.
            - proposed_shap_perp: SHAP values for model interpretability.
    """
    train_type = "train"
    test_type = "test"
    folder_4days_path = "train_4days"
    folder_30days_path = "train_30days"
    folder_test_path = "ts-dt1"
    ph = 60
    interval = 5
    proposed_cgm_seq_len = 300
    proposed_carb_seq_len = 480
    proposed_basal_seq_len = 600
    proposed_bolus_seq_len = 480
    alpha = 0.3
    beta = 0.0
    ticks = 2
    proposed_input_size = 1
    proposed_hidden_size = 38
    proposed_output_size = 1
    proposed_lr = 0.0018
    proposed_epochs = 20
    proposed_other_lr = 0.0018
    proposed_other_epochs = 40

    dws = 70
    dwe = 10

    # patient 98 and other patients have different datasets
    if patient_num == 98:
        # peak time for input variables
        proposed_peak_carb = 210
        proposed_peak_basal = 400
        proposed_peak_bolus = 290

        # preprocess the data
        proposed_train, proposed_val, proposed_input_scaler, proposed_output_scaler = (
            prep.data_preprocessing_proposed(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                folder_path_30days=folder_30days_path,
                p_num=patient_num,
                seq_len_cgm=proposed_cgm_seq_len,
                seq_len_carb=proposed_carb_seq_len,
                seq_len_basal=proposed_basal_seq_len,
                seq_len_bolus=proposed_bolus_seq_len,
                ph=ph,
                interval=interval,
            )
        )

        # create the model
        proposed_model = mdl.ProposedModel(
            input_size=proposed_input_size,
            hidden_size=proposed_hidden_size,
            output_size=proposed_output_size,
        )
        proposed_optimizer = Adam(proposed_model.parameters(), lr=proposed_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # train the model
        # proposed_model, proposed_lr_fig, _ = mdl.proposed_model_train(
        #     model=proposed_model,
        #     train_dataloader=proposed_train,
        #     val_dataloader=proposed_val,
        #     optimizer=proposed_optimizer,
        #     epochs=proposed_epochs,
        #     alpha=alpha,
        #     beta=beta,
        #     seq_len_carb_intake=proposed_carb_seq_len,
        #     seq_len_basal_insulin=proposed_basal_seq_len,
        #     seq_len_bolus_insulin=proposed_bolus_seq_len,
        #     tp_insulin_basal=proposed_peak_basal,
        #     tp_insulin_bolus=proposed_peak_bolus,
        #     tp_meal=proposed_peak_carb,
        #     interval=interval,
        #     device=device,
        # )
        # # save the model
        # torch.save(
        #     proposed_model.state_dict(),
        #     os.path.join("models", f"proposed_model_patient_{patient_num}.pth"),
        # )
        # # save the learning curve figure
        # proposed_lr_fig_name = f"proposed_learning_curve_patient_{patient_num}.png"
        # proposed_lr_fig.savefig(os.path.join("figures", proposed_lr_fig_name))
        # plt.close(proposed_lr_fig)

        # load the pretrained model
        proposed_model.load_state_dict(
            torch.load(
                os.path.join("models", f"proposed_model_patient_{patient_num}.pth")
            )
        )

        # evaluate the model on the test set
        # preprocess test data
        proposed_test = prep.test_data_preprocessing_proposed(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len_cgm=proposed_cgm_seq_len,
            seq_len_carb=proposed_carb_seq_len,
            seq_len_basal=proposed_basal_seq_len,
            seq_len_bolus=proposed_bolus_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=proposed_input_scaler,
            output_scaler=proposed_output_scaler,
        )
        # evaluate the model
        (
            proposed_preds,
            proposed_truths,
            proposed_pred_alarms,
            proposed_truth_alarms,
            attention_weights,
            proposed_eval_fig,
            _,
            proposed_eval_threshold_fig,
            _,
        ) = mdl.proposed_model_eval(
            model=proposed_model,
            test_dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            interval=interval,
            scaler=proposed_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )
        # save the evaluation figures
        proposed_eval_fig_name = f"proposed_predictions_patient_{patient_num}.png"
        proposed_eval_fig.savefig(os.path.join("figures", proposed_eval_fig_name))
        plt.close(proposed_eval_fig)

        proposed_eval_threshold_fig_name = (
            f"proposed_predictions_threshold_patient_{patient_num}.png"
        )
        proposed_eval_threshold_fig.savefig(
            os.path.join("figures", proposed_eval_threshold_fig_name)
        )
        plt.close(proposed_eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        proposed_rmse_all = eva.calculate_rmse(proposed_preds, proposed_truths)

        # RMSE for values above the threshold
        proposed_hyper_rmse, proposed_hypo_rmse, proposed_thres_rmse = (
            eva.calculate_threshold_rmse(proposed_preds, proposed_truths)
        )

        # Calculate upward delay and downward delay
        proposed_ud, proposed_dd, _, _ = eva.calculate_ud_dd(
            prediction=proposed_preds,
            truth=proposed_truths,
            ph=ph,
        )

        # calculate FIT
        proposed_fit = eva.calculate_fit(proposed_preds, proposed_truths)

        # calculate f1 score
        proposed_f1_score = eva.evaluate_alarm_multiclass(
            alarm=proposed_pred_alarms,
            truth=proposed_truth_alarms,
            dws=dws,
            dwe=dwe,
        )

        # model interpretability
        proposed_interp_fig, _, proposed_shap_perp = eva.calculate_shap_proposed(
            model=proposed_model,
            dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            seq_len_carb_intake=proposed_carb_seq_len,
            interval=interval,
            max_samples=10,
        )
        proposed_interp_fig_name = (
            f"proposed_interpretability_patient_{patient_num}.png"
        )
        proposed_interp_fig.savefig(os.path.join("figures", proposed_interp_fig_name))
        plt.close(proposed_interp_fig)

        # error between predictions and truths
        proposed_error = abs(proposed_preds - proposed_truths)

    else:
        carb_folder = "ts-dtM"
        bolus_folder = "ts-dtI"

        # peak time for input variables
        proposed_peak_basal = 400
        proposed_peak_carb = mdl.get_peak_time(
            patient_num=patient_num,
            var_type=carb_folder,
        )
        proposed_peak_bolus = mdl.get_peak_time(
            patient_num=patient_num,
            var_type=bolus_folder,
        )
        # avoid peak time exceeding the sequence length
        proposed_peak_carb = min(proposed_peak_carb, proposed_carb_seq_len)
        proposed_peak_bolus = min(proposed_peak_bolus, proposed_bolus_seq_len)

        # preprocess the data
        proposed_train, proposed_val, proposed_input_scaler, proposed_output_scaler = (
            prep.data_preprocessing_proposed_for_99(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                p_num=patient_num,
                seq_len_cgm=proposed_cgm_seq_len,
                seq_len_carb=proposed_carb_seq_len,
                seq_len_basal=proposed_basal_seq_len,
                seq_len_bolus=proposed_bolus_seq_len,
                ph=ph,
                interval=interval,
            )
        )

        # create the model
        proposed_model = mdl.ProposedModel(
            input_size=proposed_input_size,
            hidden_size=proposed_hidden_size,
            output_size=proposed_output_size,
        )
        proposed_optimizer = Adam(proposed_model.parameters(), lr=proposed_other_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train the model
        proposed_model, _, _ = mdl.proposed_model_train(
            model=proposed_model,
            train_dataloader=proposed_train,
            val_dataloader=proposed_val,
            optimizer=proposed_optimizer,
            epochs=proposed_other_epochs,
            alpha=alpha,
            beta=beta,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            interval=interval,
            device=device,
        )
        # # save the model
        # torch.save(
        #     proposed_model.state_dict(),
        #     os.path.join("models", f"proposed_model_patient_{patient_num}.pth"),
        # )

        # evaluate the model on the test set
        # preprocess test data
        proposed_test = prep.test_data_preprocessing_proposed_for_99(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len_cgm=proposed_cgm_seq_len,
            seq_len_carb=proposed_carb_seq_len,
            seq_len_basal=proposed_basal_seq_len,
            seq_len_bolus=proposed_bolus_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=proposed_input_scaler,
            output_scaler=proposed_output_scaler,
        )
        # evaluate the model
        (
            proposed_preds,
            proposed_truths,
            proposed_pred_alarms,
            proposed_truth_alarms,
            attention_weights,
            proposed_eval_fig,
            _,
            proposed_eval_threshold_fig,
            _,
        ) = mdl.proposed_model_eval(
            model=proposed_model,
            test_dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            interval=interval,
            scaler=proposed_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )

        # # save the evaluation figures
        # proposed_eval_fig_name = f"proposed_predictions_patient_{patient_num}.png"
        # proposed_eval_fig.savefig(os.path.join("figures", proposed_eval_fig_name))
        # proposed_eval_threshold_fig_name = (
        #     f"proposed_predictions_threshold_patient_{patient_num}.png"
        # )
        # proposed_eval_threshold_fig.savefig(
        #     os.path.join("figures", proposed_eval_threshold_fig_name)
        # )
        plt.close(proposed_eval_fig)
        plt.close(proposed_eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        proposed_rmse_all = eva.calculate_rmse(proposed_preds, proposed_truths)
        # RMSE for values above the threshold
        proposed_hyper_rmse, proposed_hypo_rmse, proposed_thres_rmse = (
            eva.calculate_threshold_rmse(proposed_preds, proposed_truths)
        )
        # Calculate upward delay and downward delay
        proposed_ud, proposed_dd, _, _ = eva.calculate_ud_dd(
            prediction=proposed_preds,
            truth=proposed_truths,
            ph=ph,
        )
        # calculate FIT
        proposed_fit = eva.calculate_fit(proposed_preds, proposed_truths)
        # calculate f1 score
        proposed_f1_score = eva.evaluate_alarm_multiclass(
            alarm=proposed_pred_alarms,
            truth=proposed_truth_alarms,
            dws=dws,
            dwe=dwe,
        )
        # model interpretability
        proposed_interp_fig, _, proposed_shap_perp = eva.calculate_shap_proposed(
            model=proposed_model,
            dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            seq_len_carb_intake=proposed_carb_seq_len,
            interval=interval,
            max_samples=1,
        )
        # proposed_interp_fig_name = (
        #     f"proposed_interpretability_patient_{patient_num}.png"
        # )
        # proposed_interp_fig.savefig(os.path.join("figures", proposed_interp_fig_name))
        plt.close(proposed_interp_fig)

        # error between predictions and truths
        proposed_error = abs(proposed_preds - proposed_truths)

    # delete the model to free up memory
    del proposed_model, proposed_optimizer

    # return the results
    return (
        proposed_preds,
        proposed_truths,
        attention_weights,
        proposed_rmse_all,
        proposed_hyper_rmse,
        proposed_hypo_rmse,
        proposed_thres_rmse,
        proposed_ud,
        proposed_dd,
        proposed_fit,
        proposed_f1_score,
        proposed_error,
        proposed_shap_perp,
    )


def ablation_study_loss_function(patient_num: int):
    """Ablation study for the loss function in the proposed model."""
    train_type = "train"
    test_type = "test"
    folder_4days_path = "train_4days"
    folder_30days_path = "train_30days"
    folder_test_path = "ts-dt1"
    ph = 60
    interval = 5
    proposed_cgm_seq_len = 300
    proposed_carb_seq_len = 480
    proposed_basal_seq_len = 600
    proposed_bolus_seq_len = 480
    alpha = 0.3
    beta = 0.0
    ticks = 2
    proposed_input_size = 1
    proposed_hidden_size = 38
    proposed_output_size = 1
    proposed_lr = 0.0018
    proposed_epochs = 20
    proposed_other_lr = 0.0018
    proposed_other_epochs = 40

    dws = 70
    dwe = 10

    # patient 98 and other patients have different datasets
    if patient_num == 98:
        # peak time for input variables
        proposed_peak_carb = 210
        proposed_peak_basal = 400
        proposed_peak_bolus = 290

        # preprocess the data
        proposed_train, proposed_val, proposed_input_scaler, proposed_output_scaler = (
            prep.data_preprocessing_proposed(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                folder_path_30days=folder_30days_path,
                p_num=patient_num,
                seq_len_cgm=proposed_cgm_seq_len,
                seq_len_carb=proposed_carb_seq_len,
                seq_len_basal=proposed_basal_seq_len,
                seq_len_bolus=proposed_bolus_seq_len,
                ph=ph,
                interval=interval,
            )
        )

        # create the model
        proposed_model = mdl.ProposedModel(
            input_size=proposed_input_size,
            hidden_size=proposed_hidden_size,
            output_size=proposed_output_size,
        )
        proposed_optimizer = Adam(proposed_model.parameters(), lr=proposed_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # train the model
        # proposed_model, _, _ = abla.proposed_model_train_loss(
        #     model=proposed_model,
        #     train_dataloader=proposed_train,
        #     val_dataloader=proposed_val,
        #     optimizer=proposed_optimizer,
        #     epochs=proposed_epochs,
        #     alpha=alpha,
        #     beta=beta,
        #     seq_len_carb_intake=proposed_carb_seq_len,
        #     seq_len_basal_insulin=proposed_basal_seq_len,
        #     seq_len_bolus_insulin=proposed_bolus_seq_len,
        #     tp_insulin_basal=proposed_peak_basal,
        #     tp_insulin_bolus=proposed_peak_bolus,
        #     tp_meal=proposed_peak_carb,
        #     interval=interval,
        #     device=device,
        # )
        # # save the model
        # torch.save(
        #     proposed_model.state_dict(),
        #     os.path.join(
        #         "models", f"abla_loss_proposed_model_patient_{patient_num}.pth"
        #     ),
        # )
        # # save the learning curve figure
        # proposed_lr_fig_name = (
        #     f"abla_loss_proposed_learning_curve_patient_{patient_num}.png"
        # )
        # proposed_lr_fig.savefig(os.path.join("figures", proposed_lr_fig_name))
        # plt.close(proposed_lr_fig)

        # load the pretrained model
        proposed_model.load_state_dict(
            torch.load(
                os.path.join(
                    "models", f"abla_loss_proposed_model_patient_{patient_num}.pth"
                )
            )
        )

        # evaluate the model on the test set
        # preprocess test data
        proposed_test = prep.test_data_preprocessing_proposed(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len_cgm=proposed_cgm_seq_len,
            seq_len_carb=proposed_carb_seq_len,
            seq_len_basal=proposed_basal_seq_len,
            seq_len_bolus=proposed_bolus_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=proposed_input_scaler,
            output_scaler=proposed_output_scaler,
        )
        # evaluate the model
        (
            proposed_preds,
            proposed_truths,
            proposed_pred_alarms,
            proposed_truth_alarms,
            attention_weights,
            proposed_eval_fig,
            _,
            proposed_eval_threshold_fig,
            _,
        ) = abla.proposed_model_loss_eval(
            model=proposed_model,
            test_dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            interval=interval,
            scaler=proposed_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )
        # save the evaluation figures
        proposed_eval_fig_name = (
            f"ablation_study_loss_function_patient_{patient_num}.png"
        )
        proposed_eval_fig.savefig(os.path.join("figures", proposed_eval_fig_name))
        plt.close(proposed_eval_fig)

        proposed_eval_threshold_fig_name = (
            f"ablation_study_loss_function_threshold_patient_{patient_num}.png"
        )
        proposed_eval_threshold_fig.savefig(
            os.path.join("figures", proposed_eval_threshold_fig_name)
        )
        plt.close(proposed_eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        proposed_rmse_all = eva.calculate_rmse(proposed_preds, proposed_truths)

        # RMSE for values above the threshold
        proposed_hyper_rmse, proposed_hypo_rmse, proposed_thres_rmse = (
            eva.calculate_threshold_rmse(proposed_preds, proposed_truths)
        )

        # Calculate upward delay and downward delay
        proposed_ud, proposed_dd, _, _ = eva.calculate_ud_dd(
            prediction=proposed_preds,
            truth=proposed_truths,
            ph=ph,
        )

        # Calculate FIT
        proposed_fit = eva.calculate_fit(proposed_preds, proposed_truths)

        # calculate f1 score
        proposed_f1_score = eva.evaluate_alarm_multiclass(
            alarm=proposed_pred_alarms,
            truth=proposed_truth_alarms,
            dws=dws,
            dwe=dwe,
        )

        # model interpretability
        proposed_interp_fig, _, proposed_shap_perp = eva.calculate_shap_proposed(
            model=proposed_model,
            dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            seq_len_carb_intake=proposed_carb_seq_len,
            interval=interval,
            max_samples=10,
            n_samples=1,
        )
        proposed_interp_fig_name = (
            f"ablation_study_loss_function_interpretability_patient_{patient_num}.png"
        )
        proposed_interp_fig.savefig(os.path.join("figures", proposed_interp_fig_name))
        plt.close(proposed_interp_fig)

        # error between predictions and truths
        proposed_error = abs(proposed_preds - proposed_truths)

    else:
        carb_folder = "ts-dtM"
        bolus_folder = "ts-dtI"

        # peak time for input variables
        proposed_peak_basal = 400
        proposed_peak_carb = mdl.get_peak_time(
            patient_num=patient_num,
            var_type=carb_folder,
        )
        proposed_peak_bolus = mdl.get_peak_time(
            patient_num=patient_num,
            var_type=bolus_folder,
        )
        # avoid peak time exceeding the sequence length
        proposed_peak_carb = min(proposed_peak_carb, proposed_carb_seq_len)
        proposed_peak_bolus = min(proposed_peak_bolus, proposed_bolus_seq_len)

        # preprocess the data
        proposed_train, proposed_val, proposed_input_scaler, proposed_output_scaler = (
            prep.data_preprocessing_proposed_for_99(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                p_num=patient_num,
                seq_len_cgm=proposed_cgm_seq_len,
                seq_len_carb=proposed_carb_seq_len,
                seq_len_basal=proposed_basal_seq_len,
                seq_len_bolus=proposed_bolus_seq_len,
                ph=ph,
                interval=interval,
            )
        )

        # create the model
        proposed_model = mdl.ProposedModel(
            input_size=proposed_input_size,
            hidden_size=proposed_hidden_size,
            output_size=proposed_output_size,
        )
        proposed_optimizer = Adam(proposed_model.parameters(), lr=proposed_other_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train the model
        proposed_model, _, _ = abla.proposed_model_train_loss(
            model=proposed_model,
            train_dataloader=proposed_train,
            val_dataloader=proposed_val,
            optimizer=proposed_optimizer,
            epochs=proposed_other_epochs,
            alpha=alpha,
            beta=beta,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            interval=interval,
            device=device,
        )
        # # save the model
        # torch.save(
        #     proposed_model.state_dict(),
        #     os.path.join(
        #         "models", f"abla_loss_proposed_model_patient_{patient_num}.pth"
        #     ),
        # )

        # evaluate the model on the test set
        # preprocess test data
        proposed_test = prep.test_data_preprocessing_proposed_for_99(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len_cgm=proposed_cgm_seq_len,
            seq_len_carb=proposed_carb_seq_len,
            seq_len_basal=proposed_basal_seq_len,
            seq_len_bolus=proposed_bolus_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=proposed_input_scaler,
            output_scaler=proposed_output_scaler,
        )
        # evaluate the model
        (
            proposed_preds,
            proposed_truths,
            proposed_pred_alarms,
            proposed_truth_alarms,
            attention_weights,
            proposed_eval_fig,
            _,
            proposed_eval_threshold_fig,
            _,
        ) = abla.proposed_model_loss_eval(
            model=proposed_model,
            test_dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            interval=interval,
            scaler=proposed_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )

        # # save the evaluation figures
        # proposed_eval_fig_name = (
        #     f"abla_loss_proposed_predictions_patient_{patient_num}.png"
        # )
        # proposed_eval_fig.savefig(os.path.join("figures", proposed_eval_fig_name))
        # proposed_eval_threshold_fig_name = (
        #     f"abla_loss_proposed_predictions_threshold_patient_{patient_num}.png"
        # )
        # proposed_eval_threshold_fig.savefig(
        #     os.path.join("figures", proposed_eval_threshold_fig_name)
        # )
        plt.close(proposed_eval_fig)
        plt.close(proposed_eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        proposed_rmse_all = eva.calculate_rmse(proposed_preds, proposed_truths)
        # RMSE for values above the threshold
        proposed_hyper_rmse, proposed_hypo_rmse, proposed_thres_rmse = (
            eva.calculate_threshold_rmse(proposed_preds, proposed_truths)
        )
        # Calculate upward delay and downward delay
        proposed_ud, proposed_dd, _, _ = eva.calculate_ud_dd(
            prediction=proposed_preds,
            truth=proposed_truths,
            ph=ph,
        )
        # Calculate FIT
        proposed_fit = eva.calculate_fit(proposed_preds, proposed_truths)
        # calculate f1 score
        proposed_f1_score = eva.evaluate_alarm_multiclass(
            alarm=proposed_pred_alarms,
            truth=proposed_truth_alarms,
            dws=dws,
            dwe=dwe,
        )
        # model interpretability
        proposed_interp_fig, _, proposed_shap_perp = eva.calculate_shap_proposed(
            model=proposed_model,
            dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            seq_len_carb_intake=proposed_carb_seq_len,
            interval=interval,
            max_samples=1,
        )
        # proposed_interp_fig_name = (
        #     f"abla_loss_proposed_interpretability_patient_{patient_num}.png"
        # )
        # proposed_interp_fig.savefig(os.path.join("figures", proposed_interp_fig_name))
        plt.close(proposed_interp_fig)

        # error between predictions and truths
        proposed_error = abs(proposed_preds - proposed_truths)

    # delete the model to free up memory
    del proposed_model, proposed_optimizer

    # return the results
    return (
        proposed_preds,
        proposed_truths,
        attention_weights,
        proposed_rmse_all,
        proposed_hyper_rmse,
        proposed_hypo_rmse,
        proposed_thres_rmse,
        proposed_ud,
        proposed_dd,
        proposed_fit,
        proposed_f1_score,
        proposed_error,
        proposed_shap_perp,
    )


def ablation_study_phy_layer(patient_num: int):
    """Ablation study for the Physiological Modeling Layer in the proposed model."""
    train_type = "train"
    test_type = "test"
    folder_4days_path = "train_4days"
    folder_30days_path = "train_30days"
    folder_test_path = "ts-dt1"
    ph = 60
    interval = 5
    proposed_cgm_seq_len = 300
    proposed_carb_seq_len = 480
    proposed_basal_seq_len = 600
    proposed_bolus_seq_len = 480
    alpha = 0.3
    beta = 0.0
    ticks = 2
    proposed_input_size = 1
    proposed_hidden_size = 38
    proposed_output_size = 1
    proposed_lr = 0.0018
    proposed_epochs = 20
    proposed_other_lr = 0.0018
    proposed_other_epochs = 40

    dws = 70
    dwe = 10

    # patient 98 and other patients have different datasets
    if patient_num == 98:
        # peak time for input variables
        proposed_peak_carb = 210
        proposed_peak_basal = 400
        proposed_peak_bolus = 290

        # preprocess the data
        proposed_train, proposed_val, proposed_input_scaler, proposed_output_scaler = (
            prep.data_preprocessing_proposed(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                folder_path_30days=folder_30days_path,
                p_num=patient_num,
                seq_len_cgm=proposed_cgm_seq_len,
                seq_len_carb=proposed_carb_seq_len,
                seq_len_basal=proposed_basal_seq_len,
                seq_len_bolus=proposed_bolus_seq_len,
                ph=ph,
                interval=interval,
            )
        )

        # create the model
        proposed_model = abla.ProposedModel_abl_loss(
            input_size=proposed_input_size,
            hidden_size=proposed_hidden_size,
            output_size=proposed_output_size,
        )
        proposed_optimizer = Adam(proposed_model.parameters(), lr=proposed_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # train the model
        # proposed_model, proposed_lr_fig, _ = abla.proposed_model_train_phy(
        #     model=proposed_model,
        #     train_dataloader=proposed_train,
        #     val_dataloader=proposed_val,
        #     optimizer=proposed_optimizer,
        #     epochs=proposed_epochs,
        #     alpha=alpha,
        #     beta=beta,
        #     seq_len_carb_intake=proposed_carb_seq_len,
        #     seq_len_basal_insulin=proposed_basal_seq_len,
        #     seq_len_bolus_insulin=proposed_bolus_seq_len,
        #     tp_insulin_basal=proposed_peak_basal,
        #     tp_insulin_bolus=proposed_peak_bolus,
        #     tp_meal=proposed_peak_carb,
        #     interval=interval,
        #     device=device,
        # )
        # # save the model
        # torch.save(
        #     proposed_model.state_dict(),
        #     os.path.join(
        #         "models", f"abla_phy_proposed_model_patient_{patient_num}.pth"
        #     ),
        # )
        # # save the learning curve figure
        # proposed_lr_fig_name = (
        #     f"abla_phy_proposed_learning_curve_patient_{patient_num}.png"
        # )
        # proposed_lr_fig.savefig(os.path.join("figures", proposed_lr_fig_name))
        # plt.close(proposed_lr_fig)

        # load the pretrained model
        proposed_model.load_state_dict(
            torch.load(
                os.path.join(
                    "models", f"abla_phy_proposed_model_patient_{patient_num}.pth"
                )
            )
        )

        # evaluate the model on the test set
        # preprocess test data
        proposed_test = prep.test_data_preprocessing_proposed(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len_cgm=proposed_cgm_seq_len,
            seq_len_carb=proposed_carb_seq_len,
            seq_len_basal=proposed_basal_seq_len,
            seq_len_bolus=proposed_bolus_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=proposed_input_scaler,
            output_scaler=proposed_output_scaler,
        )
        # evaluate the model
        (
            proposed_preds,
            proposed_truths,
            proposed_pred_alarms,
            proposed_truth_alarms,
            attention_weights,
            proposed_eval_fig,
            _,
            proposed_eval_threshold_fig,
            _,
        ) = abla.proposed_model_eval_phy(
            model=proposed_model,
            test_dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            interval=interval,
            scaler=proposed_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )
        # save the evaluation figures
        proposed_eval_fig_name = f"ablation_study_phy_patient_{patient_num}.png"
        proposed_eval_fig.savefig(os.path.join("figures", proposed_eval_fig_name))
        plt.close(proposed_eval_fig)

        proposed_eval_threshold_fig_name = (
            f"ablation_study_phy_threshold_patient_{patient_num}.png"
        )
        proposed_eval_threshold_fig.savefig(
            os.path.join("figures", proposed_eval_threshold_fig_name)
        )
        plt.close(proposed_eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        proposed_rmse_all = eva.calculate_rmse(proposed_preds, proposed_truths)

        # RMSE for values above the threshold
        proposed_hyper_rmse, proposed_hypo_rmse, proposed_thres_rmse = (
            eva.calculate_threshold_rmse(proposed_preds, proposed_truths)
        )

        # Calculate upward delay and downward delay
        proposed_ud, proposed_dd, _, _ = eva.calculate_ud_dd(
            prediction=proposed_preds,
            truth=proposed_truths,
            ph=ph,
        )

        # Calculate FIT
        proposed_fit = eva.calculate_fit(proposed_preds, proposed_truths)

        # calculate f1 score
        proposed_f1_score = eva.evaluate_alarm_multiclass(
            alarm=proposed_pred_alarms,
            truth=proposed_truth_alarms,
            dws=dws,
            dwe=dwe,
        )

        # model interpretability
        proposed_interp_fig, _, proposed_shap_perp = eva.calculate_shap_proposed_no_phy(
            model=proposed_model,
            dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            seq_len_carb_intake=proposed_carb_seq_len,
            interval=interval,
            max_samples=10,
        )
        proposed_interp_fig_name = (
            f"ablation_study_phy_interpretability_patient_{patient_num}.png"
        )
        proposed_interp_fig.savefig(os.path.join("figures", proposed_interp_fig_name))
        plt.close(proposed_interp_fig)

        # error between predictions and truths
        proposed_error = abs(proposed_preds - proposed_truths)

    else:
        carb_folder = "ts-dtM"
        bolus_folder = "ts-dtI"

        # peak time for input variables
        proposed_peak_basal = 400
        proposed_peak_carb = mdl.get_peak_time(
            patient_num=patient_num,
            var_type=carb_folder,
        )
        proposed_peak_bolus = mdl.get_peak_time(
            patient_num=patient_num,
            var_type=bolus_folder,
        )
        # avoid peak time exceeding the sequence length
        proposed_peak_carb = min(proposed_peak_carb, proposed_carb_seq_len)
        proposed_peak_bolus = min(proposed_peak_bolus, proposed_bolus_seq_len)

        # preprocess the data
        proposed_train, proposed_val, proposed_input_scaler, proposed_output_scaler = (
            prep.data_preprocessing_proposed_for_99(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                p_num=patient_num,
                seq_len_cgm=proposed_cgm_seq_len,
                seq_len_carb=proposed_carb_seq_len,
                seq_len_basal=proposed_basal_seq_len,
                seq_len_bolus=proposed_bolus_seq_len,
                ph=ph,
                interval=interval,
            )
        )

        # create the model
        proposed_model = abla.ProposedModel_abl_loss(
            input_size=proposed_input_size,
            hidden_size=proposed_hidden_size,
            output_size=proposed_output_size,
        )
        proposed_optimizer = Adam(proposed_model.parameters(), lr=proposed_other_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train the model
        proposed_model, _, _ = abla.proposed_model_train_phy(
            model=proposed_model,
            train_dataloader=proposed_train,
            val_dataloader=proposed_val,
            optimizer=proposed_optimizer,
            epochs=proposed_other_epochs,
            alpha=alpha,
            beta=beta,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            interval=interval,
            device=device,
        )
        # # save the model
        # torch.save(
        #     proposed_model.state_dict(),
        #     os.path.join(
        #         "models", f"abla_phy_proposed_model_patient_{patient_num}.pth"
        #     ),
        # )

        # evaluate the model on the test set
        # preprocess test data
        proposed_test = prep.test_data_preprocessing_proposed_for_99(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len_cgm=proposed_cgm_seq_len,
            seq_len_carb=proposed_carb_seq_len,
            seq_len_basal=proposed_basal_seq_len,
            seq_len_bolus=proposed_bolus_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=proposed_input_scaler,
            output_scaler=proposed_output_scaler,
        )
        # evaluate the model
        (
            proposed_preds,
            proposed_truths,
            proposed_pred_alarms,
            proposed_truth_alarms,
            attention_weights,
            proposed_eval_fig,
            _,
            proposed_eval_threshold_fig,
            _,
        ) = abla.proposed_model_eval_phy(
            model=proposed_model,
            test_dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            interval=interval,
            scaler=proposed_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )

        # # save the evaluation figures
        # proposed_eval_fig_name = f"abla_phy_predictions_patient_{patient_num}.png"
        # proposed_eval_fig.savefig(os.path.join("figures", proposed_eval_fig_name))
        # proposed_eval_threshold_fig_name = (
        #     f"abla_phy_predictions_threshold_patient_{patient_num}.png"
        # )
        # proposed_eval_threshold_fig.savefig(
        #     os.path.join("figures", proposed_eval_threshold_fig_name)
        # )
        plt.close(proposed_eval_fig)
        plt.close(proposed_eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        proposed_rmse_all = eva.calculate_rmse(proposed_preds, proposed_truths)
        # RMSE for values above the threshold
        proposed_hyper_rmse, proposed_hypo_rmse, proposed_thres_rmse = (
            eva.calculate_threshold_rmse(proposed_preds, proposed_truths)
        )
        # Calculate upward delay and downward delay
        proposed_ud, proposed_dd, _, _ = eva.calculate_ud_dd(
            prediction=proposed_preds,
            truth=proposed_truths,
            ph=ph,
        )
        # Calculate FIT
        proposed_fit = eva.calculate_fit(proposed_preds, proposed_truths)
        # calculate f1 score
        proposed_f1_score = eva.evaluate_alarm_multiclass(
            alarm=proposed_pred_alarms,
            truth=proposed_truth_alarms,
            dws=dws,
            dwe=dwe,
        )
        # model interpretability
        proposed_interp_fig, _, proposed_shap_perp = eva.calculate_shap_proposed_no_phy(
            model=proposed_model,
            dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            seq_len_carb_intake=proposed_carb_seq_len,
            interval=interval,
            max_samples=1,
        )
        # proposed_interp_fig_name = (
        #     f"abla_phy_interpretability_patient_{patient_num}.png"
        # )
        # proposed_interp_fig.savefig(os.path.join("figures", proposed_interp_fig_name))
        plt.close(proposed_interp_fig)

        # error between predictions and truths
        proposed_error = abs(proposed_preds - proposed_truths)

    # delete the model to free up memory
    del proposed_model, proposed_optimizer

    # return the results
    return (
        proposed_preds,
        proposed_truths,
        attention_weights,
        proposed_rmse_all,
        proposed_hyper_rmse,
        proposed_hypo_rmse,
        proposed_thres_rmse,
        proposed_ud,
        proposed_dd,
        proposed_fit,
        proposed_f1_score,
        proposed_error,
        proposed_shap_perp,
    )


def ablation_study_dual_input(patient_num: int):
    """Ablation study for the Dual Input in the proposed model."""
    train_type = "train"
    test_type = "test"
    folder_4days_path = "train_4days"
    folder_30days_path = "train_30days"
    folder_test_path = "ts-dt1"
    ph = 60
    interval = 5
    proposed_cgm_seq_len = 300
    proposed_carb_seq_len = 480
    proposed_basal_seq_len = 600
    proposed_bolus_seq_len = 480
    alpha = 0.3
    beta = 0.0
    ticks = 2
    proposed_input_size = 1
    proposed_hidden_size = 38
    proposed_output_size = 1
    proposed_lr = 0.0018
    proposed_epochs = 20
    proposed_other_lr = 0.0018
    proposed_other_epochs = 40

    dws = 70
    dwe = 10

    # patient 98 and other patients have different datasets
    if patient_num == 98:
        # peak time for input variables
        proposed_peak_carb = 210
        proposed_peak_basal = 400
        proposed_peak_bolus = 290

        # preprocess the data
        proposed_train, proposed_val, proposed_input_scaler, proposed_output_scaler = (
            prep.data_preprocessing_proposed(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                folder_path_30days=folder_30days_path,
                p_num=patient_num,
                seq_len_cgm=proposed_cgm_seq_len,
                seq_len_carb=proposed_carb_seq_len,
                seq_len_basal=proposed_basal_seq_len,
                seq_len_bolus=proposed_bolus_seq_len,
                ph=ph,
                interval=interval,
            )
        )

        # create the model
        proposed_model = abla.ProposedModel_dual_input(
            input_size=proposed_input_size,
            hidden_size=proposed_hidden_size,
            output_size=proposed_output_size,
        )
        proposed_optimizer = Adam(proposed_model.parameters(), lr=proposed_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # train the model
        # proposed_model, proposed_lr_fig, _ = abla.proposed_model_train_dual(
        #     model=proposed_model,
        #     train_dataloader=proposed_train,
        #     val_dataloader=proposed_val,
        #     optimizer=proposed_optimizer,
        #     epochs=proposed_epochs,
        #     alpha=alpha,
        #     beta=beta,
        #     seq_len_carb_intake=proposed_carb_seq_len,
        #     seq_len_basal_insulin=proposed_basal_seq_len,
        #     seq_len_bolus_insulin=proposed_bolus_seq_len,
        #     tp_insulin_basal=proposed_peak_basal,
        #     tp_insulin_bolus=proposed_peak_bolus,
        #     tp_meal=proposed_peak_carb,
        #     interval=interval,
        #     device=device,
        # )
        # # save the model
        # torch.save(
        #     proposed_model.state_dict(),
        #     os.path.join(
        #         "models", f"abla_dual_proposed_model_patient_{patient_num}.pth"
        #     ),
        # )
        # # save the learning curve figure
        # proposed_lr_fig_name = (
        #     f"abla_dual_proposed_learning_curve_patient_{patient_num}.png"
        # )
        # proposed_lr_fig.savefig(os.path.join("figures", proposed_lr_fig_name))
        # plt.close(proposed_lr_fig)

        # load the pretrained model
        proposed_model.load_state_dict(
            torch.load(
                os.path.join(
                    "models", f"abla_dual_proposed_model_patient_{patient_num}.pth"
                )
            )
        )

        # evaluate the model on the test set
        # preprocess test data
        proposed_test = prep.test_data_preprocessing_proposed(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len_cgm=proposed_cgm_seq_len,
            seq_len_carb=proposed_carb_seq_len,
            seq_len_basal=proposed_basal_seq_len,
            seq_len_bolus=proposed_bolus_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=proposed_input_scaler,
            output_scaler=proposed_output_scaler,
        )
        # evaluate the model
        (
            proposed_preds,
            proposed_truths,
            proposed_pred_alarms,
            proposed_truth_alarms,
            attention_weights,
            proposed_eval_fig,
            _,
            proposed_eval_threshold_fig,
            _,
        ) = abla.proposed_model_eval_dual(
            model=proposed_model,
            test_dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            interval=interval,
            scaler=proposed_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )
        # save the evaluation figures
        proposed_eval_fig_name = f"ablation_study_dual_patient_{patient_num}.png"
        proposed_eval_fig.savefig(os.path.join("figures", proposed_eval_fig_name))
        plt.close(proposed_eval_fig)

        proposed_eval_threshold_fig_name = (
            f"ablation_study_dual_threshold_patient_{patient_num}.png"
        )
        proposed_eval_threshold_fig.savefig(
            os.path.join("figures", proposed_eval_threshold_fig_name)
        )
        plt.close(proposed_eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        proposed_rmse_all = eva.calculate_rmse(proposed_preds, proposed_truths)

        # RMSE for values above the threshold
        proposed_hyper_rmse, proposed_hypo_rmse, proposed_thres_rmse = (
            eva.calculate_threshold_rmse(proposed_preds, proposed_truths)
        )

        # Calculate upward delay and downward delay
        proposed_ud, proposed_dd, _, _ = eva.calculate_ud_dd(
            prediction=proposed_preds,
            truth=proposed_truths,
            ph=ph,
        )

        # Calculate FIT
        proposed_fit = eva.calculate_fit(proposed_preds, proposed_truths)

        # calculate f1 score
        proposed_f1_score = eva.evaluate_alarm_multiclass(
            alarm=proposed_pred_alarms,
            truth=proposed_truth_alarms,
            dws=dws,
            dwe=dwe,
        )

        # model interpretability
        proposed_interp_fig, _, proposed_shap_perp = (
            eva.calculate_shap_proposed_no_dual_input(
                model=proposed_model,
                dataloader=proposed_test,
                tp_insulin_basal=proposed_peak_basal,
                tp_insulin_bolus=proposed_peak_bolus,
                tp_meal=proposed_peak_carb,
                seq_len_basal_insulin=proposed_basal_seq_len,
                seq_len_bolus_insulin=proposed_bolus_seq_len,
                seq_len_carb_intake=proposed_carb_seq_len,
                interval=interval,
                max_samples=10,
            )
        )
        proposed_interp_fig_name = (
            f"ablation_study_dual_interpretability_patient_{patient_num}.png"
        )
        proposed_interp_fig.savefig(os.path.join("figures", proposed_interp_fig_name))
        plt.close(proposed_interp_fig)

        # error between predictions and truths
        proposed_error = abs(proposed_preds - proposed_truths)

    else:
        carb_folder = "ts-dtM"
        bolus_folder = "ts-dtI"

        # peak time for input variables
        proposed_peak_basal = 400
        proposed_peak_carb = mdl.get_peak_time(
            patient_num=patient_num,
            var_type=carb_folder,
        )
        proposed_peak_bolus = mdl.get_peak_time(
            patient_num=patient_num,
            var_type=bolus_folder,
        )
        # avoid peak time exceeding the sequence length
        proposed_peak_carb = min(proposed_peak_carb, proposed_carb_seq_len)
        proposed_peak_bolus = min(proposed_peak_bolus, proposed_bolus_seq_len)

        # preprocess the data
        proposed_train, proposed_val, proposed_input_scaler, proposed_output_scaler = (
            prep.data_preprocessing_proposed_for_99(
                data_type=train_type,
                folder_path_4days=folder_4days_path,
                p_num=patient_num,
                seq_len_cgm=proposed_cgm_seq_len,
                seq_len_carb=proposed_carb_seq_len,
                seq_len_basal=proposed_basal_seq_len,
                seq_len_bolus=proposed_bolus_seq_len,
                ph=ph,
                interval=interval,
            )
        )

        # create the model
        proposed_model = abla.ProposedModel_dual_input(
            input_size=proposed_input_size,
            hidden_size=proposed_hidden_size,
            output_size=proposed_output_size,
        )
        proposed_optimizer = Adam(proposed_model.parameters(), lr=proposed_other_lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train the model
        proposed_model, _, _ = abla.proposed_model_train_dual(
            model=proposed_model,
            train_dataloader=proposed_train,
            val_dataloader=proposed_val,
            optimizer=proposed_optimizer,
            epochs=proposed_other_epochs,
            alpha=alpha,
            beta=beta,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            interval=interval,
            device=device,
        )
        # # save the model
        # torch.save(
        #     proposed_model.state_dict(),
        #     os.path.join(
        #         "models", f"abla_dual_proposed_model_patient_{patient_num}.pth"
        #     ),
        # )

        # evaluate the model on the test set
        # preprocess test data
        proposed_test = prep.test_data_preprocessing_proposed_for_99(
            data_type=test_type,
            folder_path_1=folder_test_path,
            p_num=patient_num,
            seq_len_cgm=proposed_cgm_seq_len,
            seq_len_carb=proposed_carb_seq_len,
            seq_len_basal=proposed_basal_seq_len,
            seq_len_bolus=proposed_bolus_seq_len,
            ph=ph,
            interval=interval,
            input_scalers=proposed_input_scaler,
            output_scaler=proposed_output_scaler,
        )
        # evaluate the model
        (
            proposed_preds,
            proposed_truths,
            proposed_pred_alarms,
            proposed_truth_alarms,
            attention_weights,
            proposed_eval_fig,
            _,
            proposed_eval_threshold_fig,
            _,
        ) = abla.proposed_model_eval_dual(
            model=proposed_model,
            test_dataloader=proposed_test,
            tp_insulin_basal=proposed_peak_basal,
            tp_insulin_bolus=proposed_peak_bolus,
            tp_meal=proposed_peak_carb,
            seq_len_carb_intake=proposed_carb_seq_len,
            seq_len_basal_insulin=proposed_basal_seq_len,
            seq_len_bolus_insulin=proposed_bolus_seq_len,
            interval=interval,
            scaler=proposed_output_scaler["G"],
            device=device,
            ticks_per_day=ticks,
            time_steps=interval,
        )

        # # save the evaluation figures
        # proposed_eval_fig_name = f"abla_dual_predictions_patient_{patient_num}.png"
        # proposed_eval_fig.savefig(os.path.join("figures", proposed_eval_fig_name))
        # proposed_eval_threshold_fig_name = (
        #     f"abla_dual_predictions_threshold_patient_{patient_num}.png"
        # )
        # proposed_eval_threshold_fig.savefig(
        #     os.path.join("figures", proposed_eval_threshold_fig_name)
        # )
        plt.close(proposed_eval_fig)
        plt.close(proposed_eval_threshold_fig)

        # evaluate the model using evaluation metrics
        # RMSE for all predictions
        proposed_rmse_all = eva.calculate_rmse(proposed_preds, proposed_truths)
        # RMSE for values above the threshold
        proposed_hyper_rmse, proposed_hypo_rmse, proposed_thres_rmse = (
            eva.calculate_threshold_rmse(proposed_preds, proposed_truths)
        )
        # Calculate upward delay and downward delay
        proposed_ud, proposed_dd, _, _ = eva.calculate_ud_dd(
            prediction=proposed_preds,
            truth=proposed_truths,
            ph=ph,
        )
        # Calculate FIT
        proposed_fit = eva.calculate_fit(proposed_preds, proposed_truths)
        # calculate f1 score
        proposed_f1_score = eva.evaluate_alarm_multiclass(
            alarm=proposed_pred_alarms,
            truth=proposed_truth_alarms,
            dws=dws,
            dwe=dwe,
        )
        # model interpretability
        proposed_interp_fig, _, proposed_shap_perp = (
            eva.calculate_shap_proposed_no_dual_input(
                model=proposed_model,
                dataloader=proposed_test,
                tp_insulin_basal=proposed_peak_basal,
                tp_insulin_bolus=proposed_peak_bolus,
                tp_meal=proposed_peak_carb,
                seq_len_basal_insulin=proposed_basal_seq_len,
                seq_len_bolus_insulin=proposed_bolus_seq_len,
                seq_len_carb_intake=proposed_carb_seq_len,
                interval=interval,
                max_samples=1,
            )
        )
        # proposed_interp_fig_name = (
        #     f"abla_dual_interpretability_patient_{patient_num}.png"
        # )
        # proposed_interp_fig.savefig(os.path.join("figures", proposed_interp_fig_name))
        plt.close(proposed_interp_fig)

        # error between predictions and truths
        proposed_error = abs(proposed_preds - proposed_truths)

    # delete the model to free up memory
    del proposed_model, proposed_optimizer

    # return the results
    return (
        proposed_preds,
        proposed_truths,
        attention_weights,
        proposed_rmse_all,
        proposed_hyper_rmse,
        proposed_hypo_rmse,
        proposed_thres_rmse,
        proposed_ud,
        proposed_dd,
        proposed_fit,
        proposed_f1_score,
        proposed_error,
        proposed_shap_perp,
    )


def main():
    """Main function to run this project."""
    # define metrics lists
    baseline_rmse_all_list = []
    proposed_rmse_all_list = []
    baseline_hyper_rmse_list = []
    proposed_hyper_rmse_list = []
    baseline_hypo_rmse_list = []
    proposed_hypo_rmse_list = []
    baseline_thres_rmse_list = []
    proposed_thres_rmse_list = []
    baseline_ud_list = []
    proposed_ud_list = []
    baseline_dd_list = []
    proposed_dd_list = []
    baseline_fit_list = []
    proposed_fit_list = []
    baseline_f1_score_list = []
    proposed_f1_score_list = []
    baseline_error_list = []
    proposed_error_list = []
    baseline_group_shap = {
        "CGM": [],
        "Carb Intake": [],
        "Insulin Basal": [],
        "Insulin Bolus": [],
    }
    proposed_group_shap = {
        "CGM": [],
        "Carb Intake": [],
        "Insulin Basal": [],
        "Insulin Bolus": [],
    }

    # define ablation study lists
    ablation_loss_rmse_all_list = []
    ablation_loss_hyper_rmse_list = []
    ablation_loss_hypo_rmse_list = []
    ablation_loss_thres_rmse_list = []
    ablation_loss_ud_list = []
    ablation_loss_dd_list = []
    ablation_loss_fit_list = []
    ablation_loss_f1_score_list = []
    ablation_loss_error_list = []
    ablation_loss_group_shap = {
        "CGM": [],
        "Carb Intake": [],
        "Insulin Basal": [],
        "Insulin Bolus": [],
    }

    ablation_phy_rmse_all_list = []
    ablation_phy_hyper_rmse_list = []
    ablation_phy_hypo_rmse_list = []
    ablation_phy_thres_rmse_list = []
    ablation_phy_ud_list = []
    ablation_phy_dd_list = []
    ablation_phy_fit_list = []
    ablation_phy_f1_score_list = []
    ablation_phy_error_list = []
    ablation_phy_group_shap = {
        "CGM": [],
        "Carb Intake": [],
        "Insulin Basal": [],
        "Insulin Bolus": [],
    }

    ablation_dual_rmse_all_list = []
    ablation_dual_hyper_rmse_list = []
    ablation_dual_hypo_rmse_list = []
    ablation_dual_thres_rmse_list = []
    ablation_dual_ud_list = []
    ablation_dual_dd_list = []
    ablation_dual_fit_list = []
    ablation_dual_f1_score_list = []
    ablation_dual_error_list = []
    ablation_dual_group_shap = {
        "CGM": [],
        "Carb Intake": [],
        "Insulin Basal": [],
        "Insulin Bolus": [],
    }

    # define list for patient 98
    baseline_preds_98 = []
    proposed_preds_98 = []
    truths = []
    ablation_loss_preds_98 = []
    ablation_phy_preds_98 = []
    ablation_dual_preds_98 = []

    baseline_rmse_all_98_list = []
    baseline_hyper_rmse_98_list = []
    baseline_hypo_rmse_98_list = []
    baseline_thres_rmse_98_list = []
    baseline_ud_98_list = []
    baseline_dd_98_list = []
    baseline_fit_98_list = []
    baseline_f1_score_98_list = []

    proposed_rmse_all_98_list = []
    proposed_hyper_rmse_98_list = []
    proposed_hypo_rmse_98_list = []
    proposed_thres_rmse_98_list = []
    proposed_ud_98_list = []
    proposed_dd_98_list = []
    proposed_fit_98_list = []
    proposed_f1_score_98_list = []

    ablation_loss_rmse_all_98_list = []
    ablation_loss_hyper_rmse_98_list = []
    ablation_loss_hypo_rmse_98_list = []
    ablation_loss_thres_rmse_98_list = []
    ablation_loss_ud_98_list = []
    ablation_loss_dd_98_list = []
    ablation_loss_fit_98_list = []
    ablation_loss_f1_score_98_list = []

    ablation_phy_rmse_all_98_list = []
    ablation_phy_hyper_rmse_98_list = []
    ablation_phy_hypo_rmse_98_list = []
    ablation_phy_thres_rmse_98_list = []
    ablation_phy_ud_98_list = []
    ablation_phy_dd_98_list = []
    ablation_phy_fit_98_list = []
    ablation_phy_f1_score_98_list = []

    ablation_dual_rmse_all_98_list = []
    ablation_dual_hyper_rmse_98_list = []
    ablation_dual_hypo_rmse_98_list = []
    ablation_dual_thres_rmse_98_list = []
    ablation_dual_ud_98_list = []
    ablation_dual_dd_98_list = []
    ablation_dual_fit_98_list = []
    ablation_dual_f1_score_98_list = []

    # Exploratory Data Analysis
    train_type = "train"
    data_folder_vis = "train_30days"
    test_type = "test"
    data_folder_insulin = "ts-dtI"
    data_folder_meal = "ts-dtM"
    patient_example = 98
    prep.data_eda(
        data_type_1=train_type,
        data_folder_1=data_folder_vis,
        data_type_2=test_type,
        data_folder_2=data_folder_insulin,
        data_folder_3=data_folder_meal,
        p_num=patient_example,
    )

    # visualise physio modeling layer contribution
    phy_peak_time_basal_98 = 400
    phy_peak_time_bolus_98 = 290
    phy_peak_time_meal_98 = 210
    phy_peak_time_interval_98 = 5

    mdl.visualise_phy_layer(
        data_type=train_type,
        data_folder=data_folder_vis,
        p_num=patient_example,
        figure_folder="figures",
        peak_time_basal=phy_peak_time_basal_98,
        peak_time_bolus=phy_peak_time_bolus_98,
        peak_time_carb=phy_peak_time_meal_98,
        interval=phy_peak_time_interval_98,
    )

    # make predictions
    for pat_num in range(1, 101):
        print(f"Patient {pat_num}")
        # baseline model
        (
            baseline_preds,
            baseline_truths,
            baseline_rmse_all,
            baseline_hyper_rmse,
            baseline_hypo_rmse,
            baseline_thres_rmse,
            baseline_ud,
            baseline_dd,
            baseline_fit,
            baseline_f1_score,
            baseline_error,
            baseline_shap_perp,
        ) = baseline_modeling(pat_num)

        if pat_num == 98:
            # save the predictions for patient 98
            baseline_preds_98.append(baseline_preds)
            truths.append(baseline_truths)

            baseline_rmse_all_98_list.append(baseline_rmse_all)
            baseline_hyper_rmse_98_list.append(baseline_hyper_rmse)
            baseline_hypo_rmse_98_list.append(baseline_hypo_rmse)
            baseline_thres_rmse_98_list.append(baseline_thres_rmse)
            baseline_ud_98_list.append(baseline_ud)
            baseline_dd_98_list.append(baseline_dd)
            baseline_fit_98_list.append(baseline_fit)
            baseline_f1_score_98_list.append(baseline_f1_score)

        # append the results to the lists
        baseline_rmse_all_list.append(baseline_rmse_all)
        baseline_hyper_rmse_list.append(baseline_hyper_rmse)
        baseline_hypo_rmse_list.append(baseline_hypo_rmse)
        baseline_thres_rmse_list.append(baseline_thres_rmse)
        baseline_ud_list.append(baseline_ud)
        baseline_dd_list.append(baseline_dd)
        baseline_fit_list.append(baseline_fit)
        baseline_f1_score_list.append(baseline_f1_score)
        baseline_error_list.append(baseline_error)

        # append the group SHAP values
        for key, arr in baseline_shap_perp.items():
            baseline_group_shap.setdefault(key, []).extend(
                np.asarray(arr).ravel().tolist()
            )

        # proposed model
        (
            proposed_preds,
            proposed_truths,
            attention_weights,
            proposed_rmse_all,
            proposed_hyper_rmse,
            proposed_hypo_rmse,
            proposed_thres_rmse,
            proposed_ud,
            proposed_dd,
            proposed_fit,
            proposed_f1_score,
            proposed_error,
            proposed_shap_perp,
        ) = proposed_modeling(pat_num)

        if pat_num == 98:
            # save the predictions for patient 98
            proposed_preds_98.append(proposed_preds)

            proposed_rmse_all_98_list.append(proposed_rmse_all)
            proposed_hyper_rmse_98_list.append(proposed_hyper_rmse)
            proposed_hypo_rmse_98_list.append(proposed_hypo_rmse)
            proposed_thres_rmse_98_list.append(proposed_thres_rmse)
            proposed_ud_98_list.append(proposed_ud)
            proposed_dd_98_list.append(proposed_dd)
            proposed_fit_98_list.append(proposed_fit)
            proposed_f1_score_98_list.append(proposed_f1_score)

        # append the results to the lists
        proposed_rmse_all_list.append(proposed_rmse_all)
        proposed_hyper_rmse_list.append(proposed_hyper_rmse)
        proposed_hypo_rmse_list.append(proposed_hypo_rmse)
        proposed_thres_rmse_list.append(proposed_thres_rmse)
        proposed_ud_list.append(proposed_ud)
        proposed_dd_list.append(proposed_dd)
        proposed_fit_list.append(proposed_fit)
        proposed_f1_score_list.append(proposed_f1_score)
        proposed_error_list.append(proposed_error)

        # append the group SHAP values
        for key, arr in proposed_shap_perp.items():
            proposed_group_shap.setdefault(key, []).extend(
                np.asarray(arr).ravel().tolist()
            )
        # close the figures to free up memory
        plt.close("all")

    # describe the results
    # baseline model results
    baseline_rmse_all_describe = eva.describe_results(baseline_rmse_all_list)
    baseline_hyper_rmse_describe = eva.describe_results(baseline_hyper_rmse_list)
    baseline_hypo_rmse_describe = eva.describe_results(baseline_hypo_rmse_list)
    baseline_thres_rmse_describe = eva.describe_results(baseline_thres_rmse_list)
    baseline_ud_describe = eva.describe_results(baseline_ud_list)
    baseline_dd_describe = eva.describe_results(baseline_dd_list)
    baseline_fit_describe = eva.describe_results(baseline_fit_list)
    baseline_f1_score_describe = eva.describe_results(baseline_f1_score_list)

    baseline_rmse_all_98_describe = eva.describe_results(baseline_rmse_all_98_list)
    baseline_hyper_rmse_98_describe = eva.describe_results(baseline_hyper_rmse_98_list)
    baseline_hypo_rmse_98_describe = eva.describe_results(baseline_hypo_rmse_98_list)
    baseline_thres_rmse_98_describe = eva.describe_results(baseline_thres_rmse_98_list)
    baseline_ud_98_describe = eva.describe_results(baseline_ud_98_list)
    baseline_dd_98_describe = eva.describe_results(baseline_dd_98_list)
    baseline_fit_98_describe = eva.describe_results(baseline_fit_98_list)
    baseline_f1_score_98_describe = eva.describe_results(baseline_f1_score_98_list)

    # proposed model results
    proposed_rmse_all_describe = eva.describe_results(proposed_rmse_all_list)
    proposed_hyper_rmse_describe = eva.describe_results(proposed_hyper_rmse_list)
    proposed_hypo_rmse_describe = eva.describe_results(proposed_hypo_rmse_list)
    proposed_thres_rmse_describe = eva.describe_results(proposed_thres_rmse_list)
    proposed_ud_describe = eva.describe_results(proposed_ud_list)
    proposed_dd_describe = eva.describe_results(proposed_dd_list)
    proposed_fit_describe = eva.describe_results(proposed_fit_list)
    proposed_f1_score_describe = eva.describe_results(proposed_f1_score_list)

    proposed_rmse_all_98_describe = eva.describe_results(proposed_rmse_all_98_list)
    proposed_hyper_rmse_98_describe = eva.describe_results(proposed_hyper_rmse_98_list)
    proposed_hypo_rmse_98_describe = eva.describe_results(proposed_hypo_rmse_98_list)
    proposed_thres_rmse_98_describe = eva.describe_results(proposed_thres_rmse_98_list)
    proposed_ud_98_describe = eva.describe_results(proposed_ud_98_list)
    proposed_dd_98_describe = eva.describe_results(proposed_dd_98_list)
    proposed_fit_98_describe = eva.describe_results(proposed_fit_98_list)
    proposed_f1_score_98_describe = eva.describe_results(proposed_f1_score_98_list)

    # save the results to a text file
    with open("results/prediction_results.txt", "w", encoding="utf-8") as f:
        f.write("Baseline Model Results:\n")
        f.write(f"RMSE All: {baseline_rmse_all_describe}\n")
        f.write(f"Hyperglycemia RMSE: {baseline_hyper_rmse_describe}\n")
        f.write(f"Hypoglycemia RMSE: {baseline_hypo_rmse_describe}\n")
        f.write(f"Threshold RMSE: {baseline_thres_rmse_describe}\n")
        f.write(f"Upward Delay: {baseline_ud_describe}\n")
        f.write(f"Downward Delay: {baseline_dd_describe}\n")
        f.write(f"FIT: {baseline_fit_describe}\n")
        f.write(f"F1 Score: {baseline_f1_score_describe}\n")
        f.write("\nProposed Model Results:\n")
        f.write(f"RMSE All: {proposed_rmse_all_describe}\n")
        f.write(f"Hyperglycemia RMSE: {proposed_hyper_rmse_describe}\n")
        f.write(f"Hypoglycemia RMSE: {proposed_hypo_rmse_describe}\n")
        f.write(f"Threshold RMSE: {proposed_thres_rmse_describe}\n")
        f.write(f"Upward Delay: {proposed_ud_describe}\n")
        f.write(f"Downward Delay: {proposed_dd_describe}\n")
        f.write(f"FIT: {proposed_fit_describe}\n")
        f.write(f"F1 Score: {proposed_f1_score_describe}\n\n")

    # plot the error distributions
    error_fig, _ = vis.plot_errors(
        baseline_error_list,
        proposed_error_list,
        label1="Baseline Model",
        label2="Proposed Model",
    )
    error_fig_name = "Absolute Error plot.png"
    error_fig.savefig(os.path.join("figures", error_fig_name))
    plt.close(error_fig)

    # plot the group SHAP values
    # baseline model
    # violin plot for group SHAP values
    baseline_group_shap_fig, _ = vis.plot_shap_violin(baseline_group_shap)
    baseline_group_shap_fig_name = "Baseline Group SHAP values.png"
    baseline_group_shap_fig.savefig(
        os.path.join("figures", baseline_group_shap_fig_name)
    )
    plt.close(baseline_group_shap_fig)

    # box plot for group SHAP values
    baseline_group_shap_box_fig, _ = vis.plot_shap_boxplot(baseline_group_shap)
    baseline_group_shap_box_fig_name = "Baseline Group SHAP values Boxplot.png"
    baseline_group_shap_box_fig.savefig(
        os.path.join("figures", baseline_group_shap_box_fig_name)
    )
    plt.close(baseline_group_shap_box_fig)

    # proposed model
    proposed_group_shap_fig, _ = vis.plot_shap_violin(proposed_group_shap)
    proposed_group_shap_fig_name = "Proposed Group SHAP values.png"
    proposed_group_shap_fig.savefig(
        os.path.join("figures", proposed_group_shap_fig_name)
    )
    plt.close(proposed_group_shap_fig)

    # box plot for group SHAP values
    proposed_group_shap_box_fig, _ = vis.plot_shap_boxplot(proposed_group_shap)
    proposed_group_shap_box_fig_name = "Proposed Group SHAP values Boxplot.png"
    proposed_group_shap_box_fig.savefig(
        os.path.join("figures", proposed_group_shap_box_fig_name)
    )
    plt.close(proposed_group_shap_box_fig)

    # ---- Ablation Study ----
    for pat_num in range(1, 101):
        # ablation study for loss function
        print(f"Ablation Study Loss Function Patient {pat_num}")
        (
            ablation_loss_preds,
            ablation_loss_truths,
            attention_weights,
            ablation_loss_rmse_all,
            ablation_loss_hyper_rmse,
            ablation_loss_hypo_rmse,
            ablation_loss_thres_rmse,
            ablation_loss_ud,
            ablation_loss_dd,
            ablation_loss_fit,
            ablation_loss_f1_score,
            ablation_loss_error,
            ablation_loss_shap_perp,
        ) = ablation_study_loss_function(pat_num)

        if pat_num == 98:
            # save the predictions for patient 98
            ablation_loss_preds_98.append(ablation_loss_preds)

            ablation_loss_rmse_all_98_list.append(ablation_loss_rmse_all)
            ablation_loss_hyper_rmse_98_list.append(ablation_loss_hyper_rmse)
            ablation_loss_hypo_rmse_98_list.append(ablation_loss_hypo_rmse)
            ablation_loss_thres_rmse_98_list.append(ablation_loss_thres_rmse)
            ablation_loss_ud_98_list.append(ablation_loss_ud)
            ablation_loss_dd_98_list.append(ablation_loss_dd)
            ablation_loss_fit_98_list.append(ablation_loss_fit)
            ablation_loss_f1_score_98_list.append(ablation_loss_f1_score)

        # append the results to the lists
        ablation_loss_rmse_all_list.append(ablation_loss_rmse_all)
        ablation_loss_hyper_rmse_list.append(ablation_loss_hyper_rmse)
        ablation_loss_hypo_rmse_list.append(ablation_loss_hypo_rmse)
        ablation_loss_thres_rmse_list.append(ablation_loss_thres_rmse)
        ablation_loss_ud_list.append(ablation_loss_ud)
        ablation_loss_dd_list.append(ablation_loss_dd)
        ablation_loss_fit_list.append(ablation_loss_fit)
        ablation_loss_f1_score_list.append(ablation_loss_f1_score)
        ablation_loss_error_list.append(ablation_loss_error)

        # append the group SHAP values
        for key, arr in ablation_loss_shap_perp.items():
            ablation_loss_group_shap.setdefault(key, []).extend(
                np.asarray(arr).ravel().tolist()
            )

        # ablation study for physical layer
        print(f"Ablation Study Physical Layer Patient {pat_num}")
        (
            ablation_phy_preds,
            ablation_phy_truths,
            attention_weights,
            ablation_phy_rmse_all,
            ablation_phy_hyper_rmse,
            ablation_phy_hypo_rmse,
            ablation_phy_thres_rmse,
            ablation_phy_ud,
            ablation_phy_dd,
            ablation_phy_fit,
            ablation_phy_f1_score,
            ablation_phy_error,
            ablation_phy_shap_perp,
        ) = ablation_study_phy_layer(pat_num)

        if pat_num == 98:
            # save the predictions for patient 98
            ablation_phy_preds_98.append(ablation_phy_preds)

            ablation_phy_rmse_all_98_list.append(ablation_phy_rmse_all)
            ablation_phy_hyper_rmse_98_list.append(ablation_phy_hyper_rmse)
            ablation_phy_hypo_rmse_98_list.append(ablation_phy_hypo_rmse)
            ablation_phy_thres_rmse_98_list.append(ablation_phy_thres_rmse)
            ablation_phy_ud_98_list.append(ablation_phy_ud)
            ablation_phy_dd_98_list.append(ablation_phy_dd)
            ablation_phy_fit_98_list.append(ablation_phy_fit)
            ablation_phy_f1_score_98_list.append(ablation_phy_f1_score)

        # append the results to the lists
        ablation_phy_rmse_all_list.append(ablation_phy_rmse_all)
        ablation_phy_hyper_rmse_list.append(ablation_phy_hyper_rmse)
        ablation_phy_hypo_rmse_list.append(ablation_phy_hypo_rmse)
        ablation_phy_thres_rmse_list.append(ablation_phy_thres_rmse)
        ablation_phy_ud_list.append(ablation_phy_ud)
        ablation_phy_dd_list.append(ablation_phy_dd)
        ablation_phy_fit_list.append(ablation_phy_fit)
        ablation_phy_f1_score_list.append(ablation_phy_f1_score)
        ablation_phy_error_list.append(ablation_phy_error)
        # append the group SHAP values
        for key, arr in ablation_phy_shap_perp.items():
            ablation_phy_group_shap.setdefault(key, []).extend(
                np.asarray(arr).ravel().tolist()
            )
        # close the figures to free up memory
        plt.close("all")

        # ablation study for dual input
        print(f"Ablation Study Dual Input Patient {pat_num}")
        (
            ablation_dual_preds,
            ablation_dual_truths,
            attention_weights,
            ablation_dual_rmse_all,
            ablation_dual_hyper_rmse,
            ablation_dual_hypo_rmse,
            ablation_dual_thres_rmse,
            ablation_dual_ud,
            ablation_dual_dd,
            ablation_dual_fit,
            ablation_dual_f1_score,
            ablation_dual_error,
            ablation_dual_shap_perp,
        ) = ablation_study_dual_input(pat_num)
        if pat_num == 98:
            # save the predictions for patient 98
            ablation_dual_preds_98.append(ablation_dual_preds)

            ablation_dual_rmse_all_98_list.append(ablation_dual_rmse_all)
            ablation_dual_hyper_rmse_98_list.append(ablation_dual_hyper_rmse)
            ablation_dual_hypo_rmse_98_list.append(ablation_dual_hypo_rmse)
            ablation_dual_thres_rmse_98_list.append(ablation_dual_thres_rmse)
            ablation_dual_ud_98_list.append(ablation_dual_ud)
            ablation_dual_dd_98_list.append(ablation_dual_dd)
            ablation_dual_fit_98_list.append(ablation_dual_fit)
            ablation_dual_f1_score_98_list.append(ablation_dual_f1_score)

        # append the results to the lists
        ablation_dual_rmse_all_list.append(ablation_dual_rmse_all)
        ablation_dual_hyper_rmse_list.append(ablation_dual_hyper_rmse)
        ablation_dual_hypo_rmse_list.append(ablation_dual_hypo_rmse)
        ablation_dual_thres_rmse_list.append(ablation_dual_thres_rmse)
        ablation_dual_ud_list.append(ablation_dual_ud)
        ablation_dual_dd_list.append(ablation_dual_dd)
        ablation_dual_fit_list.append(ablation_dual_fit)
        ablation_dual_f1_score_list.append(ablation_dual_f1_score)
        ablation_dual_error_list.append(ablation_dual_error)
        # append the group SHAP values
        for key, arr in ablation_dual_shap_perp.items():
            ablation_dual_group_shap.setdefault(key, []).extend(
                np.asarray(arr).ravel().tolist()
            )
        # close the figures to free up memory
        plt.close("all")

    # describe the results of ablation study
    # ablation study for loss function results
    ablation_loss_rmse_all_describe = eva.describe_results(ablation_loss_rmse_all_list)
    ablation_loss_hyper_rmse_describe = eva.describe_results(
        ablation_loss_hyper_rmse_list
    )
    ablation_loss_hypo_rmse_describe = eva.describe_results(
        ablation_loss_hypo_rmse_list
    )
    ablation_loss_thres_rmse_describe = eva.describe_results(
        ablation_loss_thres_rmse_list
    )
    ablation_loss_ud_describe = eva.describe_results(ablation_loss_ud_list)
    ablation_loss_dd_describe = eva.describe_results(ablation_loss_dd_list)
    ablation_loss_fit_describe = eva.describe_results(ablation_loss_fit_list)
    ablation_loss_f1_score_describe = eva.describe_results(ablation_loss_f1_score_list)

    # patient 98
    ablation_loss_rmse_all_98_describe = eva.describe_results(
        ablation_loss_rmse_all_98_list
    )
    ablation_loss_hyper_rmse_98_describe = eva.describe_results(
        ablation_loss_hyper_rmse_98_list
    )
    ablation_loss_hypo_rmse_98_describe = eva.describe_results(
        ablation_loss_hypo_rmse_98_list
    )
    ablation_loss_thres_rmse_98_describe = eva.describe_results(
        ablation_loss_thres_rmse_98_list
    )
    ablation_loss_ud_98_describe = eva.describe_results(ablation_loss_ud_98_list)
    ablation_loss_dd_98_describe = eva.describe_results(ablation_loss_dd_98_list)
    ablation_loss_fit_98_describe = eva.describe_results(ablation_loss_fit_98_list)
    ablation_loss_f1_score_98_describe = eva.describe_results(
        ablation_loss_f1_score_98_list
    )

    # ablation study for physical layer results
    ablation_phy_rmse_all_describe = eva.describe_results(ablation_phy_rmse_all_list)
    ablation_phy_hyper_rmse_describe = eva.describe_results(
        ablation_phy_hyper_rmse_list
    )
    ablation_phy_hypo_rmse_describe = eva.describe_results(ablation_phy_hypo_rmse_list)
    ablation_phy_thres_rmse_describe = eva.describe_results(
        ablation_phy_thres_rmse_list
    )
    ablation_phy_ud_describe = eva.describe_results(ablation_phy_ud_list)
    ablation_phy_dd_describe = eva.describe_results(ablation_phy_dd_list)
    ablation_phy_fit_describe = eva.describe_results(ablation_phy_fit_list)
    ablation_phy_f1_score_describe = eva.describe_results(ablation_phy_f1_score_list)

    # patient 98
    ablation_phy_rmse_all_98_describe = eva.describe_results(
        ablation_phy_rmse_all_98_list
    )
    ablation_phy_hyper_rmse_98_describe = eva.describe_results(
        ablation_phy_hyper_rmse_98_list
    )
    ablation_phy_hypo_rmse_98_describe = eva.describe_results(
        ablation_phy_hypo_rmse_98_list
    )
    ablation_phy_thres_rmse_98_describe = eva.describe_results(
        ablation_phy_thres_rmse_98_list
    )
    ablation_phy_ud_98_describe = eva.describe_results(ablation_phy_ud_98_list)
    ablation_phy_dd_98_describe = eva.describe_results(ablation_phy_dd_98_list)
    ablation_phy_fit_98_describe = eva.describe_results(ablation_phy_fit_98_list)
    ablation_phy_f1_score_98_describe = eva.describe_results(
        ablation_phy_f1_score_98_list
    )

    # ablation study for dual input results
    ablation_dual_rmse_all_describe = eva.describe_results(ablation_dual_rmse_all_list)
    ablation_dual_hyper_rmse_describe = eva.describe_results(
        ablation_dual_hyper_rmse_list
    )
    ablation_dual_hypo_rmse_describe = eva.describe_results(
        ablation_dual_hypo_rmse_list
    )
    ablation_dual_thres_rmse_describe = eva.describe_results(
        ablation_dual_thres_rmse_list
    )
    ablation_dual_ud_describe = eva.describe_results(ablation_dual_ud_list)
    ablation_dual_dd_describe = eva.describe_results(ablation_dual_dd_list)
    ablation_dual_fit_describe = eva.describe_results(ablation_dual_fit_list)
    ablation_dual_f1_score_describe = eva.describe_results(ablation_dual_f1_score_list)

    # patient 98
    ablation_dual_rmse_all_98_describe = eva.describe_results(
        ablation_dual_rmse_all_98_list
    )
    ablation_dual_hyper_rmse_98_describe = eva.describe_results(
        ablation_dual_hyper_rmse_98_list
    )
    ablation_dual_hypo_rmse_98_describe = eva.describe_results(
        ablation_dual_hypo_rmse_98_list
    )
    ablation_dual_thres_rmse_98_describe = eva.describe_results(
        ablation_dual_thres_rmse_98_list
    )
    ablation_dual_ud_98_describe = eva.describe_results(ablation_dual_ud_98_list)
    ablation_dual_dd_98_describe = eva.describe_results(ablation_dual_dd_98_list)
    ablation_dual_fit_98_describe = eva.describe_results(ablation_dual_fit_98_list)
    ablation_dual_f1_score_98_describe = eva.describe_results(
        ablation_dual_f1_score_98_list
    )

    # save the results to a text file
    with open("results/ablation_study_results.txt", "w", encoding="utf-8") as f:
        f.write("Ablation Study Loss Function Results:\n")
        f.write(f"RMSE All: {ablation_loss_rmse_all_describe}\n")
        f.write(f"Hyperglycemia RMSE: {ablation_loss_hyper_rmse_describe}\n")
        f.write(f"Hypoglycemia RMSE: {ablation_loss_hypo_rmse_describe}\n")
        f.write(f"Threshold RMSE: {ablation_loss_thres_rmse_describe}\n")
        f.write(f"Upward Delay: {ablation_loss_ud_describe}\n")
        f.write(f"Downward Delay: {ablation_loss_dd_describe}\n")
        f.write(f"FIT: {ablation_loss_fit_describe}\n")
        f.write(f"F1 Score: {ablation_loss_f1_score_describe}\n\n")

        f.write("Ablation Study Physical Layer Results:\n")
        f.write(f"RMSE All: {ablation_phy_rmse_all_describe}\n")
        f.write(f"Hyperglycemia RMSE: {ablation_phy_hyper_rmse_describe}\n")
        f.write(f"Hypoglycemia RMSE: {ablation_phy_hypo_rmse_describe}\n")
        f.write(f"Threshold RMSE: {ablation_phy_thres_rmse_describe}\n")
        f.write(f"Upward Delay: {ablation_phy_ud_describe}\n")
        f.write(f"Downward Delay: {ablation_phy_dd_describe}\n")
        f.write(f"FIT: {ablation_phy_fit_describe}\n")
        f.write(f"F1 Score: {ablation_phy_f1_score_describe}\n\n")

        f.write("Ablation Study Dual Input Results:\n")
        f.write(f"RMSE All: {ablation_dual_rmse_all_describe}\n")
        f.write(f"Hyperglycemia RMSE: {ablation_dual_hyper_rmse_describe}\n")
        f.write(f"Hypoglycemia RMSE: {ablation_dual_hypo_rmse_describe}\n")
        f.write(f"Threshold RMSE: {ablation_dual_thres_rmse_describe}\n")
        f.write(f"Upward Delay: {ablation_dual_ud_describe}\n")
        f.write(f"Downward Delay: {ablation_dual_dd_describe}\n")
        f.write(f"FIT: {ablation_dual_fit_describe}\n")
        f.write(f"F1 Score: {ablation_dual_f1_score_describe}\n\n")

    # save the results of patient 98 to a text file
    with open("results/patient_98_results.txt", "w", encoding="utf-8") as f:
        f.write("Baseline Model Results:\n")
        f.write(f"RMSE All: {baseline_rmse_all_98_describe}\n")
        f.write(f"Hyperglycemia RMSE: {baseline_hyper_rmse_98_describe}\n")
        f.write(f"Hypoglycemia RMSE: {baseline_hypo_rmse_98_describe}\n")
        f.write(f"Threshold RMSE: {baseline_thres_rmse_98_describe}\n")
        f.write(f"Upward Delay: {baseline_ud_98_describe}\n")
        f.write(f"Downward Delay: {baseline_dd_98_describe}\n")
        f.write(f"FIT: {baseline_fit_98_describe}\n")
        f.write(f"F1 Score: {baseline_f1_score_98_describe}\n")

        f.write("\nProposed Model Results:\n")
        f.write(f"RMSE All: {proposed_rmse_all_98_describe}\n")
        f.write(f"Hyperglycemia RMSE: {proposed_hyper_rmse_98_describe}\n")
        f.write(f"Hypoglycemia RMSE: {proposed_hypo_rmse_98_describe}\n")
        f.write(f"Threshold RMSE: {proposed_thres_rmse_98_describe}\n")
        f.write(f"Upward Delay: {proposed_ud_98_describe}\n")
        f.write(f"Downward Delay: {proposed_dd_98_describe}\n")
        f.write(f"FIT: {proposed_fit_98_describe}\n")
        f.write(f"F1 Score: {proposed_f1_score_98_describe}\n\n")

        f.write("Ablation Study Loss Function Results:\n")
        f.write(f"RMSE All: {ablation_loss_rmse_all_98_describe}\n")
        f.write(f"Hyperglycemia RMSE: {ablation_loss_hyper_rmse_98_describe}\n")
        f.write(f"Hypoglycemia RMSE: {ablation_loss_hypo_rmse_98_describe}\n")
        f.write(f"Threshold RMSE: {ablation_loss_thres_rmse_98_describe}\n")
        f.write(f"Upward Delay: {ablation_loss_ud_98_describe}\n")
        f.write(f"Downward Delay: {ablation_loss_dd_98_describe}\n")
        f.write(f"FIT: {ablation_loss_fit_98_describe}\n")
        f.write(f"F1 Score: {ablation_loss_f1_score_98_describe}\n\n")

        f.write("Ablation Study Physical Layer Results:\n")
        f.write(f"RMSE All: {ablation_phy_rmse_all_98_describe}\n")
        f.write(f"Hyperglycemia RMSE: {ablation_phy_hyper_rmse_98_describe}\n")
        f.write(f"Hypoglycemia RMSE: {ablation_phy_hypo_rmse_98_describe}\n")
        f.write(f"Threshold RMSE: {ablation_phy_thres_rmse_98_describe}\n")
        f.write(f"Upward Delay: {ablation_phy_ud_98_describe}\n")
        f.write(f"Downward Delay: {ablation_phy_dd_98_describe}\n")
        f.write(f"FIT: {ablation_phy_fit_98_describe}\n")
        f.write(f"F1 Score: {ablation_phy_f1_score_98_describe}\n\n")

        f.write("Ablation Study Dual Input Results:\n")
        f.write(f"RMSE All: {ablation_dual_rmse_all_98_describe}\n")
        f.write(f"Hyperglycemia RMSE: {ablation_dual_hyper_rmse_98_describe}\n")
        f.write(f"Hypoglycemia RMSE: {ablation_dual_hypo_rmse_98_describe}\n")
        f.write(f"Threshold RMSE: {ablation_dual_thres_rmse_98_describe}\n")
        f.write(f"Upward Delay: {ablation_dual_ud_98_describe}\n")
        f.write(f"Downward Delay: {ablation_dual_dd_98_describe}\n")
        f.write(f"FIT: {ablation_dual_fit_98_describe}\n")
        f.write(f"F1 Score: {ablation_dual_f1_score_98_describe}\n\n")

    # plot the error distributions for ablation study
    # ablation study for loss function
    ablation_loss_error_fig, _ = vis.plot_errors(
        proposed_error_list,
        ablation_loss_error_list,
        label1="Proposed Model",
        label2="Ablation (MSE Loss)",
    )
    ablation_loss_error_fig_name = "Ablation (Loss Function) Absolute Error plot.png"
    ablation_loss_error_fig.savefig(
        os.path.join("figures", ablation_loss_error_fig_name)
    )
    plt.close(ablation_loss_error_fig)
    # ablation study for physical layer
    ablation_phy_error_fig, _ = vis.plot_errors(
        proposed_error_list,
        ablation_phy_error_list,
        label1="Proposed Model",
        label2="Ablation (Physio Layer)",
    )
    ablation_phy_error_fig_name = "Ablation (Physio Layer) Absolute Error plot.png"
    ablation_phy_error_fig.savefig(os.path.join("figures", ablation_phy_error_fig_name))
    plt.close(ablation_phy_error_fig)
    # ablation study for dual input
    ablation_dual_error_fig, _ = vis.plot_errors(
        proposed_error_list,
        ablation_dual_error_list,
        label1="Proposed Model",
        label2="Ablation (Dual Input)",
    )
    ablation_dual_error_fig_name = "Ablation (Dual Input) Absolute Error plot.png"
    ablation_dual_error_fig.savefig(
        os.path.join("figures", ablation_dual_error_fig_name)
    )
    plt.close(ablation_dual_error_fig)

    # plot the group SHAP values for ablation study
    # ablation study for loss function
    ablation_loss_group_shap_fig, _ = vis.plot_shap_violin(ablation_loss_group_shap)
    ablation_loss_group_shap_fig_name = "Ablation (Loss Function) Group SHAP values.png"
    ablation_loss_group_shap_fig.savefig(
        os.path.join("figures", ablation_loss_group_shap_fig_name)
    )
    plt.close(ablation_loss_group_shap_fig)
    # box plot for group SHAP values
    ablation_loss_group_shap_box_fig, _ = vis.plot_shap_boxplot(
        ablation_loss_group_shap
    )
    ablation_loss_group_shap_box_fig_name = (
        "Ablation (Loss Function) Group SHAP values Boxplot.png"
    )
    ablation_loss_group_shap_box_fig.savefig(
        os.path.join("figures", ablation_loss_group_shap_box_fig_name)
    )
    plt.close(ablation_loss_group_shap_box_fig)

    # ablation study for physical layer
    ablation_phy_group_shap_fig, _ = vis.plot_shap_violin(ablation_phy_group_shap)
    ablation_phy_group_shap_fig_name = "Ablation (Physio Layer) Group SHAP values.png"
    ablation_phy_group_shap_fig.savefig(
        os.path.join("figures", ablation_phy_group_shap_fig_name)
    )
    plt.close(ablation_phy_group_shap_fig)
    # box plot for group SHAP values
    ablation_phy_group_shap_box_fig, _ = vis.plot_shap_boxplot(ablation_phy_group_shap)
    ablation_phy_group_shap_box_fig_name = (
        "Ablation (Physio Layer) Group SHAP values Boxplot.png"
    )
    ablation_phy_group_shap_box_fig.savefig(
        os.path.join("figures", ablation_phy_group_shap_box_fig_name)
    )
    plt.close(ablation_phy_group_shap_box_fig)

    # ablation study for dual input
    ablation_dual_group_shap_fig, _ = vis.plot_shap_violin(ablation_dual_group_shap)
    ablation_dual_group_shap_fig_name = "Ablation (Dual Input) Group SHAP values.png"
    ablation_dual_group_shap_fig.savefig(
        os.path.join("figures", ablation_dual_group_shap_fig_name)
    )
    plt.close(ablation_dual_group_shap_fig)
    # box plot for group SHAP values
    ablation_dual_group_shap_box_fig, _ = vis.plot_shap_boxplot(
        ablation_dual_group_shap
    )
    ablation_dual_group_shap_box_fig_name = (
        "Ablation (Dual Input) Group SHAP values Boxplot.png"
    )
    ablation_dual_group_shap_box_fig.savefig(
        os.path.join("figures", ablation_dual_group_shap_box_fig_name)
    )
    plt.close(ablation_dual_group_shap_box_fig)

    # plot the predictions for patient 98
    baseline_preds_98 = np.concatenate(baseline_preds_98, axis=0)
    proposed_preds_98 = np.concatenate(proposed_preds_98, axis=0)
    truths = np.concatenate(truths, axis=0)
    ablation_loss_preds_98 = np.concatenate(ablation_loss_preds_98, axis=0)
    ablation_phy_preds_98 = np.concatenate(ablation_phy_preds_98, axis=0)
    ablation_dual_preds_98 = np.concatenate(ablation_dual_preds_98, axis=0)

    # visualise the predictions for patient 98
    # proposed model and baseline model
    patient_98_com_fig, _ = vis.visualise_preds_comparison(
        pred1=baseline_preds_98,
        pred2=proposed_preds_98,
        truth=truths,
        label1="Baseline Model",
        label2="Proposed Model",
        ticks_per_day=2,
    )
    patient_98_com_fig_name = "Patient 98 Predictions baseline vs proposed.png"
    patient_98_com_fig.savefig(os.path.join("figures", patient_98_com_fig_name))
    plt.close(patient_98_com_fig)
    patient_98_com_thres_fig, _ = vis.visualise_preds_comparison_threshold(
        pred1=baseline_preds_98,
        pred2=proposed_preds_98,
        truth=truths,
        label1="Baseline Model",
        label2="Proposed Model",
        ticks_per_day=2,
    )
    patient_98_com_thres_fig_name = (
        "Patient 98 Predictions baseline vs proposed threshold.png"
    )
    patient_98_com_thres_fig.savefig(
        os.path.join("figures", patient_98_com_thres_fig_name)
    )
    plt.close(patient_98_com_thres_fig)

    # Ablation study for patient 98
    # ablation study for loss function
    patient_98_abla_loss_com_fig, _ = vis.visualise_preds_comparison(
        pred1=proposed_preds_98,
        pred2=ablation_loss_preds_98,
        truth=truths,
        label1="Proposed Model",
        label2="Ablation (MSE Loss)",
        ticks_per_day=2,
    )
    patient_98_abla_loss_com_fig_name = (
        "Patient 98 Predictions proposed vs ablation loss.png"
    )
    patient_98_abla_loss_com_fig.savefig(
        os.path.join("figures", patient_98_abla_loss_com_fig_name)
    )
    plt.close(patient_98_abla_loss_com_fig)

    patient_98_abla_loss_com_thres_fig, _ = vis.visualise_preds_comparison_threshold(
        pred1=proposed_preds_98,
        pred2=ablation_loss_preds_98,
        truth=truths,
        label1="Proposed Model",
        label2="Ablation (MSE Loss)",
        ticks_per_day=2,
    )
    patient_98_abla_loss_com_thres_fig_name = (
        "Patient 98 Predictions proposed vs ablation loss threshold.png"
    )
    patient_98_abla_loss_com_thres_fig.savefig(
        os.path.join("figures", patient_98_abla_loss_com_thres_fig_name)
    )
    plt.close(patient_98_abla_loss_com_thres_fig)

    # ablation study for physio layer
    patient_98_abla_phy_com_fig, _ = vis.visualise_preds_comparison(
        pred1=proposed_preds_98,
        pred2=ablation_phy_preds_98,
        truth=truths,
        label1="Proposed Model",
        label2="Ablation (Physio Layer)",
        ticks_per_day=2,
    )
    patient_98_abla_phy_com_fig_name = (
        "Patient 98 Predictions proposed vs ablation phy.png"
    )
    patient_98_abla_phy_com_fig.savefig(
        os.path.join("figures", patient_98_abla_phy_com_fig_name)
    )
    plt.close(patient_98_abla_phy_com_fig)

    patient_98_abla_phy_com_thres_fig, _ = vis.visualise_preds_comparison_threshold(
        pred1=proposed_preds_98,
        pred2=ablation_phy_preds_98,
        truth=truths,
        label1="Proposed Model",
        label2="Ablation (Physio Layer)",
        ticks_per_day=2,
    )
    patient_98_abla_phy_com_thres_fig_name = (
        "Patient 98 Predictions proposed vs ablation phy threshold.png"
    )
    patient_98_abla_phy_com_thres_fig.savefig(
        os.path.join("figures", patient_98_abla_phy_com_thres_fig_name)
    )
    plt.close(patient_98_abla_phy_com_thres_fig)

    # ablation study for dual input
    patient_98_abla_dual_com_fig, _ = vis.visualise_preds_comparison(
        pred1=proposed_preds_98,
        pred2=ablation_dual_preds_98,
        truth=truths,
        label1="Proposed Model",
        label2="Ablation (Dual Input)",
        ticks_per_day=2,
    )
    patient_98_abla_dual_com_fig_name = (
        "Patient 98 Predictions proposed vs ablation dual.png"
    )
    patient_98_abla_dual_com_fig.savefig(
        os.path.join("figures", patient_98_abla_dual_com_fig_name)
    )
    plt.close(patient_98_abla_dual_com_fig)
    patient_98_abla_dual_com_thres_fig, _ = vis.visualise_preds_comparison_threshold(
        pred1=proposed_preds_98,
        pred2=ablation_dual_preds_98,
        truth=truths,
        label1="Proposed Model",
        label2="Ablation (Dual Input)",
        ticks_per_day=2,
    )
    patient_98_abla_dual_com_thres_fig_name = (
        "Patient 98 Predictions proposed vs ablation dual threshold.png"
    )
    patient_98_abla_dual_com_thres_fig.savefig(
        os.path.join("figures", patient_98_abla_dual_com_thres_fig_name)
    )
    plt.close(patient_98_abla_dual_com_thres_fig)


if __name__ == "__main__":

    # create the folder for saving models and figures and results
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # run the main function
    main()
