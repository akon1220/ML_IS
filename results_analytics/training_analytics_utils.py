import torch

import sklearn.metrics as skmetrics
from torch.nn.functional import (
    threshold,
)  # see https://www.codegrepper.com/code-examples/python/roc+curve+pytorch

import os
from matplotlib import pyplot as plt
from matplotlib import cm
import pickle as pkl
import numpy as np
import argparse

from mpl_toolkits.mplot3d import Axes3D


def get_number_of_correct_preds_by_class(y_pred, y):
    # print("y == ", y)
    # print("torch.min(y) == ", torch.min(y).item())
    label_lower_bound = int(torch.min(y).item())
    # print("torch.max(y) == ", torch.max(y).item())
    label_upper_bound = int(torch.max(y).item()) + 1  # count the zero label
    num_represented_classes = label_upper_bound
    if label_lower_bound < 0:
        num_represented_classes -= label_lower_bound

    pred_stats_by_label = {
        i: {"num_correct": 0, "num_total": 0} for i in range(num_represented_classes)
    }  # label_lower_bound, label_upper_bound)}
    for curr_label in pred_stats_by_label.keys():
        label_mask = y == curr_label
        pred_indices = []
        for sample_ind in range(len(y)):
            if label_mask[sample_ind]:
                pred_indices.append(sample_ind)
        # label_mask = label_mask.long()
        # y_pred = y_pred.long()
        # y = y.long()

        if len(y[pred_indices]) > 0:
            # curr_num_correct = (torch.argmax(y_pred[label_mask], dim=1) == y[label_mask]).float().sum().item()
            # curr_num_total = len(y[label_mask])
            curr_num_correct, curr_num_total = get_number_of_correct_preds(
                y_pred[pred_indices], y[pred_indices]
            )
            pred_stats_by_label[curr_label]["num_correct"] = curr_num_correct
            pred_stats_by_label[curr_label]["num_total"] = curr_num_total

    return pred_stats_by_label


def get_number_of_correct_preds(y_pred, y):
    # print("y.size() == ", y.size())
    # print("y_pred.size() == ", y_pred.size())
    # print("y_pred after get_number_of_correct_preds call == ", y_pred)
    # print("y after get_number_of_correct_preds call == ", y)
    # print("torch.argmax(y_pred, dim=1) == ", torch.argmax(y_pred, dim=1))
    # print("(torch.argmax(y_pred, dim=1) == y) == ", (torch.argmax(y_pred, dim=1) == y))
    num_correct = (torch.argmax(y_pred, dim=1).view(y.size()) == y).float().sum().item()
    num_total = len(y)
    return int(num_correct), num_total


def get_accuracy_numerator_and_denominator_for_cdisn_preds(y_pred, y_labeled):
    # print("y_pred after get_accuracy_numerator_and_denominator_for_cdisn_preds call == ", y_pred)
    # print("y_labeled after get_accuracy_numerator_and_denominator_for_cdisn_preds call == ", y_labeled)
    new_num_correct = None
    new_num_total = None
    new_pred_stats_by_class = None

    # print("train.train_cl.get_accuracy_numerator_and_denominator_for_cdisn_preds: y_labeled.size() == ", y_labeled.size())
    new_num_correct, new_num_total = get_number_of_correct_preds(y_pred, y_labeled)
    new_pred_stats_by_class = get_number_of_correct_preds_by_class(y_pred, y_labeled)

    return new_num_correct, new_num_total, new_pred_stats_by_class


def get_roc_metrics_from_preds(preds, ground_truth):
    fpr, tpr, threshold = skmetrics.roc_curve(ground_truth, preds)
    roc_auc = skmetrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_cdisn_ensemble_training_avgs(
    avg_train_losses,
    avg_train_accs,
    avg_val_accs,
    avg_train_accs_by_class,
    avg_val_accs_by_class,
    avg_train_roc_aucs,
    avg_val_roc_aucs,
    plot_series_name,
    save_path,
):
    """
    Makes plots of average training losses, training accuracies, and validation accuracies.
    Saves the resulting plots to save_path.
    see utils.semi_supervised_results_analytics_utils.plot_InfoMin_Downstream_training_avgs for similar function
    """
    fig1, ax1 = plt.subplots()
    ax1.plot(avg_train_losses, label="training")
    ax1.set_yscale(
        "log"
    )  # see https://stackoverflow.com/questions/773814/plot-logarithmic-axes-with-matplotlib-in-python
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss (log-scale)")
    ax1.set_title(plot_series_name + ": Average Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    loss_plot_save_path = os.path.join(
        save_path, plot_series_name + "_training_loss_visualization.png"
    )
    fig1.savefig(loss_plot_save_path)
    plt.close()

    fig2, ax2 = plt.subplots()
    ax2.plot(avg_train_accs, label="training")
    ax2.plot(avg_val_accs, label="validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Accuracy")
    ax2.set_title(plot_series_name + ": Average Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    train_accuracy_plot_save_path = os.path.join(
        save_path, plot_series_name + "_accuracy_visualization.png"
    )
    fig2.savefig(train_accuracy_plot_save_path)
    plt.close()

    fig3, ax3 = plt.subplots()
    for label_id in avg_train_accs_by_class.keys():
        ax3.plot(avg_train_accs_by_class[label_id], label="class " + str(label_id))
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Average Training Accuracy")
    ax3.set_title(plot_series_name + ": Average Class-wise Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    classwise_train_accuracy_plot_save_path = os.path.join(
        save_path, plot_series_name + "_classwise_train_accuracy_visualization.png"
    )
    fig3.savefig(classwise_train_accuracy_plot_save_path)
    plt.close()

    fig4, ax4 = plt.subplots()
    for label_id in avg_val_accs_by_class.keys():
        ax4.plot(avg_val_accs_by_class[label_id], label="class " + str(label_id))
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Average Validation Accuracy")
    ax4.set_title(plot_series_name + ": Average Class-wise Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    classwise_val_accuracy_plot_save_path = os.path.join(
        save_path, plot_series_name + "_classwise_val_accuracy_visualization.png"
    )
    fig4.savefig(classwise_val_accuracy_plot_save_path)
    plt.close()

    fig5, ax5 = plt.subplots()
    ax5.plot(avg_train_roc_aucs, label="training")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Average ROC-AUC Score Per Batch")
    ax5.set_title(plot_series_name + ": Average Training ROC-AUC Scores")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    training_roc_auc_plot_save_path = os.path.join(
        save_path, plot_series_name + "_avg_train_roc_auc_visualization.png"
    )
    fig5.savefig(training_roc_auc_plot_save_path)
    plt.close()

    fig6, ax6 = plt.subplots()
    ax6.plot(avg_val_roc_aucs, label="validation")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Average ROC-AUC Score Per Batch")
    ax6.set_title(plot_series_name + ": Average Validation ROC-AUC Scores")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    val_roc_auc_plot_save_path = os.path.join(
        save_path, plot_series_name + "_avg_val_roc_auc_visualization.png"
    )
    fig6.savefig(val_roc_auc_plot_save_path)
    plt.close()
    pass


def plot_roc_auc_curve(fpr, tpr, roc_auc_score, plot_series_name, save_path):
    # see https://www.codegrepper.com/code-examples/python/roc+curve+pytorch
    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, "b", label="AUC=" + str(roc_auc_score))
    ax1.plot([0, 1], [0, 1], "r--")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(plot_series_name + ": Receiver Operating Characteristic (Test)")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plot_save_path = os.path.join(
        save_path, plot_series_name + "_test_roc_auc_visualization.png"
    )
    fig1.savefig(plot_save_path)
    plt.close()
    pass


def plot_cdisn_ensemble_retrain_summary(
    avg_train_losses,
    avg_train_accs,
    avg_val_accs,
    avg_train_roc_aucs,
    avg_val_roc_aucs,
    test_roc_metrics,
    plot_series_name,
    save_path,
):
    """
    Makes plots of average training losses, training accuracies, and validation accuracies.
    Saves the resulting plots to save_path.
    """
    fig1, ax1 = plt.subplots()
    for task_id in avg_train_losses.keys():
        ax1.plot(avg_train_losses[task_id], label=task_id)
    ax1.set_yscale(
        "log"
    )  # see https://stackoverflow.com/questions/773814/plot-logarithmic-axes-with-matplotlib-in-python
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss (log-scale)")
    ax1.set_title(plot_series_name + ": Average Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    loss_plot_save_path = os.path.join(
        save_path, plot_series_name + "_training_loss_visualization.png"
    )
    fig1.savefig(loss_plot_save_path)
    plt.close()

    fig2, ax2 = plt.subplots()
    for task_id in avg_train_accs.keys():
        ax2.plot(avg_train_accs[task_id], label=task_id)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Accuracy")
    ax2.set_title(plot_series_name + ": Average Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    train_accuracy_plot_save_path = os.path.join(
        save_path, plot_series_name + "_training_accuracy_visualization.png"
    )
    fig2.savefig(train_accuracy_plot_save_path)
    plt.close()

    fig3, ax3 = plt.subplots()
    for task_id in avg_val_accs.keys():
        ax3.plot(avg_val_accs[task_id], label=task_id)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Average Accuracy")
    ax3.set_title(plot_series_name + ": Average Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    val_accuracy_plot_save_path = os.path.join(
        save_path, plot_series_name + "_validation_accuracy_visualization.png"
    )
    fig3.savefig(val_accuracy_plot_save_path)
    plt.close()

    fig4, ax4 = plt.subplots()
    for task_id in avg_train_roc_aucs.keys():
        ax4.plot(avg_train_roc_aucs[task_id], label=task_id)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Average ROC-AUC Score Per Batch")
    ax4.set_title(plot_series_name + ": Average Training ROC-AUC Scores")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    training_roc_auc_plot_save_path = os.path.join(
        save_path, plot_series_name + "_avg_train_roc_auc_visualization.png"
    )
    fig4.savefig(training_roc_auc_plot_save_path)
    plt.close()

    fig5, ax5 = plt.subplots()
    for task_id in avg_val_roc_aucs.keys():
        ax5.plot(avg_val_roc_aucs[task_id], label=task_id)
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Average ROC-AUC Score Per Batch")
    ax5.set_title(plot_series_name + ": Average Validation ROC-AUC Scores")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    val_roc_auc_plot_save_path = os.path.join(
        save_path, plot_series_name + "_avg_validation_roc_auc_visualization.png"
    )
    fig5.savefig(val_roc_auc_plot_save_path)
    plt.close()

    fig6, ax6 = plt.subplots()
    ax6.plot([0, 1], [0, 1], "r--")
    for task_id in test_roc_metrics.keys():
        ax6.plot(
            test_roc_metrics[task_id]["fpr"],
            test_roc_metrics[task_id]["tpr"],
            label=task_id + " AUC=" + str(test_roc_metrics[task_id]["roc_auc"]),
        )
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.set_xlabel("False Positive Rate")
    ax6.set_ylabel("True Positive Rate")
    ax6.set_title(plot_series_name + ": Receiver Operating Characteristic (Test)")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    test_roc_auc_plot_save_path = os.path.join(
        save_path, plot_series_name + "_test_roc_metrics_visualization.png"
    )
    fig6.savefig(test_roc_auc_plot_save_path)
    plt.close()
    pass


def plot_cdisn_ensemble_historical_test_summary(
    historical_avg_test_accs, final_test_roc_metrics, save_path
):
    """
    Makes plots of average test accuracies.
    Saves the resulting plots to save_path.
    """
    fig1, ax1 = plt.subplots()
    for task_id in historical_avg_test_accs.keys():
        ax1.plot(historical_avg_test_accs[task_id], label=task_id)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Accuracy")
    ax1.set_title("Average Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plot_save_path = os.path.join(
        save_path, "historical_test_accuracy_visualization.png"
    )
    fig1.savefig(plot_save_path)
    plt.close()

    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [0, 1], "r--")
    for task_id in final_test_roc_metrics.keys():
        ax2.plot(
            final_test_roc_metrics[task_id]["fpr"],
            final_test_roc_metrics[task_id]["tpr"],
            label=task_id + " AUC=" + str(final_test_roc_metrics[task_id]["roc_auc"]),
        )
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Receiver Operating Characteristic (Test)")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    test_roc_auc_plot_save_path = os.path.join(
        save_path, "final_test_roc_metrics_visualization.png"
    )
    fig2.savefig(test_roc_auc_plot_save_path)
    plt.close()
    pass