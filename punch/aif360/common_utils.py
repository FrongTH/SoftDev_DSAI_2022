import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from aif360.metrics import ClassificationMetric


def compute_metrics(dataset_true, dataset_pred,
                    unprivileged_groups, privileged_groups,
                    disp=True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                  dataset_pred,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate() +
                                        classified_metric_pred.true_negative_rate())
    metrics["Precision"] = classified_metric_pred.precision()
    metrics["Recall"] = classified_metric_pred.recall()
    metrics["F1"] = 2 * ((classified_metric_pred.precision() * classified_metric_pred.recall()
                          ) / (classified_metric_pred.precision()+classified_metric_pred.recall()))
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics


def plot_fairness_impact(thresholds, optimal_treshold, optimal_thresh_index, balaced_accuracy, metrics_values, met_type):

    fig, ax1 = plt.subplots(figsize=(10, 7))

    ax1.plot(thresholds, balaced_accuracy)
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b',
                   fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    if met_type == "DI":
        ax2.plot(thresholds, np.abs(1.0-np.array(metrics_values)), color='r')
        ax2.set_ylabel('abs(1-disparate impact)', color='r',
                       fontsize=16, fontweight='bold')
    if met_type == "AOD":
        ax2.plot(thresholds, metrics_values, color='r')
        ax2.set_ylabel('avg. odds diff.', color='r',
                       fontsize=16, fontweight='bold')

    ax2.axvline(optimal_treshold, color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)

    print("Best balance accuracy: %.4f" %
          balaced_accuracy[optimal_thresh_index])
    if met_type == "DI":
        print("abs(1-disparate impact): %.4f" %
              np.abs(1.0-np.array(metrics_values[optimal_thresh_index])))
    if met_type == "AOD":
        print("Avg. odds diff: %.4f" % metrics_values[optimal_thresh_index])
