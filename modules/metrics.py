from sklearn.metrics import confusion_matrix


def calculate_sensitivity_specificity(y_true, y_pred):
    """
    Calculate sensitivity and specificity from confusion matrix.

    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return sensitivity, specificity
    else:
        # For multi-class (if extended later), return dummy values
        return 0.0, 0.0
