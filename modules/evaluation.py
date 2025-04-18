import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from modules.metrics import calculate_sensitivity_specificity


def evaluate_model(vars_dict, testX, testY, trainY, trainX, save_path=None):
    """
    Evaluate learned representations using KNN with grid search.

    Parameters:
    -----------
    vars_dict: dict
        Dictionary containing 'Wv' (projection matrices)
    testX, testY: list, np.ndarray
        Test multi-view data and true labels
    trainX, trainY: list, np.ndarray
        Training multi-view data and labels
    save_path: str or None
        Optional path to save ROC curve or confusion matrix

    Returns:
    --------
    acc, sen, spe, auc: float
        Evaluation metrics
    """
    Wv = vars_dict['Wv']
    n_views = len(Wv)

    # Step 1: Project train and test into latent space
    Hall_train = sum(X @ Wv[v].T for v, X in enumerate(trainX))
    Hall_test = sum(X @ Wv[v].T for v, X in enumerate(testX))

    # Step 2: GridSearchCV on KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=0
    )

    grid_search.fit(Hall_train, trainY)
    best_knn = grid_search.best_estimator_

    # Step 3: Predict on test set
    y_pred = best_knn.predict(Hall_test)
    y_proba = best_knn.predict_proba(Hall_test)[:, 1]  # binary case

    # Step 4: Compute metrics
    acc = accuracy_score(testY, y_pred)
    sen, spe = calculate_sensitivity_specificity(testY, y_pred)
    auc = roc_auc_score(testY, y_proba)

    # Optional enhancement: ROC curve or confusion matrix saving
    if save_path:
        from matplotlib import pyplot as plt
        fpr, tpr, _ = roc_curve(testY, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(f"{save_path}/roc_curve.png")
        plt.close()

    return acc, sen, spe, auc
