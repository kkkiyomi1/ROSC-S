import numpy as np
from config import TrainingConfig
from modules.data_preprocessing import preprocess_similarity_graph
from modules.initialization import initialize_variables
from modules.optimization import optimize_model
from modules.evaluation import evaluate_model
from data.load_adni import load_adni_data  # Replace with real loader


def main():
    config = TrainingConfig()

    # Set seed for reproducibility
    np.random.seed(config.seed)

    # Load data (replace with your actual loader)
    X_train, Y_train, X_test, Y_test = load_adni_data()

    # Preprocess similarity graph
    S, LS = preprocess_similarity_graph(X_train)

    # Initialize model variables
    variables = initialize_variables(X_train, Y_train, config.nv)

    # Train the model
    optimized_vars = optimize_model(
        X_train, Y_train, X_test, Y_test,
        S, LS,
        mu=config.mu,
        lambda_values=config.lambda_tuple(),
        nv=config.nv,
        rho=config.rho,
        mu_0=config.mu_0,
        vars_dict=variables,
        max_iters=config.max_iters,
        tol=config.tol,
        learning_rate=config.learning_rate,
        verbose=True
    )

    # Evaluate the final model
    acc, sen, spe, auc = evaluate_model(
        optimized_vars, X_test, Y_test, Y_train, X_train
    )

    print(f"\n✅ Final Evaluation\nACC: {acc:.4f} | SEN: {sen:.4f} | SPE: {spe:.4f} | AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
