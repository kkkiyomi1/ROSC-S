from dataclasses import dataclass


@dataclass
class TrainingConfig:
    seed: int = 42
    mu: float = 1e-2
    rho: float = 1.5
    mu_0: float = 1e4
    nv: int = 3  # latent dimension
    max_iters: int = 50
    tol: float = 1e-3
    learning_rate: float = 0.01
    lambda_1: float = 0.1
    lambda_2: float = 0.1
    lambda_3: float = 0.1
    lambda_4: float = 0.1

    def lambda_tuple(self):
        return (self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4)
