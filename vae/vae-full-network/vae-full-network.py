import numpy as np

class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initialize weights here
        hidden_dim = 64
        eps = 0.05
        self.W1 = np.random.randn(input_dim, hidden_dim) * eps
        self.b1 = np.random.randn(hidden_dim) * eps
        
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * eps
        self.b_mu = np.random.randn(latent_dim) * eps

        self.W_log_var = np.random.randn(hidden_dim, latent_dim) * eps
        self.b_log_var = np.random.randn(latent_dim) * eps
        
        self.W_d = np.random.randn(latent_dim, input_dim) * eps
        self.b_d = np.random.randn(input_dim) * eps
      
    
    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass through VAE.
        """
        # Your implementation here
        h = np.maximum(0, x@self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_log_var + self.b_log_var

        e = np.random.randn(mu.shape[0], mu.shape[1])
        z = mu + np.exp(log_var/2) * e

        x_recon = z @ self.W_d + self.b_d
        
        return x_recon, mu, log_var
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples from prior.
        """
        # Your implementation here
        z = np.random.randn(n_samples, self.latent_dim)

        samples = z @ self.W_d + self.b_d
      
        return samples