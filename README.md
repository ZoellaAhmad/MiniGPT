# LLM
A personal project undertaken to deepen my understanding about the training of Large Language Models. It is an implementation of a generative, decoder-only Transformer language model (GPT architecture).

### Key Implementations:
* **Causal Masked Self-Attention:** Implemented multi-head attention blocks with lower-triangular causal masks to ensure strictly autoregressive token generation.
* **Deep Residual Architecture:** Utilized pre-activation Layer Normalization (Pre-LN) coupled with residual connections to optimize gradient scaling across deep layers.
* **Feed-Forward Networks (FFN):** Implemented position-wise feed-forward networks using linear projections and non-linear activations to project features across higher-dimensional latent spaces.

### Training:
Since local training on deep architectures was highly resource-intensive and not feasible on my personal computer, the model's training pipeline was fully offloaded to the cloud:
* **Cloud Compute:** Orchestrated and trained using **Google Colab's hosted GPU acceleration infrastructure**.
* **Hardware Utilization:** Leveraged PyTorch’s CUDA backend to optimize tensor operations, batch sizes, and gradient accumulation for faster convergence.
* **Optimization:** Finetuned using the **AdamW optimizer** to apply decoupled weight decay, enhancing generalization capabilities.
