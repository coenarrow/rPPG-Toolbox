import torch
import torch.nn as nn

class CCC_Loss(nn.Module):
    """
    The CCC_Loss module computes the loss based on the Concordance Correlation 
    Coefficient (CCC) for each sample in the batch. The loss for a single sample is:
    
        loss = 1 - CCC,
        
    where the CCC is computed per channel as:
    
        CCC = (2 * cov(x, y)) / (var(x) + var(y) + (mean(x) - mean(y))^2 + epsilon)
    
    The loss is averaged over all channels and samples.
    """
    def __init__(self):
        super(CCC_Loss, self).__init__()

    def forward(self, preds, labels):
        """
        Args:
            preds: Tensor of shape (batch_size, channels, frames)
            labels: Tensor of shape (batch_size, channels, frames)
                    or possibly (batch_size, frames, channels)
        Returns:
            loss: Averaged CCC loss over the batch.
        """
        epsilon = 1e-8  # small constant to prevent division by zero
        batch_size = preds.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            # Extract the i-th sample from predictions and labels
            x = preds[i]
            y = labels[i]

            # Ensure that x and y are at least 2D. 
            # If they are 1D, assume a single channel and unsqueeze.
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Now shape becomes (1, frames)
            if y.dim() == 1:
                y = y.unsqueeze(0)
            
            # Check if shapes are mismatched (e.g., (channels, frames) vs (frames, channels))
            if x.shape != y.shape:
                # If transposing y fixes the mismatch, do it.
                if x.shape == (y.shape[1], y.shape[0]):
                    y = y.transpose(0, 1)
                else:
                    raise ValueError(f"Shape mismatch: pred {x.shape} vs label {y.shape}")
            
            # Now x and y should both have shape (channels, frames)
            # Compute means along the time dimension
            mean_x = torch.mean(x, dim=-1, keepdim=True)  # shape: (channels, 1)
            mean_y = torch.mean(y, dim=-1, keepdim=True)
            
            # Compute variances along the time dimension
            var_x = torch.mean((x - mean_x) ** 2, dim=-1, keepdim=True)
            var_y = torch.mean((y - mean_y) ** 2, dim=-1, keepdim=True)
            
            # Compute covariance along the time dimension
            cov_xy = torch.mean((x - mean_x) * (y - mean_y), dim=-1, keepdim=True)
            
            # Compute the CCC per channel
            ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2 + epsilon)
            
            # Compute loss for this sample by averaging the loss over channels
            sample_loss = torch.mean(1 - ccc)
            total_loss += sample_loss
            
        # Average the loss over the batch
        loss = total_loss / batch_size
        return loss
