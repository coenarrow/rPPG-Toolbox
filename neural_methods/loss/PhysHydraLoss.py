import torch
import torch.nn as nn

class CCC_Loss(nn.Module):
    """
    The CCC_Loss module computes the loss based on the Concordance Correlation 
    Coefficient (CCC) for each sample in the batch. The loss for a single sample is:
    
        loss = 1 - CCC,
        
    where the CCC is computed as:
    
        CCC = (2 * cov(x, y)) / (var(x) + var(y) + (mean(x) - mean(y))^2 + epsilon)
    
    Here, x is the predicted signal, y is the ground truth signal, and epsilon is a small
    value to avoid division by zero.
    """
    def __init__(self):
        super(CCC_Loss, self).__init__()

    def forward(self, preds, labels):
        loss = 0.0
        batch_size = preds.shape[0]
        epsilon = 1e-8  # small constant to prevent division by zero
        
        for i in range(batch_size):
            x = preds[i]
            y = labels[i]
            
            # Calculate means
            mean_x = torch.mean(x)
            mean_y = torch.mean(y)
            
            # Calculate variances (using the population definition)
            var_x = torch.mean((x - mean_x) ** 2)
            var_y = torch.mean((y - mean_y) ** 2)
            
            # Calculate covariance
            cov_xy = torch.mean((x - mean_x) * (y - mean_y))
            
            # Calculate the Concordance Correlation Coefficient (CCC)
            ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2 + epsilon)
            
            # Accumulate the loss for this sample
            loss += 1 - ccc
            
        # Average the loss over the batch
        loss = loss / batch_size
        return loss
