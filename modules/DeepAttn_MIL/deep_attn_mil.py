
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepAttnMIL_Surv(nn.Module):
    """
    Deep Attention MIL Model
    Supports multi-class classification with configurable parameters
    """

    def __init__(self, in_dim=1024, embedding_dim=64, attention_dim=32, 
                 fc_dim=32, num_classes=2, dropout=0.5, cluster_num=10, act='relu'):
        super(DeepAttnMIL_Surv, self).__init__()
        
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.cluster_num = cluster_num
        
        # Select activation function
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'gelu':
            self.activation = nn.GELU()
        elif act == 'silu':
            self.activation = nn.SiLU()
        elif act == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Embedding network - Conv2d to process instance features
        self.embedding_net = nn.Sequential(
            nn.Conv2d(in_dim, embedding_dim, kernel_size=1),
            self.activation,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Attention network (V and W in the paper)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_dim),  # V
            nn.Tanh(),
            nn.Linear(attention_dim, 1)  # W
        )

        # Classification head (fc6 in original paper, adapted for multi-class)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, fc_dim),
            self.activation,
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, num_classes)
        )

    def masked_softmax(self, x, mask=None):
        """
        Performs masked softmax for attention weights
        
        Args:
            x: [1, N] attention scores
            mask: [1, N] binary mask (optional)
        
        Returns:
            Normalized attention weights [1, N]
        """
        if mask is not None:
            mask = mask.float()
            x_masked = x * mask + (1 - 1 / (mask + 1e-5))
        else:
            x_masked = x
            
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        
        if mask is not None:
            x_exp = x_exp * mask.float()
            
        return x_exp / (x_exp.sum(1).unsqueeze(-1) + 1e-8)

    def forward(self, x, mask=None):
        """
        Forward pass for WSI classification
        
        Args:
            x: Input features - supports two formats:
               1. Tensor [1, N, in_dim] or [N, in_dim]
               2. List of tensors (for cluster-based input)
            mask: Optional attention mask [1, N]
        
        Returns:
            Dictionary containing:
                - logits: [1, num_classes]
                - Y_prob: [1, num_classes]
                - Y_hat: [1, 1]
                - attention: [1, N]
                - embeddings: [N, embedding_dim]
        """
        device = x[0].device if isinstance(x, list) else x.device
        
        # ============ Process Input ============
        if isinstance(x, list):
            # Cluster-based input (list of tensors)
            embeddings = []
            for cluster_features in x:
                # Ensure shape is [N_i, in_dim, 1, 1]
                if cluster_features.dim() == 2:  # [N_i, in_dim]
                    cluster_features = cluster_features.unsqueeze(-1).unsqueeze(-1)
                
                embedded = self.embedding_net(cluster_features)  # [N_i, embedding_dim, 1, 1]
                embedded = embedded.view(embedded.size(0), -1)  # [N_i, embedding_dim]
                embeddings.append(embedded)
            
            h = torch.cat(embeddings, dim=0)  # [N_total, embedding_dim]
            
        else:
            # Standard tensor input
            if x.dim() == 3:  # [1, N, in_dim]
                x = x.squeeze(0)  # [N, in_dim]
            
            # Reshape to [N, in_dim, 1, 1] for Conv2d
            if x.dim() == 2:
                x = x.unsqueeze(-1).unsqueeze(-1)  # [N, in_dim, 1, 1]
            
            embedded = self.embedding_net(x)  # [N, embedding_dim, 1, 1]
            h = embedded.view(embedded.size(0), -1)  # [N, embedding_dim]

        # ============ Attention Mechanism ============
        A = self.attention(h)  # [N, 1]
        A = A.transpose(1, 0)  # [1, N]
        
        # Apply masked softmax
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # [1, N]
            A = self.masked_softmax(A, mask)
        else:
            A = F.softmax(A, dim=1)  # [1, N]

        # ============ Weighted Aggregation ============
        M = torch.mm(A, h)  # [1, embedding_dim]

        # ============ Classification ============
        logits = self.classifier(M)  # [1, num_classes]
        Y_prob = F.softmax(logits, dim=1)  # [1, num_classes]
        Y_hat = torch.argmax(logits, dim=1).reshape(1, 1)  # [1, 1]

        # ============ Return Results ============
        return {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'attention': A,
            'embeddings': h,
            'aggregated': M
        }

    def get_attention_weights(self, x, mask=None):
        """
        Extract attention weights for visualization
        
        Args:
            x: Input features
            mask: Optional mask
            
        Returns:
            attention: [1, N] attention weights
        """
        with torch.no_grad():
            output = self.forward(x, mask)
            return output['attention']
    
    def l1_regularization(self):
        """
        Compute L1 regularization term (as in original paper)
        
        Returns:
            L1 norm of all parameters
        """
        l1_reg = torch.tensor(0., device=next(self.parameters()).device)
        for param in self.parameters():
            l1_reg += torch.abs(param).sum()
        return l1_reg
 