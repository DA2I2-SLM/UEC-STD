import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

import xgboost as xgb

def create_error_corrector(ecm, input_dim, T, output_dim, hidden_dim, device):
    """
    Create error corrector model and training components based on ecm type.

    Args:
        ecm (str): Type of error corrector model.
        input_dim (int): Input feature dimension.
        T (int): Time steps (sequence length).
        output_dim (int): Output feature dimension.
        hidden_dim (int): Hidden layer dimension.
        device (torch.device): Device to place the model on.

    Returns:
        modelerr: error corrector model instance.
        optimizer: optimizer (None if not applicable).
        loss_fn: loss function (None if not applicable).
        scheduler: learning rate scheduler (None if not applicable).
        is_torch_model: bool indicating if model is a PyTorch model.
    """

    is_torch_model = True
    optimizer = None
    loss_fn = None
    scheduler = None

    if ecm == "linear":
        modelerr = ErrorCorrector(input_dim=input_dim, T=T, output_dim=output_dim, hidden_dim=hidden_dim).to(device)

    elif ecm == "logistic":
        modelerr = LogisticRegressionErrorCorrectorSklearn(input_dim=input_dim, T=T, output_dim=output_dim)
        is_torch_model = False
        return modelerr, optimizer, loss_fn, scheduler, is_torch_model
    elif ecm == "random_forest":
        modelerr = RandomForestErrorCorrector(input_dim=input_dim, T=T, output_dim=output_dim)
        is_torch_model = False
        return modelerr, optimizer, loss_fn, scheduler, is_torch_model

    elif ecm == "xgboost":
        modelerr = XGBoostErrorCorrector(input_dim=input_dim, T=T, output_dim=output_dim)
        is_torch_model = False
        return modelerr, optimizer, loss_fn, scheduler, is_torch_model

    elif ecm in ["lstm", "GRU"]:
        modelerr = RNNErrorCorrector(input_dim=input_dim, T=T, output_dim=output_dim,
                                    hidden_dim=hidden_dim, rnn_type=ecm).to(device)

    elif ecm == "CNN":
        modelerr = CNNErrorCorrector(input_dim=input_dim, T=T, output_dim=output_dim,
                                    hidden_dim=hidden_dim).to(device)

    elif ecm == "TF":
        modelerr = TransformerErrorCorrector(input_dim=input_dim, T=T, output_dim=output_dim,
                                            hidden_dim=hidden_dim).to(device)

    else:
        raise ValueError("Invalid ecm type. Choose from: 'linear', 'logistic', 'random_forest', 'xgboost', 'lstm', 'GRU', 'CNN', 'TF'.")

    # Setup optimizer, loss, scheduler for PyTorch models
    optimizer = torch.optim.Adam(modelerr.parameters(), lr=1e-2)
    loss_fn = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5, verbose=True)

    return modelerr, optimizer, loss_fn, scheduler, is_torch_model

##########################################################DL Model###############################

class ErrorCorrector(nn.Module):
    def __init__(self, input_dim,T, output_dim, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(T, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T),
            nn.Dropout(dropout)
        )
        self.model2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),

        )
        self.output_dim = output_dim

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        x = x.permute(0, 2, 1)        # (B, D, T)
        x = x.reshape(B * D, T)       # (B*D, T)
        out = self.model(x)          # (B*D, T)
        out = out.view(B, D, T)       # (B, D, T)

        return self.model2(out.permute(0, 2, 1))   # back to (B, T, D)
    

class LogisticErrorCorrector(nn.Module):
    def __init__(self, input_dim, T, output_dim, hidden_dim=32, dropout=0.5):
        super().__init__()
        # Linear layer to model temporal dimension
        self.temporal_linear = nn.Linear(T, T)

        # Logistic regression
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # hidden layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)  # output layer
        )

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        x = x.permute(0, 2, 1)            # (B, D, T)
        x = x.reshape(B * D, T)           # (B*D, T)

        x = self.temporal_linear(x)       # (B*D, T)
        x = x.view(B, D, T).permute(0, 2, 1)  # (B, T, D)

        logits = self.classifier(x)       # (B, T, output_dim)
        probs = torch.sigmoid(logits) if self.classifier[-1].out_features == 1 else F.softmax(logits, dim=-1)

        return probs

class LogisticRegressionErrorCorrectorSklearn:
    def __init__(self, input_dim, T, output_dim, penalty='l2', C=1.0, tol=1e-3, max_iter=1000):
        self.input_dim = input_dim
        self.T = T
        self.output_dim = output_dim

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),   # keep 95% variance automatically
            ('regressor', Ridge(alpha=1.0,     
                                solver='sag',        # ← iterative solver            
                                max_iter=max_iter, 
                                tol=tol,             # ← convergence threshold  # ← your max iterations
                                random_state=42))
        ])

    def fit(self, X, y):
        """
        X: (B, T, D) input features
        y: (B, T, output_dim) targets (converted to flat 1D if binary classification)
        """
        B, T, D = X.shape
        assert D == self.input_dim

        # Flatten features
        X_flat = X.reshape(B, -1)

        # Flatten targets
        # if self.output_dim == 1:
        #     y_flat = y.reshape(B * T)
        # else:
        #     y_flat = y.reshape(B, -1)  # For multiclass regression-like tasks
        y_flat = y.reshape(B, -1)  # Ensure y is 2D for sklearn

        print(f"Training Logistic Regression with {X_flat.shape[0]} samples and {X_flat.shape[1]} features.")
        self.model.fit(X_flat, y_flat)
        print("Training complete.")
    
    def predict(self, X):
        """
        X: (B, T, D)
        Returns: (B, T, output_dim)
        """
        B, T, D = X.shape
        X_flat = X.reshape(B, -1)
        y_pred_flat = self.model.predict(X_flat)  # (B, T * output_dim)
        return y_pred_flat.reshape(B, T, self.output_dim)  


class RNNErrorCorrector(nn.Module):
    def __init__(self, input_dim, T, output_dim, hidden_dim=32, rnn_type='GRU', dropout=0.5, num_layers=1):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Recurrent model for temporal modeling
        rnn_cls = nn.GRU if self.rnn_type == 'GRU' else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection model
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: (B, T, D)
        # Recurrent part
        rnn_out, _ = self.rnn(x)  # rnn_out: (B, T, hidden_dim)

        # Project to desired output dimension
        out = self.output_proj(rnn_out)  # (B, T, output_dim)

        return out
    

class CNNErrorCorrector(nn.Module):
    def __init__(self, input_dim, T, output_dim, hidden_dim=32, dropout=0.5, kernel_size=3):
        super().__init__()
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Dropout(dropout)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape

        # Rearrange to (B, D, T) for Conv1d
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.temporal_conv(x)  # (B, D, T)
        x = x.permute(0, 2, 1)     # (B, T, D)

        # Project to output
        return self.output_proj(x)  # (B, T, output_dim)
    
class TransformerErrorCorrector(nn.Module):
    def __init__(self, input_dim, T, output_dim, hidden_dim=64, nhead=4, num_layers=2, dropout=0.5):
        super().__init__()
        self.T = T

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, T, hidden_dim))  # (1, T, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: (B, T, D)
        x = self.input_proj(x)  # (B, T, hidden_dim)
        x = x + self.pos_embedding[:, :self.T, :]  # add positional embedding
        x = self.transformer(x)
        return self.output_proj(x)

##########################################################Traditional Model###############################

class RandomForestErrorCorrector:
    def __init__(self, input_dim, T, output_dim, n_estimators=20, max_depth=6):
        """
        RandomForestErrorCorrector:
        - input_dim: number of features per time step (D)
        - T: sequence length (number of time steps)
        - output_dim: number of target dimensions per time step
        - n_estimators: number of trees in the forest (default=100)
        - max_depth: maximum depth of each tree (default=10)
        """
        self.input_dim = input_dim
        self.T = T
        self.output_dim = output_dim
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            verbose=2,
        )

    def fit(self, X, y):
        """
        X: (B, T, D) input features
        y: (B, T, output_dim) targets
        """
        B, T, D = X.shape
        assert D == self.input_dim

        # Flatten features: (B, T, D) -> (B, T*D)
        X_flat = X.reshape(B, -1)
        y_flat = y.reshape(B, -1)  # (B, T * output_dim)

        print(f"Training model with {B} samples and {T*D} input features.")

        self.model.fit(X_flat, y_flat)

        print("Training complete.")

    def predict(self, X):
        """
        X: (B, T, D)
        Returns: (B, T, output_dim)
        """
        B, T, D = X.shape
        X_flat = X.reshape(B, -1)
        y_pred_flat = self.model.predict(X_flat)  # (B, T * output_dim)
        return y_pred_flat.reshape(B, T, self.output_dim)    
    
class XGBoostErrorCorrector:
    def __init__(self, input_dim, T, output_dim, n_estimators=20, max_depth=6, learning_rate=0.3, subsample=1, early_stopping_rounds=10):
        """
        XGBoostErrorCorrector: Corrects errors using XGBoost.

        input_dim: The dimension of the input features (D).
        T: The number of time steps.
        output_dim: The dimension of the target output (typically same as input_dim).
        n_estimators: Number of boosting rounds (default 100).
        max_depth: Maximum depth of the trees (default 2).
        learning_rate: Learning rate for boosting (default 0.1).
        subsample: Subsample ratio (default 1, meaning no subsampling).
        early_stopping_rounds: Number of rounds with no improvement before stopping
        """
        self.input_dim = input_dim
        self.T = T
        self.output_dim = output_dim
        
        # Initialize XGBRegressor
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            n_jobs=-1,
            verbosity=2,
            tree_method='gpu_hist',  # Use GPU-accelerated method
            device='cuda'  # Use the first available CUDA GPU
        )
        
    def fit(self, X, y):
        """
        X: (B, T, D) input features
        y: (B, T, output_dim) targets
        """
        B, T, D = X.shape
        assert D == self.input_dim

        # Flatten features: (B, T, D) -> (B, T*D)
        X_flat = X.reshape(B, -1)
        y_flat = y.reshape(B, -1)  # (B, T * output_dim)

        print(f"Training XGBoost model with {B} samples and {T*D} input features.")

        # Train the model
        self.model.fit(X_flat, y_flat)

        print("Training complete.")

    def predict(self, X):
        """
        X: (B, T, D)
        Returns: (B, T, output_dim)
        """
        B, T, D = X.shape
        X_flat = X.reshape(B, -1)
        y_pred_flat = self.model.predict(X_flat)  # (B, T * output_dim)
        return y_pred_flat.reshape(B, T, self.output_dim)


def moving_avg(x, kernel_size=25):
    pad = (kernel_size - 1) // 2
    filt = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
    # assume x is [..., T], so we fold channels into batch
    b, t, c = x.shape
    x2 = x.permute(0,2,1).reshape(b*c, 1, t)           # [B*C,1,T]
    trend = F.conv1d(x2, filt, padding=pad).reshape(b, c, t).permute(0,2,1)
    return trend



