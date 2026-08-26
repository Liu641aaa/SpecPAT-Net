import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from proprecess import *
from processing import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
import os
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Input, Dropout,AveragePooling1D
from keras.utils import to_categorical
from keras.layers import BatchNormalization
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import StandardScaler  # MinMaxScaler



class SpectralTransformer1(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(SpectralTransformer1, self).__init__()

        self.input_proj = nn.Linear(1, hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch_size, num_bands] -> [batch_size, num_bands, 1]
        x = x.unsqueeze(-1)

        x = self.input_proj(x)  # [batch_size, num_bands, d_model]

        # Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, num_bands, d_model]

        x = x.permute(0, 2, 1)  # [batch_size, d_model, num_bands]
        x = self.pool(x).squeeze(-1)  # [batch_size, d_model]

        x = self.output_layer(x)  # [batch_size, 1]
        return x
    


class SpectralTransformerV2_Lite_Improved_V4(nn.Module):
    def __init__(self, num_bands, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1, wavelengths=None):
        super().__init__()
        # --- CNN ---
        self.cnn = nn.Sequential(
                    nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
        self.cnn_output_seq_len = num_bands // 4
        self.cnn_to_transformer = nn.Linear(self.cnn_output_seq_len * hidden_dim, hidden_dim)
        
        self.pos_encoding = PositionalEncodingFromWavelength(wavelengths, hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: [B, seq_len]
        x = x.unsqueeze(-1)                         # [B, seq_len, 1]
        x = self.embedding(x)                       # [B, seq_len, hidden_dim]
        x = self.pos_encoding(x)                    # 
        x = self.encoder(x)                         # Transformer 

        weights = self.attention_pool(x)            # [B, seq_len, 1]
        x = (x * weights).sum(dim=1)                # [B, hidden_dim]

        out = self.output_layer(x)                  # [B, 1]
        return out
    
