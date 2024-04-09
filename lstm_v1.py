import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, 
                 static_channels: int, 
                 dynamic_history_channels: int, 
                 dynamic_future_channels: int, 
                 d_model: int, 
                 layers: int, 
                 dropout: float, 
                 out_channels: int, 
                 tgt_len: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=static_channels + dynamic_history_channels, 
            hidden_size=d_model, 
            num_layers=layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.decoder = nn.LSTM(
            input_size=static_channels + dynamic_future_channels, 
            hidden_size=d_model, 
            num_layers=layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.projection = nn.Linear(d_model, out_channels)
        self.tgt_len = tgt_len

    def forward(self, static_features: torch.Tensor, 
                history_features: torch.Tensor, future_features: torch.Tensor) -> torch.Tensor:
        """
        static_features shape: (batch_size, static_channels)
        history_features shape: (batch_size, seq_len, history_channels)
        future_features shape: (batch_size, pred_len, future_channels)
        """
        batch_size, seq_len, _ = history_features.shape
        _, pred_len, _ = future_features.shape
        encoder_in = torch.cat([history_features, static_features.unsqueeze(1).repeat(1, seq_len, 1)], dim=2)
        enc_out, enc_state = self.encoder(encoder_in)
        dec_in = torch.cat([future_features, static_features.unsqueeze(1).repeat(1, pred_len, 1)], dim=2)
        dec_out, _ = self.decoder(dec_in, enc_state)
        out = torch.cat([enc_out[:, -1:, :], dec_out], dim=1)
        return self.projection(out)[:, -self.tgt_len:, :]
        