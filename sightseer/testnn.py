import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import time


class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, graph_input_dim, gcn_output_dim, d_model, nhead,
                 num_encoder_layers, dim_feedforward):
        super(HybridModel, self).__init__()

        # RNN (LSTM) layer
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

        # CNN layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Graph Convolutional Network (GCN) layer
        self.gcn = GCNConv(graph_input_dim, gcn_output_dim)

        # Fully connected (linear) layers
        self.fc1 = nn.Linear(hidden_size + 32 * 7 * 7 + d_model + gcn_output_dim,
                             512)  # Adjust the size based on your input
        self.fc2 = nn.Linear(512, num_classes)

        # Activation function
        self.relu = nn.ReLU()

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(512)

        self.time_calculator = {
            "cnn_calculate_time": 0.0,
            "pool_calculate_time": 0.0,
            "rnn_calculate_time": 0.0,
            "transformer_calculate_time": 0.0,
            "gcn_calculate_time": 0.0,
            "linear_calculate_time": 0.0,
            "cnn_size": "nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)",
            "pool_size": "nn.MaxPool2d(kernel_size=2, stride=2, padding=0)",
            "rnn_size": f"nn.LSTM({input_size}, {hidden_size}, batch_first=True)",
            "tf_encoder_size": f"nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward})",
            "tf_size": f"nn.TransformerEncoder(encoder_layer, num_layers={num_encoder_layers})",
            "gcn_size": f"GCNConv({graph_input_dim}, {gcn_output_dim})",
            "bn": "nn.BatchNorm1d(512)",
            "linear_size": f"nn.Linear({hidden_size} + 32 * 7 * 7 + {d_model} + {gcn_output_dim} >>> 512 >>> {num_classes})",
            "activate_func": "ReLU",
            "cnt": 0,
        }

    def forward(self, x_rnn, x_cnn, x_transformer, x_graph, edge_index):

        self.time_calculator["cnt"] += 1

        start_time = time.time()
        # RNN (LSTM) forward pass
        out_rnn, _ = self.rnn(x_rnn)
        out_rnn = out_rnn[:, -1, :]  # Take the last time step output
        end_time = time.time()
        self.time_calculator["rnn_calculate_time"] += end_time - start_time

        # CNN forward pass
        start_time = time.time()
        out_cnn = self.relu(self.conv1(x_cnn))
        end_time = time.time()
        self.time_calculator["cnn_calculate_time"] += end_time - start_time

        start_time = time.time()
        out_cnn = self.pool(out_cnn)
        end_time = time.time()
        self.time_calculator["pool_calculate_time"] += end_time - start_time

        out_cnn = out_cnn.view(out_cnn.size(0), -1)  # Flatten the output

        # Transformer encoder forward pass
        start_time = time.time()
        out_transformer = self.transformer_encoder(x_transformer)
        out_transformer = out_transformer.mean(dim=1)  # Global mean pooling for transformer data
        end_time = time.time()
        self.time_calculator["transformer_calculate_time"] += end_time - start_time

        # GCN forward pass
        start_time = time.time()
        out_gcn = self.gcn(x_graph, edge_index)
        out_gcn = out_gcn.mean(dim=0)  # Global mean pooling for graph data
        end_time = time.time()
        self.time_calculator["gcn_calculate_time"] += end_time - start_time

        # Concatenate all outputs
        print(out_rnn.shape, out_cnn.shape, out_transformer.shape, out_gcn.shape)
        out = torch.cat((out_rnn, out_cnn, out_transformer, out_gcn.reshape(32, 1)), dim=1)

        # Fully connected layers
        out = self.relu(self.bn1(self.fc1(out)))

        start_time = time.time()
        out = self.fc2(out)
        end_time = time.time()
        self.time_calculator["linear_calculate_time"] += end_time - start_time

        return out
