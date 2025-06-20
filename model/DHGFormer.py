from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import math
from model.Encoder import FCEncoder
import pickle


class CrossEmbed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def get_subnetwork_matrix(self, adjacency_matrix, subnetwork_ends):
        subnetwork_starts = [0] + subnetwork_ends[:-1]
        num_subnetworks = len(subnetwork_ends)
        batch_size = adjacency_matrix.shape[0]
        subnetwork_matrix = torch.zeros((batch_size, num_subnetworks, num_subnetworks),
                                        device=adjacency_matrix.device)

        for i in range(num_subnetworks):
            for j in range(i, num_subnetworks):
                block = adjacency_matrix[:,
                        subnetwork_starts[i]:subnetwork_ends[i],
                        subnetwork_starts[j]:subnetwork_ends[j]]
                mean_strength = block.mean(dim=(1, 2))
                subnetwork_matrix[:, i, j] = mean_strength
                subnetwork_matrix[:, j, i] = mean_strength
        return subnetwork_matrix

    def forward(self, embeddings, subnetwork_ends):
        # Compute full adjacency matrix
        adjacency_matrix = torch.einsum('ijk,ipk->ijp', embeddings, embeddings)

        roi_count = embeddings.shape[1]
        start_index = 0
        device = embeddings.device
        intra_mask = torch.zeros((roi_count, roi_count), dtype=torch.bool, device=device)

        for end_index in subnetwork_ends:
            intra_mask[start_index:end_index, start_index:end_index] = True
            start_index = end_index

        intra_adjacency = adjacency_matrix * intra_mask.unsqueeze(0)

        # Compute subnetwork-level connectivity
        inter_adjacency = self.get_subnetwork_matrix(adjacency_matrix, subnetwork_ends)

        # Add channel dimension for consistency
        intra_adjacency = torch.unsqueeze(intra_adjacency, -1)
        inter_adjacency = torch.unsqueeze(inter_adjacency, -1)
        adjacency_matrix = torch.unsqueeze(adjacency_matrix, -1)

        return intra_adjacency, inter_adjacency, adjacency_matrix


class CrossGCNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        self.roi_num = roi_num
        self.subnetwork_ends = [41, 70, 91, 110, 130, 137, 158, 200]

        # Graph convolution layers
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, roi_num),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(roi_num, roi_num)
        )
        self.bn1 = nn.BatchNorm1d(roi_num)

        self.gcn1 = nn.Sequential(
            nn.Linear(roi_num, roi_num),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = nn.BatchNorm1d(roi_num)

        self.gcn2 = nn.Sequential(
            nn.Linear(roi_num, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = nn.BatchNorm1d(roi_num)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(8 * roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

    def average_subnetwork_features(self, features, subnetwork_ends):
        batch_size, _, feature_dim = features.shape
        num_subnetworks = len(subnetwork_ends)
        subnetwork_starts = [0] + subnetwork_ends[:-1]
        subnetwork_features = torch.zeros((batch_size, num_subnetworks, feature_dim),
                                          device=features.device)

        for i in range(num_subnetworks):
            start_idx = subnetwork_starts[i]
            end_idx = subnetwork_ends[i]
            region_features = features[:, start_idx:end_idx, :]
            subnetwork_features[:, i, :] = region_features.mean(dim=1)

        return subnetwork_features

    def propagate_subnetwork_features(self, subnetwork_features, node_features, subnetwork_ends):
        subnetwork_starts = [0] + subnetwork_ends[:-1]
        num_subnetworks = len(subnetwork_ends)
        propagated_features = torch.zeros_like(node_features)

        for i in range(num_subnetworks):
            start_idx = subnetwork_starts[i]
            end_idx = subnetwork_ends[i]
            # Expand subnetwork feature to match region count
            expanded_features = subnetwork_features[:, i, :].unsqueeze(1).expand(-1, end_idx - start_idx, -1)
            propagated_features[:, start_idx:end_idx, :] = expanded_features

        # Combine original and propagated features
        return (node_features + propagated_features) / 2

    def forward(self, adjacency_matrix, intra_adjacency, inter_adjacency, node_features):
        batch_size = intra_adjacency.shape[0]

        # First propagation layer
        intra_features = torch.einsum('ijk,ijp->ijp', intra_adjacency, node_features)
        subnetwork_features = self.average_subnetwork_features(node_features, self.subnetwork_ends)
        subnetwork_features = torch.einsum('ijk,ijp->ijp', inter_adjacency, subnetwork_features)
        x = self.propagate_subnetwork_features(subnetwork_features, intra_features, self.subnetwork_ends)
        x = self.gcn(x)

        x = x.reshape((batch_size * self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((batch_size, self.roi_num, -1))

        # Second propagation layer
        intra_features = torch.einsum('ijk,ijp->ijp', intra_adjacency, x)
        subnetwork_features = self.average_subnetwork_features(x, self.subnetwork_ends)
        subnetwork_features = torch.einsum('ijk,ijp->ijp', inter_adjacency, subnetwork_features)
        x = self.propagate_subnetwork_features(subnetwork_features, intra_features, self.subnetwork_ends)
        x = self.gcn1(x)

        x = x.reshape((batch_size * self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((batch_size, self.roi_num, -1))

        # Third propagation layer
        intra_features = torch.einsum('ijk,ijp->ijp', intra_adjacency, x)
        subnetwork_features = self.average_subnetwork_features(x, self.subnetwork_ends)
        subnetwork_features = torch.einsum('ijk,ijp->ijp', inter_adjacency, subnetwork_features)
        x = self.propagate_subnetwork_features(subnetwork_features, intra_features, self.subnetwork_ends)
        x = self.gcn2(x)
        x = self.bn3(x)

        # Classifier
        x = x.view(batch_size, -1)
        return self.classifier(x)


class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.feature_proj = nn.Linear(input_dim * 2, input_dim)
        self.edge_predictor = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            class_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            return np.array(list(map(class_dict.get, labels)), dtype=np.int32)

        # Create receiver and sender matrices
        off_diag = np.ones([roi_num, roi_num])
        rel_rec = encode_onehot(np.where(off_diag)[0])
        rel_send = encode_onehot(np.where(off_diag)[1])

        self.receiver_matrix = torch.FloatTensor(rel_rec).cuda()
        self.sender_matrix = torch.FloatTensor(rel_send).cuda()

    def forward(self, embeddings):
        batch_size, region_count, _ = embeddings.shape

        receivers = torch.matmul(self.receiver_matrix, embeddings)
        senders = torch.matmul(self.sender_matrix, embeddings)

        # Concatenate and predict edges
        edge_features = torch.cat([senders, receivers], dim=2)
        edge_features = torch.relu(self.feature_proj(edge_features))
        edge_scores = self.edge_predictor(edge_features)
        edge_scores = torch.relu(edge_scores)

        # Reshape to adjacency matrix
        adjacency_matrix = edge_scores.reshape(batch_size, region_count, region_count, -1)
        return adjacency_matrix


class DHGFormer(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series_len=512):
        super().__init__()
        self.graph_generation = model_config['graph_generation']

        # Feature extractor
        if model_config['extractor_type'] == 'transformer':
            self.feature_extractor = FCEncoder(
                input_dim=time_series_len,
                num_head=4,
                embed_dim=model_config['embedding_size']
            )

        # Graph generator
        if self.graph_generation == "linear":
            self.graph_generator = Embed2GraphByLinear(
                model_config['embedding_size'],
                roi_num=roi_num
            )
        elif self.graph_generation == "product":
            self.graph_generator = CrossEmbed2GraphByProduct(
                model_config['embedding_size'],
                roi_num=roi_num
            )

        self.predictor = CrossGCNPredictor(node_feature_dim, roi_num=roi_num)

        # Load node cluster mapping
        with open('./node_clus_map.pickle', 'rb') as f:
            self.node_cluster_map = pickle.load(f)

        self.subnetwork_ends = [41, 70, 91, 110, 130, 137, 158, 200]
        self.cluster_order = list(self.node_cluster_map.keys())

    def reorder_nodes(self, features, dimension=1):
        """Reorder features according to cluster mapping"""
        return features[:, self.cluster_order, :] if dimension == 1 else \
            features[:, self.cluster_order, :][:, :, self.cluster_order]

    def forward(self, time_series: torch.Tensor, node_features: torch.Tensor):
        # Reorder inputs according to cluster mapping
        time_series = self.reorder_nodes(time_series, dimension=1)
        node_features = self.reorder_nodes(node_features, dimension=2)

        # Extract features and generate graph
        embeddings = self.feature_extractor(time_series, node_features)
        embeddings = F.softmax(embeddings, dim=-1)

        # Generate adjacency matrices
        intra_adjacency, inter_adjacency, full_adjacency = self.graph_generator(
            embeddings, self.subnetwork_ends
        )

        # Remove channel dimension
        full_adjacency = full_adjacency[:, :, :, 0]
        intra_adjacency = intra_adjacency[:, :, :, 0]
        inter_adjacency = inter_adjacency[:, :, :, 0]

        # Compute edge variance regularization
        batch_size = full_adjacency.shape[0]
        edge_variance = torch.mean(torch.var(full_adjacency.reshape((batch_size, -1)), dim=1))

        # Make prediction
        prediction = self.predictor(
            full_adjacency,
            intra_adjacency,
            inter_adjacency,
            node_features
        )

        return prediction, full_adjacency, edge_variance