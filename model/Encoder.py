import torch.nn
import torch


class FullyConnectedOutput(torch.nn.Module):
    def __init__(self, embed_dim, input_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, embed_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1)
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)

    def forward(self, x):

        x = self.norm(x)
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(x)

        return out


def attention(Q, K, V, mask=None):
    l = Q.shape[2]
    num_head = Q.shape[1]
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    score /= (Q.shape[-1] ** 0.5)

    if mask is not None:
        mask = torch.abs(mask)
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, 4, -1, -1)
        score = score * mask

    score = torch.softmax(score, dim=-1)
    x = torch.matmul(score, V)

    x = x.permute(0, 2, 1, 3).reshape(-1, l, num_head * Q.shape[3])
    return x


class MultiHead(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super().__init__()
        self.fc_Q = torch.nn.Linear(input_dim, 32)
        self.fc_K = torch.nn.Linear(input_dim, 32)
        self.fc_V = torch.nn.Linear(input_dim, 32)

        self.num_head = num_head

        self.out_fc = torch.nn.Linear(32, embed_dim)

        self.norm = torch.nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask=None):

        # Q, K, V = [b, 50, 32]
        b = Q.shape[0]
        len = Q.shape[1]

        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        Q = Q.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        K = K.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        V = V.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)

        score = attention(Q, K, V, mask)
        score = self.dropout(self.out_fc(score))

        return score


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super(EncoderLayer, self).__init__()
        self.mh = MultiHead(input_dim, num_head, embed_dim)
        self.fc = FullyConnectedOutput(embed_dim, input_dim)

    def forward(self, x, mask=None):
        score = self.mh(x, x, x, mask)
        out = self.fc(score)

        return out


class FCEncoder(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super(FCEncoder, self).__init__()
        self.layer = EncoderLayer(input_dim, num_head, embed_dim)

    def forward(self, x, mask=None):
        x = self.layer(x, mask)

        return x
