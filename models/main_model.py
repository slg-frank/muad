import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import MetricEncoder, TraceEncoder, LogEncoder
from models.graph_model import GraphModel1, SimpleAttention
from models.layers import LinearLayer


class UncertainBlock(nn.Module):
    def __init__(self, in_dim=64, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
        self.fc_mu = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())
        self.fc_var = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())
        self.attention_mu = SimpleAttention(64)
        self.attention_var = SimpleAttention(64)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        var = self.fc_var(encoded)
        mu_attn = self.attention_mu(mu)
        var_attn = self.attention_var(var)
        new_sample = self.reparametrize(mu_attn, var_attn)
        return new_sample, self.kl_loss(mu_attn, var_attn)

    def reparametrize(self, mu, var):
        std = var.sqrt()
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu, var):
        return -0.5 * torch.mean(torch.sum(1 + var - mu ** 2 - var.exp(), dim=-1))


class MainModel(nn.Module):
    def __init__(self, device, num_class=2, hidden_dim=[64], aloss=0.4, b_loss=0.01, **kwargs):
        super().__init__()
        self.device = device
        self.metric_encoder = MetricEncoder(device, **kwargs)
        self.trace_encoder = TraceEncoder(device, **kwargs)
        self.log_encoder = LogEncoder(device, **kwargs)

        self.metric_uncertain_block = UncertainBlock()
        self.log_uncertain_block = UncertainBlock()
        self.trace_uncertain_block = UncertainBlock()

        # TCP classifier and confidence layers
        self.TCPClassifierLayer_trace = LinearLayer(hidden_dim[0], num_class)
        self.TCPConfidenceLayer_trace = LinearLayer(hidden_dim[0], 1)
        self.TCPClassifierLayer_metric = LinearLayer(hidden_dim[0], num_class)
        self.TCPConfidenceLayer_metric = LinearLayer(hidden_dim[0], 1)
        self.TCPClassifierLayer_log = LinearLayer(hidden_dim[0], num_class)
        self.TCPConfidenceLayer_log = LinearLayer(hidden_dim[0], 1)
        self.TCPConfidenceLayer = LinearLayer(192, 1)

        # Multimodal classifier
        mm_layers = []
        for i in range(1, len(hidden_dim)):
            in_dim = 3 * hidden_dim[0] if i == 1 else hidden_dim[i - 1]
            mm_layers.append(LinearLayer(in_dim, hidden_dim[i]))
            mm_layers.append(nn.ReLU())
        mm_layers.append(LinearLayer(hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*mm_layers)

        self.k1 = aloss
        self.k2 = b_loss

    def forward(self, graph, fault_indexs):
        batch_size = graph.batch_size
        metric = self.metric_encoder(graph)
        trace = self.trace_encoder(graph)
        log = self.log_encoder(graph)

        new_metric, m_kl_loss = self.metric_uncertain_block(metric["metric_embedding1"])
        new_trace, t_kl_loss = self.trace_uncertain_block(trace["trace_embedding1"])
        new_log, l_kl_loss = self.log_uncertain_block(log["log_embedding"])

        mean_kl_loss = (m_kl_loss + t_kl_loss + l_kl_loss) / 3
        y_anomaly = (fault_indexs >= 1).long()

        criterion = nn.CrossEntropyLoss()
        # Compute confidence loss for each modality
        confidence_loss_trace = self._compute_confidence_loss(
            new_trace, y_anomaly, self.TCPClassifierLayer_trace, self.TCPConfidenceLayer_trace, criterion)
        confidence_loss_metric = self._compute_confidence_loss(
            new_metric, y_anomaly, self.TCPClassifierLayer_metric, self.TCPConfidenceLayer_metric, criterion)
        confidence_loss_log = self._compute_confidence_loss(
            new_log, y_anomaly, self.TCPClassifierLayer_log, self.TCPConfidenceLayer_log, criterion)
        confidence_loss = confidence_loss_trace + confidence_loss_metric + confidence_loss_log

        # Feature fusion
        TCPConfidence_trace_sig = torch.sigmoid(self.TCPConfidenceLayer_trace(new_trace))
        TCPConfidence_metric_sig = torch.sigmoid(self.TCPConfidenceLayer_metric(new_metric))
        TCPConfidence_log_sig = torch.sigmoid(self.TCPConfidenceLayer_log(new_log))

        feature = torch.cat((
            new_trace * TCPConfidence_trace_sig,
            new_metric * TCPConfidence_metric_sig,
            new_log * TCPConfidence_log_sig
        ), dim=-1)

        MMlogit = self.MMClasifier(feature)
        MMLoss = criterion(MMlogit, y_anomaly)

        # Total loss
        total_loss = MMLoss + 0.6 * confidence_loss + mean_kl_loss
        y_pred = self.inference(MMlogit)

        return {
            'MMlogit': MMlogit,
            'loss': total_loss,
            'y_pred': y_pred,
            'feture': feature,
            'TCPConfidence_sig': torch.sigmoid(self.TCPConfidenceLayer(feature))
        }

    def _compute_confidence_loss(self, x, y, classifier, confidence_layer, criterion):
        logit = classifier(x)
        prob = F.softmax(logit, dim=1)
        p_target = torch.gather(prob, 1, y.unsqueeze(1)).squeeze()
        confidence = torch.sigmoid(confidence_layer(x)).squeeze()
        return F.mse_loss(confidence, p_target) + criterion(logit, y)

    def inference(self, MMlogit):
        dect_logit = MMlogit.detach().argmax(dim=1)
        return (dect_logit >= 1).long().tolist()