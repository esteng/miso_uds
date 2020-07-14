import numpy as np
import torch

from miso.metrics.continuous_metrics import ContinuousMetric
from miso.modules.linear.bilinear import BiLinear
from miso.losses.loss import MSECrossEntropyLoss

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layer_list = [torch.nn.Linear(self.input_dim, self.hidden_dim ),
                                      torch.nn.ReLU()]
        for layer in range(n_layers - 1):
            layer_list += [torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                                      torch.nn.ReLU()]
        layer_list += [torch.nn.Linear(self.hidden_dim, output_dim)]

        self.net = torch.nn.Sequential(*layer_list)

    def forward(self, inp):
        out = self.net(inp)
        return out

class EdgeAttributeDecoder(torch.nn.Module):

    def __init__(
            self,
            h_input_dim, 
            hidden_dim,
            n_layers,
            output_dim, 
            loss_multiplier = 10,
            loss_function = MSECrossEntropyLoss,
            share_networks = False):
        super(EdgeAttributeDecoder, self).__init__()

        self.mask_loss_function = torch.nn.BCEWithLogitsLoss()
        self.h_input_dim = h_input_dim
        self.m_input_dim = h_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_multiplier = loss_multiplier
        self.loss_function = loss_function

        self.attr_bilinear = BiLinear(self.h_input_dim, self.m_input_dim, self.output_dim)
        self.attr_MLP = MLP(self.output_dim + self.h_input_dim + self.m_input_dim, hidden_dim, self.output_dim, n_layers)

        if share_networks:
            self.mask_bilinear = self.attr_bilinear
            self.mask_MLP = self.attr_MLP
        else:
            self.mask_bilinear = BiLinear(self.h_input_dim, self.m_input_dim, self.output_dim)
            self.mask_MLP = MLP(self.output_dim + self.h_input_dim + self.m_input_dim, hidden_dim, self.output_dim, n_layers)

        self.metrics = ContinuousMetric(prefix = "edge")

    def forward(self, edge_h, edge_m):
        # do bilinear
        attr_output = self.attr_bilinear(edge_h, edge_m)
        mask_output = self.mask_bilinear(edge_h, edge_m)
        # cat in the original as well
        attr_output = torch.cat([attr_output, edge_h, edge_m],  2)
        mask_output = torch.cat([mask_output, edge_h, edge_m],  2)
        # feedforward 
        attr_output = self.attr_MLP(attr_output)
        mask_output = self.mask_MLP(mask_output)

        return dict(pred_attributes = attr_output,
                    pred_mask = mask_output)

    def compute_loss(self, 
                    predicted_attrs,
                    predicted_mask,
                    target,
                    edge_attribute_mask):
        # mask it 
        predicted_attrs = predicted_attrs * edge_attribute_mask
        target = target * edge_attribute_mask

        
        attr_loss = self.loss_function(predicted_attrs, target) * self.loss_multiplier
        edge_mask_binary = torch.gt(edge_attribute_mask, 0).float()
        mask_loss = self.mask_loss_function(predicted_mask, edge_mask_binary) * self.loss_multiplier
        self.metrics(attr_loss.item())
        self.metrics(mask_loss.item())
        return dict(
                loss = attr_loss + mask_loss)

    @classmethod
    def from_params(cls, params, **kwargs):
        return cls(params['h_input_dim'],
                   params['hidden_dim'],
                   params['n_layers'],
                   params['output_dim'],
                   params.get("loss_multiplier", 1),
                   params.get("loss_function",  torch.nn.MSELoss()),
                   params.get("share_networks", False))



