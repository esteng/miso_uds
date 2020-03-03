import numpy as np
import torch
import logging

from miso.metrics.continuous_metrics import ContinuousMetric

logger = logging.getLogger(__name__) 

np.set_printoptions(suppress=True)

class NodeAttributeDecoder(torch.nn.Module):
    def __init__(self,
                input_dim, 
                hidden_dim, 
                output_dim,
                n_layers,
                loss_multiplier = 1,
                loss_function = torch.nn.MSELoss(),
                activation = torch.nn.ReLU(),
                share_networks = False):
        super(NodeAttributeDecoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_multiplier = loss_multiplier
        self.attr_loss_function = loss_function
        self.mask_loss_function = torch.nn.BCEWithLogitsLoss()

        self.n_layers = n_layers
        self.activation = activation
        

        attr_input_layer = torch.nn.Linear(input_dim, hidden_dim)
        attr_hidden_layers = [torch.nn.Linear(hidden_dim, hidden_dim) 
                            for i in range(n_layers-1)]

        boolean_input_layer = torch.nn.Linear(input_dim, hidden_dim)
        boolean_hidden_layers = [torch.nn.Linear(hidden_dim, hidden_dim) 
                            for i in range(n_layers-1)]
    
        attr_output_layer = torch.nn.Linear(hidden_dim, output_dim)
        boolean_output_layer = torch.nn.Linear(hidden_dim, output_dim)

        all_attr_layers = [attr_input_layer] + attr_hidden_layers 
        all_boolean_layers = [boolean_input_layer] + boolean_hidden_layers 
         
        attr_net =   []
        boolean_net =   []
        for l in all_attr_layers:
            attr_net.append(l)
            attr_net.append(self.activation)

        for l in all_boolean_layers:
            boolean_net.append(l)
            boolean_net.append(self.activation)

        attr_net.append(attr_output_layer)
        boolean_net.append(boolean_output_layer)

        self.attribute_network = torch.nn.Sequential(*attr_net)

        if share_networks:
            self.boolean_network = self.attribute_network
        else:
            self.boolean_network = torch.nn.Sequential(*boolean_net)

        self.metrics = ContinuousMetric(prefix = "node")

    def forward(self, 
            decoder_output):
        """
        decoder_output: batch, target_len, input_dim
        """
        # get rid of eos
        output = decoder_output[:,:-1,:] 
        boolean_output = self.boolean_network(output)
        attr_output = self.attribute_network(output)

        return dict(
                pred_attributes= attr_output,
                pred_mask = boolean_output
               )

    def compute_loss(self, 
                    predicted_attrs,
                    predicted_mask,
                    target_attrs,
                    mask):

        # mask out non-predicted stuff
        predicted_attrs = predicted_attrs * mask
        target_attrs = target_attrs* mask
        attr_loss = self.attr_loss_function(predicted_attrs, target_attrs) * self.loss_multiplier
        # see if annotated at all; don't model annotator confidence, already modeled above
        mask_binary = torch.gt(mask, 0).float()
        mask_loss = self.mask_loss_function(predicted_mask, mask_binary) * self.loss_multiplier
        self.metrics(attr_loss)
        self.metrics(mask_loss)

        return dict(loss=attr_loss + mask_loss)

