import torch

from stog.metrics.seq2seq_metrics import Seq2SeqMetrics


class CopyGenerator(torch.nn.Module):

    def __init__(self, input_size, switch_input_size, vocab_size, vocab_pad_idx, force_copy):
        super(CopyGenerator, self).__init__()
        self.linear = torch.nn.Linear(input_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.linear_copy = torch.nn.Linear(switch_input_size , 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.metrics = Seq2SeqMetrics()

        self.vocab_size = vocab_size
        self.vocab_pad_idx = vocab_pad_idx

        self.force_copy = force_copy

        self.eps = 1e-20

    def forward(self, hiddens, switchs, attentions, attention_maps):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying target nodes.

        :param hiddens: decoder outputs, [batch_size, num_target_nodes, hidden_size]
        :param augmented_hiddens: augmented decoder outputs,
            [batch_size, num_target_nodes, hidden_size * 2]
        :param attentions: attention of each target node,
            [batch_size, num_target_nodes, num_target_nodes]
        :param attention_maps: a sparse indicator matrix
            mapping each target word to its index  in the dynamic vocabulary.
            [batch_size, num_target_nodes, dynamic_vocab_size]
        """
        batch_size, num_target_nodes, _ = hiddens.size()
        hiddens = hiddens.view(batch_size * num_target_nodes, -1)
        switchs = switchs.view(batch_size * num_target_nodes, -1)

        # Copying probability.
        p_copy = self.sigmoid(self.linear_copy(switchs))
        p_copy = p_copy.view(batch_size, num_target_nodes, 1)
        # The first target node is always generated.
        # p_copy[:, 0] = 0

        # Generating probability.
        p_generate = 1 - p_copy

        # Probability distribution over the vocabulary.
        # [batch_size * num_target_nodes, vocab_size]
        scores = self.linear(hiddens)
        scores[:, self.vocab_pad_idx] = -float('inf')
        # [batch_size, num_target_nodes, vocab_size]
        scores = scores.view(batch_size, num_target_nodes, -1)
        vocab_probs = self.softmax(scores)
        scaled_vocab_probs = torch.mul(vocab_probs, p_generate.expand_as(vocab_probs))

        # Probability distribution over the dynamic vocabulary.
        # [batch_size, num_target_nodes, num_target_nodes]
        # TODO: make sure for target_node_i, its attention to target_node_j >= target_node_i
        # should be zero.
        scaled_attentions = torch.mul(attentions, p_copy.expand_as(attentions))
        # [batch_size, num_target_nodes, dymanic_vocab_size]
        scaled_copy_probs = torch.bmm(scaled_attentions, attention_maps.float())

        # [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        probs = torch.cat([scaled_vocab_probs.contiguous(), scaled_copy_probs.contiguous()], dim=2)

        _, predictions = probs.max(2)

        return dict(
            probs=probs,
            predictions=predictions,
        )

    def compute_loss(self, probs, predictions, vocab_targets, copy_targets):
        """

        :param probs: probability distribution,
            [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        :param predictions: [batch_size, num_target_nodes]
        :param vocab_targets: target node index in the vocabulary,
            [batch_size, num_target_nodes]
        :param copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        """
        batch_size, num_nodes = copy_targets.size()
        non_pad_mask = vocab_targets.ne(self.vocab_pad_idx)
        # If copy does not happen, then the copy target points to the node itself.
        self_pointer = torch.arange(1, num_nodes + 1).long().unsqueeze(0).type_as(copy_targets)
        copy_mask = copy_targets.ne(self_pointer)
        non_copy_mask = copy_targets.eq(self_pointer)

        # [batch_size, num_target_nodes, 1]
        copy_targets_with_offset = copy_targets.unsqueeze(2) + self.vocab_size
        # [batch_size, num_target_nodes]
        copy_target_probs = probs.gather(dim=2, index=copy_targets_with_offset).squeeze(2)
        copy_target_probs = copy_target_probs.mul(copy_mask.float()) + self.eps

        # [batch_size, num_target_nodes]
        vocab_target_probs = probs.gather(dim=2, index=vocab_targets.unsqueeze(2)).squeeze(2)

        if self.force_copy:
            # Except copy-oov nodes, all other nodes should be copied.
            likelihood = copy_target_probs + vocab_target_probs.mul(non_copy_mask.float())
        else:
            likelihood = copy_target_probs + vocab_target_probs

        # Drop pads.
        loss = -likelihood.log().mul(non_pad_mask.float())

        # Mask out copy targets for which copy does not happen.
        correct_copy = (copy_targets + self.vocab_size) * copy_mask.long()
        # Mask out vocab targets for which copy happens.
        correct_vocab = vocab_targets * non_copy_mask.long()

        targets = correct_vocab + correct_copy
        targets.masked_fill_(1 - non_pad_mask, self.vocab_pad_idx)
        pred_eq = predictions.eq(targets).mul(non_pad_mask)
        num_non_pad = non_pad_mask.sum().item()
        num_correct_pred = pred_eq.sum().item()
        num_copy = copy_mask.mul(non_pad_mask).sum().item()
        num_correct_copy = pred_eq.mul(copy_mask).sum().item()
        num_correct_binary = predictions.ge(self.vocab_size).mul(copy_mask).mul(non_pad_mask).sum().item()
        self.metrics(loss.sum().item(), num_non_pad, num_correct_pred,
                     num_copy, num_correct_copy, num_correct_binary)

        if self.force_copy:
            num_tokens = num_non_pad
        else:
            num_tokens = num_non_pad + copy_mask.sum().item()

        return dict(
            loss=loss.sum().div(float(num_tokens)),
            predictions=predictions
        )
