import torch

from stog.metrics.seq2seq_metrics import Seq2SeqMetrics


class CopyGenerator(torch.nn.Module):

    def __init__(self, input_size, vocab_size, force_copy,
                 vocab_pad_idx, vocab_oov_idx, copy_oov_idx):
        super(CopyGenerator, self).__init__()
        self.linear = torch.nn.Linear(input_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.linear_copy = torch.nn.Linear(input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.metrics = Seq2SeqMetrics()

        self.force_copy = force_copy

        self.vocab_size = vocab_size
        self.vocab_pad_idx = vocab_pad_idx
        self.vocab_oov_idx = vocab_oov_idx
        self.copy_oov_idx = copy_oov_idx

        self.eps = 1e-20

    def forward(self, hiddens, attentions, attention_maps):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying target nodes.

        :param hiddens: decoder outputs, [batch_size, num_target_nodes, hidden_size]
        :param attentions: attention of each target node,
            [batch_size, num_target_nodes, num_target_nodes]
        :param attention_maps: a sparse indicator matrix
            mapping each target word to its index  in the dynamic vocabulary.
            [batch_size, num_target_nodes, dynamic_vocab_size]
        """
        batch_size, num_target_nodes, _ = hiddens.size()
        hiddens = hiddens.view(batch_size * num_target_nodes, -1)

        # Copying probability.
        p_copy = self.sigmoid(self.linear_copy(hiddens))
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
        scaled_copy_probs = torch.bmm(scaled_attentions, attention_maps)

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
        # [batch_size, num_target_nodes, 1]
        copy_targets_with_offset = copy_targets.unsqueeze(2) + self.vocab_size
        # [batch_size, num_target_nodes]
        copy_target_probs = probs.gather(dim=2, index=copy_targets_with_offset).squeeze(2)

        # Exclude copy-oov nodes; copy oovs mean that nodes should be generated.
        copy_not_oov_mask = copy_targets.ne(self.copy_oov_idx).float()
        copy_oov_mask = copy_targets.eq(self.copy_oov_idx).float()
        copy_target_probs = copy_target_probs.mul(copy_not_oov_mask) + self.eps

        # [batch_size, num_target_nodes]
        vocab_target_probs = probs.gather(dim=2, index=vocab_targets.unsqueeze(2)).squeeze(2)

        if self.force_copy:
            # Except copy-oov nodes, all other nodes should be copied.
            likelihood = copy_target_probs + vocab_target_probs.mul(copy_oov_mask)
        else:
            vocab_not_oov_mask = vocab_targets.ne(self.vocab_oov_idx).float()
            vocab_oov_mask = vocab_targets.eq(self.vocab_oov_idx).float()
            # Add prob for non-oov nodes in vocab
            # This means that non copy-oov nodes can be either copied or generated,
            # As long as they are not vocab-oov nodes.
            likelihood = copy_target_probs + vocab_target_probs.mul(vocab_not_oov_mask)
            # Add prob for oov nodes in both vocab and copy
            likelihood = likelihood + vocab_target_probs.mul(vocab_oov_mask).mul(copy_oov_mask)

        # Drop pads.
        loss = -likelihood.log().mul(vocab_targets.ne(self.vocab_pad_idx).float())

        # Copy happens when the copy target is not oov.
        correct_copy_mask = copy_targets.ne(self.copy_oov_idx)
        correct_copy = (copy_targets + self.vocab_size) * correct_copy_mask.long()
        # Set the place where copy happens to zero.
        correct_vocab = vocab_targets * (1 - correct_copy_mask).long()

        targets = correct_vocab + correct_copy
        non_pad = targets.ne(self.vocab_pad_idx)
        num_correct = predictions.eq(targets).masked_select(non_pad).sum().item()
        num_non_pad = non_pad.sum().item()
        self.metrics(loss.sum().item(), num_non_pad, num_correct)

        return dict(
            loss=loss.div(float(num_non_pad)),
            predictions=predictions
        )
