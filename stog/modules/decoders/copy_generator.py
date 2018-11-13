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

    def forward(self, hidden, attention, attention_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying target nodes.

        :param hidden: decoder outputs, [batch_size, num_target_nodes, hidden_size]
        :param attention: attention of each target node,
            [batch_size, num_target_nodes, num_target_nodes]
        :param attention_map: a sparse indicator matrix
            mapping each target word to its index  in the dynamic vocabulary.
            [batch_size, num_target_nodes, dynamic_vocab_size]
        """
        batch_size, num_target_nodes, _ = hidden.size()
        hidden = hidden.view(batch_size * num_target_nodes, -1)

        # Copying probability.
        p_copy = self.sigmoid(self.linear_copy(hidden))
        p_copy = p_copy.view(batch_size, num_target_nodes, 1)
        # The first target node is always generated.
        p_copy[:, 0] = 0

        # Generating probability.
        p_generate = 1 - p_copy

        # Probability distribution over the vocabulary.
        # [batch_size * num_target_nodes, vocab_size]
        score = self.linear(hidden)
        score[:, self.vocab_pad_idx] = -float('inf')
        # [batch_size, num_target_nodes, vocab_size]
        score = score.view(batch_size, num_target_nodes, -1)
        vocab_prob = self.softmax(score)
        scaled_vocab_prob = torch.mul(vocab_prob, p_generate.expand_as(vocab_prob))

        # Probability distribution over the dynamic vocabulary.
        # [batch_size, num_target_nodes, num_target_nodes]
        # TODO: make sure for target_node_i, its attention to target_node_j >= target_node_i
        # should be zero.
        scaled_attention = torch.mul(attention, p_copy.expand_as(attention))
        # [batch_size, num_target_nodes, dymanic_vocab_size]
        scaled_copy_prob = torch.bmm(scaled_attention, attention_map.transpose(0, 1))

        # [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        prob = torch.cat([scaled_vocab_prob.contiguous(), scaled_copy_prob.contiguous()], dim=2)

        _, prediction = prob.max(2)

        return dict(
            prob=prob,
            prediction=prediction,
        )

    def compute_loss(self, prob, prediction, vocab_target, copy_target):
        """

        :param prob: probability distribution,
            [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        :param prediction: [batch_size, num_target_nodes]
        :param vocab_target: target node index in the vocabulary,
            [batch_size, num_target_nodes]
        :param copy_target:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        """
        # [batch_size, num_target_nodes, 1]
        copy_target_with_offset = copy_target.unsqueeze(2) + self.vocab_size
        # [batch_size, num_target_nodes]
        copy_target_prob = prob.gather(dim=2, index=copy_target_with_offset).squeeze(2)

        # Exclude copy-oov nodes; copy oovs mean that nodes should be generated.
        copy_not_oov_mask = copy_target.ne(self.copy_oov_idx).float()
        copy_oov_mask = copy_target.eq(self.copy_oov_idx).float()
        copy_target_prob = copy_target_prob.mul(copy_not_oov_mask) + self.eps

        # [batch_size, num_target_nodes]
        vocab_target_prob = prob.gather(dim=2, index=vocab_target.unsqueeze(2)).squeeze(2)

        if self.force_copy:
            # Except copy-oov nodes, all other nodes should be copied.
            likelihood = copy_target_prob + vocab_target_prob.mul(copy_oov_mask)
        else:
            vocab_not_oov_mask = vocab_target.ne(self.vocab_oov_idx).float()
            vocab_oov_mask = vocab_target.eq(self.vocab_oov_idx).float()
            # Add prob for non-oov nodes in vocab
            # This means that non copy-oov nodes can be either copied or generated,
            # As long as they are not vocab-oov nodes.
            likelihood = copy_target_prob + vocab_target_prob.mul(vocab_not_oov_mask)
            # Add prob for oov nodes in both vocab and copy
            likelihood = likelihood + vocab_target_prob.mul(vocab_oov_mask).mul(copy_oov_mask)

        # Drop pads.
        loss = -likelihood.log().mul(vocab_target.ne(self.vocab_pad_idx).float())

        # Copy happens when the copy target is not oov.
        correct_copy_mask = copy_target.ne(self.copy_oov_idx)
        correct_copy = (copy_target + self.vocab_size) * correct_copy_mask.long()
        # Set the place where copy happens to zero.
        correct_vocab = vocab_target * (1 - correct_copy_mask).long()

        target = correct_vocab + correct_copy
        non_pad = target.ne(self.vocab_pad_idx)
        num_correct = prediction.eq(target).masked_select(non_pad).sum().item()
        num_non_pad = non_pad.sum().item()
        self.metrics(loss.sum().item(), num_non_pad, num_correct)

        return dict(
            loss=loss.div(float(num_non_pad)),
            prediction=prediction
        )
