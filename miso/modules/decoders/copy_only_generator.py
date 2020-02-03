import torch

from stog.metrics.seq2seq_metrics import Seq2SeqMetrics


class CopyOnlyGenerator(torch.nn.Module):

    def __init__(
            self, 
            input_size, 
            switch_input_size, 
            vocab_pad_idx, 
            force_copy,
            source_copy=True, 
            target_copy=True
    ):
        super(CopyOnlyGenerator, self).__init__()
        
        self.softmax = torch.nn.Softmax(dim=-1)

        self.linear_pointer = torch.nn.Linear(switch_input_size, 3)
        self.sigmoid = torch.nn.Sigmoid()

        self.metrics = Seq2SeqMetrics()

        self.vocab_pad_idx = vocab_pad_idx

        self.source_copy = source_copy
        self.target_copy = target_copy

        self.force_copy = force_copy

        self.eps = 1e-20

        self.vocab_size = 1

    def forward(self, hiddens, source_attentions, source_attention_maps,
                target_attentions, target_attention_maps, invalid_indexes=None):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying target nodes.

        :param hiddens: decoder outputs, [batch_size, num_target_nodes, hidden_size]
        :param source_attentions: attention of each source node,
            [batch_size, num_target_nodes, num_source_nodes]
        :param source_attention_maps: a sparse indicator matrix
            mapping each source node to its index in the dynamic vocabulary.
            [batch_size, num_source_nodes, dynamic_vocab_size]
        :param target_attentions: attention of each target node,
            [batch_size, num_target_nodes, num_target_nodes]
        :param target_attention_maps: a sparse indicator matrix
            mapping each target node to its index in the dynamic vocabulary.
            [batch_size, num_target_nodes, dynamic_vocab_size]
        :param invalid_indexes: indexes which are not considered in prediction.
        """
        batch_size, num_target_nodes, _ = hiddens.size()
        source_dynamic_vocab_size = source_attention_maps.size(2)
        target_dynamic_vocab_size = target_attention_maps.size(2)
        hiddens = hiddens.view(batch_size * num_target_nodes, -1)

        # Pointer probability.
        p = torch.nn.functional.softmax(self.linear_pointer(hiddens), dim=1)

        if self.source_copy:
            p_copy_source = p[:, 0].view(batch_size, num_target_nodes, 1)
        else:
            p_copy_source = p.new_zeros(batch_size, num_target_nodes, 1)

        if self.target_copy:
            p_copy_target = p[:, 1].view(batch_size, num_target_nodes, 1)
        else:
            p_copy_target = p.new_zeros(batch_size, num_target_nodes, 1)

        # This is the prob of generate eos
        p_generate = p[:, 2].view(batch_size, num_target_nodes, 1)

        # [batch_size, num_target_nodes, num_source_nodes]
        scaled_source_attentions = torch.mul(source_attentions, p_copy_source.expand_as(source_attentions))
        # [batch_size, num_target_nodes, dynamic_vocab_size]
        scaled_copy_source_probs = torch.bmm(scaled_source_attentions, source_attention_maps.float())

        # Probability distribution over the dynamic vocabulary.
        # [batch_size, num_target_nodes, num_target_nodes]
        # TODO: make sure for target_node_i, its attention to target_node_j >= target_node_i
        # should be zero.
        scaled_target_attentions = torch.mul(target_attentions, p_copy_target.expand_as(target_attentions))
        # [batch_size, num_target_nodes, dymanic_vocab_size]
        scaled_copy_target_probs = torch.bmm(scaled_target_attentions, target_attention_maps.float())

        #if invalid_indexes:
        if False:
            if invalid_indexes.get('vocab', None) is not None:
                vocab_invalid_indexes = invalid_indexes['vocab']
                for i, indexes in enumerate(vocab_invalid_indexes):
                    for index in indexes:
                        scaled_vocab_probs[i, :, index] = 0

            if invalid_indexes.get('source_copy', None) is not None:
                source_copy_invalid_indexes = invalid_indexes['source_copy']
                for i, indexes in enumerate(source_copy_invalid_indexes):
                    for index in indexes:
                        scaled_copy_source_probs[i, :, index] = 0

        # [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        probs = torch.cat([
            p_generate.contiguous(),
            scaled_copy_source_probs.contiguous(),
            scaled_copy_target_probs.contiguous()
        ], dim=2)

        _, predictions = probs.max(2)

        return dict(
            probs=probs,
            predictions=predictions,
            source_dynamic_vocab_size=source_dynamic_vocab_size,
            target_dynamic_vocab_size=target_dynamic_vocab_size
        )

    def compute_loss(
            self, 
            probs, 
            predictions, 
            generate_targets,
            source_copy_targets, 
            source_dynamic_vocab_size,
            target_copy_targets, 
            target_dynamic_vocab_size,
            coverage_records, 
            copy_attentions
    ):
        """
        Priority: target_copy > source_copy > generate

        :param probs: probability distribution,
            [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        :param predictions: [batch_size, num_target_nodes]
        :param generate_targets: target node index in the vocabulary,
            [batch_size, num_target_nodes]
        :param source_copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        :param source_dynamic_vocab_size: int
        :param target_copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        :param target_dynamic_vocab_size: int
        :param coverage_records: None or a tensor recording source-side coverages.
            [batch_size, num_target_nodes, num_source_nodes]
        :param copy_attentions: [batch_size, num_target_nodes, num_source_nodes]
        """
        if not self.source_copy:
            source_copy_mask = source_copy_targets.gt(0)
            source_copy_targets = source_copy_mask.type_as(source_copy_targets) 

        if not self.target_copy:
            target_copy_targets = target_copy_targets.new_zeros(target_copy_targets.size())


        generate_mask = generate_targets.ne(self.vocab_pad_idx)

        source_copy_mask = source_copy_targets.ne(0)  # 1 is the index for unknown words
        non_source_copy_mask = 1 - source_copy_mask

        target_copy_mask = target_copy_targets.ne(0)  # 0 is the index for coref NA
        non_target_copy_mask = 1 - target_copy_mask

        non_pad_mask = source_copy_mask | target_copy_mask | generate_mask

        # [batch_size, num_target_nodes, 1]
        target_copy_targets_with_offset = target_copy_targets.unsqueeze(2) + self.vocab_size + source_dynamic_vocab_size
        # [batch_size, num_target_nodes]
        target_copy_target_probs = probs.gather(dim=2, index=target_copy_targets_with_offset).squeeze(2)
        target_copy_target_probs = target_copy_target_probs.mul(target_copy_mask.float())

        # [batch_size, num_target_nodes, 1]
        source_copy_targets_with_offset = source_copy_targets.unsqueeze(2) + self.vocab_size
        # [batch_size, num_target_nodes]
        source_copy_target_probs = probs.gather(dim=2, index=source_copy_targets_with_offset).squeeze(2)
        source_copy_target_probs = source_copy_target_probs.mul(non_target_copy_mask.float()).mul(source_copy_mask.float())

        # [batch_size, num_target_nodes]
        generate_target_probs = probs[:, :, 0] * generate_targets.type_as(probs)

        # Except copy-oov nodes, all other nodes should be copied.
        likelihood = target_copy_target_probs + source_copy_target_probs + \
                     generate_target_probs.mul(non_target_copy_mask.float()).mul(non_source_copy_mask.float())
        num_tokens = non_pad_mask.sum().item()

        if not self.force_copy:
            non_generate_oov_mask = generate_targets.ne(1)
            additional_generate_mask = (non_target_copy_mask & source_copy_mask & non_generate_oov_mask)
            likelihood = likelihood + generate_target_probs.mul(additional_generate_mask.float())
            num_tokens += additional_generate_mask.sum().item()

        # Add eps for numerical stability.
        likelihood = likelihood + self.eps

        coverage_loss = 0
        if coverage_records is not None:
            coverage_loss = torch.sum(
                torch.min(coverage_records, copy_attentions), 2).mul(non_pad_mask.float())

        # Drop pads.
        loss = -likelihood.log().mul(non_pad_mask.float()) + coverage_loss

        # Mask out copy targets for which copy does not happen.
        # We don't add generate targets here because it's zero (only one token : eos)
        # Target mask will handle the padding problem
        targets = target_copy_targets_with_offset.squeeze(2) * target_copy_mask.long() + \
            source_copy_targets_with_offset.squeeze(2) * non_target_copy_mask.long() * source_copy_mask.long()
        targets = targets * non_pad_mask.long()

        pred_eq = predictions.eq(targets).mul(non_pad_mask)

        num_non_pad = non_pad_mask.sum().item()
        num_correct_pred = pred_eq.sum().item()

        num_target_copy = target_copy_mask.mul(non_pad_mask).sum().item()
        num_correct_target_copy = pred_eq.mul(target_copy_mask).sum().item()
        num_correct_target_point = predictions.ge(self.vocab_size + source_dynamic_vocab_size).\
            mul(target_copy_mask).mul(non_pad_mask).sum().item()

        num_source_copy = source_copy_mask.mul(non_target_copy_mask).mul(non_pad_mask).sum().item()
        num_correct_source_copy = pred_eq.mul(non_target_copy_mask).mul(source_copy_mask).sum().item()
        num_correct_source_point = predictions.ge(self.vocab_size).mul(predictions.lt(self.vocab_size + source_dynamic_vocab_size)).\
            mul(non_target_copy_mask).mul(source_copy_mask).mul(non_pad_mask).sum().item()

        self.metrics(loss.sum().item(), num_non_pad, num_correct_pred,
                     num_source_copy, num_correct_source_copy, num_correct_source_point,
                     num_target_copy, num_correct_target_copy, num_correct_target_point
                     )

        return dict(
            loss=loss.sum().div(float(num_tokens)),
            total_loss=loss.sum(),
            num_tokens=torch.tensor([float(num_tokens)]).type_as(loss),
            predictions=predictions
        )


class SimplePointerGenerator(torch.nn.Module):

    def __init__(self, input_size, switch_input_size, vocab_size, vocab_pad_idx, force_copy):
        super(SimplePointerGenerator, self).__init__()
        self.linear = torch.nn.Linear(input_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.linear_pointer = torch.nn.Linear(switch_input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.metrics = Seq2SeqMetrics()

        self.vocab_size = vocab_size
        self.vocab_pad_idx = vocab_pad_idx

        self.force_copy = force_copy

        self.eps = 1e-20

    def forward(self, hiddens, source_attentions, source_attention_maps):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying target nodes.

        :param hiddens: decoder outputs, [batch_size, num_target_nodes, hidden_size]
        :param source_attentions: attention of each source node,
            [batch_size, num_target_nodes, num_source_nodes]
        :param source_attention_maps: a sparse indicator matrix
            mapping each source node to its index in the dynamic vocabulary.
            [batch_size, num_source_nodes, dynamic_vocab_size]
        """
        batch_size, num_target_nodes, _ = hiddens.size()

        # Pointer probability.
        p_copy = self.sigmoid(self.linear_pointer(hiddens))
        p_generate = 1 - p_copy

        # Probability distribution over the vocabulary.
        # [batch_size, num_target_nodes, vocab_size]
        scores = self.linear(hiddens)
        scores[:, :, self.vocab_pad_idx] = -float('inf')
        # [batch_size, num_target_nodes, vocab_size]
        vocab_probs = self.softmax(scores)
        scaled_vocab_probs = torch.mul(vocab_probs, p_generate.expand_as(vocab_probs))

        # [batch_size, num_target_nodes, num_source_nodes]
        scaled_source_attentions = torch.mul(source_attentions, p_copy.expand_as(source_attentions))
        # [batch_size, num_target_nodes, dynamic_vocab_size]
        scaled_copy_source_probs = torch.bmm(scaled_source_attentions, source_attention_maps.float())

        # [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        probs = torch.cat([
            scaled_vocab_probs.contiguous(),
            scaled_copy_source_probs.contiguous(),
        ], dim=2)

        _, predictions = probs.max(2)

        return dict(
            probs=probs,
            predictions=predictions,
        )

    def compute_loss(self, probs, predictions, generate_targets, source_copy_targets, coverage_records, copy_attentions):
        """
        Priority: source_copy > generate

        :param probs: probability distribution,
            [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        :param predictions: [batch_size, num_target_nodes]
        :param generate_targets: target node index in the vocabulary,
            [batch_size, num_target_nodes]
        :param source_copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        :param coverage_records: None or a tensor recording source-side coverages.
            [batch_size, num_target_nodes, num_source_nodes]
        :param copy_attentions: [batch_size, num_target_nodes, num_source_nodes]
        """

        non_pad_mask = generate_targets.ne(self.vocab_pad_idx)
        source_copy_mask = source_copy_targets.ne(1) & source_copy_targets.ne(0)  # 1 is the index for unknown
        non_source_copy_mask = 1 - source_copy_mask

        # [batch_size, num_target_nodes, 1]
        source_copy_targets_with_offset = source_copy_targets.unsqueeze(2) + self.vocab_size
        # [batch_size, num_target_nodes]
        source_copy_target_probs = probs.gather(dim=2, index=source_copy_targets_with_offset).squeeze(2)
        source_copy_target_probs = source_copy_target_probs.mul(source_copy_mask.float())

        # [batch_size, num_target_nodes]
        generate_target_probs = probs.gather(dim=2, index=generate_targets.unsqueeze(2)).squeeze(2)

        # Except copy-oov nodes, all other nodes should be copied.
        likelihood = source_copy_target_probs + generate_target_probs.mul(non_source_copy_mask.float())
        num_tokens = non_pad_mask.sum().item()

        if not self.force_copy:
            non_generate_oov_mask = generate_targets.ne(1)
            additional_generate_mask = (source_copy_mask & non_generate_oov_mask)
            likelihood = likelihood + generate_target_probs.mul(additional_generate_mask.float())
            num_tokens += additional_generate_mask.sum().item()

        # Add eps for numerical stability.
        likelihood = likelihood + self.eps

        coverage_loss = 0
        if coverage_records is not None:
            coverage_loss = torch.sum(
                torch.min(coverage_records, copy_attentions), 2).mul(non_pad_mask.float())

        # Drop pads.
        loss = -likelihood.log().mul(non_pad_mask.float()) + coverage_loss

        # Mask out copy targets for which copy does not happen.
        targets = source_copy_targets_with_offset.squeeze(2) * source_copy_mask.long() + \
                  generate_targets * non_source_copy_mask.long()
        targets = targets * non_pad_mask.long()

        pred_eq = predictions.eq(targets).mul(non_pad_mask)

        num_non_pad = non_pad_mask.sum().item()
        num_correct_pred = pred_eq.sum().item()

        num_source_copy = source_copy_mask.mul(non_pad_mask).sum().item()
        num_correct_source_copy = pred_eq.mul(source_copy_mask).sum().item()
        num_correct_source_point = predictions.ge(self.vocab_size).mul(source_copy_mask & non_pad_mask).sum().item()

        self.metrics(loss.sum().item(), num_non_pad, num_correct_pred,
                     num_source_copy, num_correct_source_copy, num_correct_source_point)

        return dict(
            loss=loss.sum().div(float(num_tokens)),
            total_loss=loss.sum(),
            num_tokens=float(num_tokens),
            predictions=predictions
        )