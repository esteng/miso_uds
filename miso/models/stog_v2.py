import torch

from miso.models.model import Model
from miso.utils.logging import init_logger
from miso.modules.token_embedders.embedding import Embedding
from miso.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from miso.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from miso.modules.stacked_bilstm import StackedBidirectionalLstm
from miso.modules.stacked_lstm import StackedLstm
from miso.modules.decoders.rnn_decoder import InputFeedRNNDecoder
from miso.modules.attention_layers.global_attention import GlobalAttention
from miso.modules.attention import DotProductAttention
from miso.modules.attention import MLPAttention
from miso.modules.input_variational_dropout import InputVariationalDropout
from miso.modules.decoders.generator import Generator
from miso.modules.decoders.pointer_generator import SimplePointerGenerator
from miso.modules.decoders.deep_biaffine_graph_decoder import DeepBiaffineGraphDecoder
from miso.modules.decoders.coref_scorer import CorefScorer
from miso.utils.nn import get_text_field_mask
from miso.utils.string import START_SYMBOL, END_SYMBOL
from miso.data.vocabulary import DEFAULT_OOV_TOKEN
from miso.data.tokenizers.character_tokenizer import CharacterTokenizer
# The following imports are added for mimick testing.
from miso.data.dataset_builder import load_dataset_reader
from miso.predictors.predictor import Predictor
from miso.commands.predict import _PredictManager
import subprocess


logger = init_logger()


def character_tensor_from_token_tensor(
        token_tensor,
        vocab,
        character_tokenizer,
        namespace=dict(tokens="decoder_token_ids", characters="decoder_token_characters")
):
    token_str = [vocab.get_token_from_index(i, namespace["tokens"]) for i in token_tensor.view(-1).tolist()]
    max_char_len = max([len(token) for token in token_str])
    indices = []
    for token in token_str:
        token_indices = [vocab.get_token_index(vocab._padding_token) for _ in range(max_char_len)]
        for char_i, character in enumerate(character_tokenizer.tokenize(token)):
            index = vocab.get_token_index(character.text, namespace["characters"])
            token_indices[char_i] = index
        indices.append(token_indices)

    return torch.tensor(indices).view(token_tensor.size(0), 1, -1).type_as(token_tensor)


class STOGv2(Model):

    def __init__(self,
                 vocab,
                 use_char_cnn,
                 use_coverage,
                 max_decode_length,
                 # Encoder
                 encoder_token_embedding,
                 encoder_char_embedding,
                 encoder_char_cnn,
                 encoder_embedding_dropout,
                 encoder,
                 encoder_output_dropout,
                 # Decoder
                 decoder_token_embedding,
                 decoder_char_embedding,
                 decoder_char_cnn,
                 decoder_embedding_dropout,
                 decoder,
                 # Generator
                 generator,
                 # Coref
                 coref_scorer,
                 # Graph decoder
                 graph_decoder,
                 test_config
                 ):
        super(STOGv2, self).__init__()

        self.vocab = vocab
        self.use_char_cnn = use_char_cnn
        self.use_coverage = use_coverage
        self.max_decode_length = max_decode_length

        self.encoder_token_embedding = encoder_token_embedding
        self.encoder_char_embedding = encoder_char_embedding
        self.encoder_char_cnn = encoder_char_cnn
        self.encoder_embedding_dropout = encoder_embedding_dropout
        self.encoder = encoder
        self.encoder_output_dropout = encoder_output_dropout

        self.decoder_token_embedding = decoder_token_embedding
        self.decoder_char_embedding = decoder_char_embedding
        self.decoder_char_cnn = decoder_char_cnn
        self.decoder_embedding_dropout = decoder_embedding_dropout
        self.decoder = decoder

        self.generator = generator

        self.coref_scorer = coref_scorer

        self.graph_decoder = graph_decoder

        self.beam_size = 1

        self.test_config = test_config

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def set_decoder_token_indexers(self, token_indexers):
        self.decoder_token_indexers = token_indexers
        self.character_tokenizer = CharacterTokenizer()

    def get_metrics(self, reset: bool = False, mimick_test: bool = False):
        metrics = dict()
        if mimick_test and self.test_config:
            metrics = self.mimick_test()
        generator_metrics = self.generator.metrics.get_metric(reset)
        coref_scorer_metrics = self.coref_scorer.metrics.get_metric(reset)
        graph_decoder_metrics = self.graph_decoder.metrics.get_metric(reset)
        metrics.update(dict(
            all_acc=generator_metrics['all_acc'],
            src_acc=generator_metrics['src_acc'],
            tgt_acc=coref_scorer_metrics['cacc1'],
            ppl=generator_metrics['ppl'],
            UAS=graph_decoder_metrics['UAS'],
            LAS=graph_decoder_metrics['LAS'],
        ))
        if 'F1' not in metrics:
            metrics['F1'] = metrics['all_acc']
        return metrics

    def mimick_test(self):
        dataset_reader = load_dataset_reader('AMR')
        predictor = Predictor.by_name('STOG')(self, dataset_reader)
        manager = _PredictManager(
            predictor,
            self.test_config['data'],
            self.test_config['prediction'],
            self.test_config['batch_size'],
            False,
            True,
            1
        )
        try:
            logger.info('Mimicking test...')
            manager.run()
        except:
            logger.info('Exception threw out when running the manager.')
            return {}
        try:
            logger.info('Computing the Smatch score...')
            result = subprocess.check_output([
                self.test_config['eval_script'],
                self.test_config['smatch_dir'],
                self.test_config['data'],
                self.test_config['prediction']
            ]).decode().split()
            result = list(map(float, result))
            return dict(PREC=result[0]*100, REC=result[1]*100, F1=result[2]*100)
        except:
            logger.info('Exception threw out when computing smatch.')
            return {}

    def print_batch_details(self, batch, batch_idx):
        print(batch["amr"][batch_idx])
        print()

        print("Source tokens:")
        print([(i, x) for i, x in enumerate(batch["src_tokens_str"][batch_idx])])
        print()

        print('Source copy vocab')
        print(batch["src_copy_vocab"][batch_idx])
        print()

        print('Source map')
        print(batch["src_copy_map"][batch_idx].int())
        print()

        print("Target tokens")
        print([(i, x) for i, x in enumerate(batch["tgt_tokens_str"][batch_idx])])
        print()

        print('Source copy indices')
        print([(i, x) for i, x in enumerate(batch["src_copy_indices"][batch_idx].tolist())])

        print('Target copy indices')
        print([(i, x) for i, x in enumerate(batch["tgt_copy_indices"][batch_idx].tolist())])

    def prepare_batch_input(self, batch):
        # [batch, num_tokens]
        encoder_token_inputs = batch['src_tokens']['encoder_tokens']
        # [batch, num_tokens, num_chars]
        encoder_char_inputs = batch['src_tokens']['encoder_characters']
        # [batch, num_tokens]
        encoder_mask = get_text_field_mask(batch['src_tokens'])

        encoder_inputs = dict(
            token=encoder_token_inputs,
            char=encoder_char_inputs,
            mask=encoder_mask
        )

        # [batch, num_tokens]
        decoder_token_inputs = batch['tgt_tokens']['decoder_tokens'][:, :-1].contiguous()
        # [batch, num_tokens, num_chars]
        decoder_char_inputs = batch['tgt_tokens']['decoder_characters'][:, :-1].contiguous()
        # TODO: The following change can be done in amr.py.
        # Initially, raw_coref_inputs has value like [0, 0, 0, 1, 0]
        # where '0' indicates that the input token has no precedent, and
        # '1' indicates that the input token's first precedent is at position '1'.
        # Here, we change it to [0, 1, 2, 1, 4] which means if the input token
        # has no precedent, then it is referred to itself.
        raw_coref_inputs = batch["tgt_copy_indices"][:, :-1].contiguous()
        coref_happen_mask = raw_coref_inputs.ne(0)
        decoder_coref_inputs = torch.ones_like(raw_coref_inputs) * torch.arange(
            0, raw_coref_inputs.size(1)).type_as(raw_coref_inputs).unsqueeze(0)
        decoder_coref_inputs.masked_fill_(coref_happen_mask, 0)
        # [batch, num_tokens]
        decoder_coref_inputs = decoder_coref_inputs + raw_coref_inputs

        decoder_inputs = dict(
            token=decoder_token_inputs,
            char=decoder_char_inputs,
            coref=decoder_coref_inputs
        )

        # [batch, num_tokens]
        vocab_targets = batch['tgt_tokens']['decoder_tokens'][:, 1:].contiguous()
        # [batch, num_tgt_tokens, num_src_tokens + unk]
        copy_targets = batch["src_copy_indices"][:, 1:]
        # [batch, num_src_tokens + unk, src_dynamic_vocab_size]
        # Exclude the last pad.
        copy_attention_maps = batch['src_copy_map'][:, 1:-1]

        generator_inputs = dict(
            vocab_targets=vocab_targets,
            copy_targets=copy_targets,
            copy_attention_maps=copy_attention_maps
        )

        # [batch, num_tokens]
        coref_inputs = dict(
            targets=batch['tgt_copy_indices'][:, 1:-1],
            mask=batch['tgt_copy_mask'][:, 1:-1]
        )

        # Remove the last two pads so that they have the same size of other inputs?
        edge_heads = batch['head_indices'][:, :-2]
        edge_labels = batch['head_tags'][:, :-2]
        # TODO: The following computation can be done in amr.py.
        # Get the parser mask.
        parser_token_inputs = torch.zeros_like(decoder_token_inputs)
        parser_token_inputs.copy_(decoder_token_inputs)
        parser_token_inputs[
            parser_token_inputs == self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')
        ] = 0
        parser_mask = (parser_token_inputs != 0).float()

        parser_inputs = dict(
            edge_heads=edge_heads,
            edge_labels=edge_labels,
            corefs=None,
            mask=parser_mask
        )

        return encoder_inputs, decoder_inputs, generator_inputs, coref_inputs, parser_inputs

    def forward(self, batch, for_training=False):
        encoder_inputs, decoder_inputs, generator_inputs, coref_inputs, parser_inputs = self.prepare_batch_input(batch)

        encoder_outputs = self.encode(
            encoder_inputs['token'],
            encoder_inputs['char'],
            encoder_inputs['mask']
        )

        if for_training:
            decoder_outputs = self.decode_for_training(
                decoder_inputs['token'],
                decoder_inputs['char'],
                decoder_inputs['coref'],
                encoder_outputs['memory_bank'],
                encoder_inputs['mask'],
                encoder_outputs['final_states']
            )

            generator_output = self.generator(
                decoder_outputs['memory_bank'],
                decoder_outputs['copy_attentions'],
                generator_inputs['copy_attention_maps'],
            )

            generator_loss_output = self.generator.compute_loss(
                generator_output['probs'],
                generator_output['predictions'],
                generator_inputs['vocab_targets'],
                generator_inputs['copy_targets'],
                decoder_outputs['coverage_records'],
                decoder_outputs['copy_attentions']
            )

            coref_output = self.coref_scorer(
                decoder_outputs['rnn_memory_bank'][:, 1:]
            )

            coref_loss_output = self.coref_scorer.compute_loss(
                coref_output['probs'],
                coref_output['predictions'],
                coref_inputs['targets'],
                coref_inputs['mask']
            )

            graph_decoder_outputs = self.graph_decode(
                decoder_outputs['rnn_memory_bank'],
                parser_inputs['edge_heads'],
                parser_inputs['edge_labels'],
                parser_inputs['corefs'],
                parser_inputs['mask'],
            )

            return dict(
                loss=generator_loss_output['loss'] + coref_loss_output['loss'] + graph_decoder_outputs['loss'],
            )

        else:
            return dict(
                encoder_memory_bank=encoder_outputs['memory_bank'],
                encoder_mask=encoder_inputs['mask'],
                encoder_final_states=encoder_outputs['final_states'],
                copy_attention_maps=generator_inputs['copy_attention_maps'],
                copy_vocabs=batch['src_copy_vocab']
            )

    def encode(self, tokens, chars, mask):
        # [batch, num_tokens, embedding_size]
        token_embeddings = self.encoder_token_embedding(tokens)
        if self.use_char_cnn:
            char_cnn_output = self._get_encoder_char_cnn_output(chars)
            encoder_inputs = torch.cat([token_embeddings, char_cnn_output], 2)
        else:
            encoder_inputs = token_embeddings

        encoder_inputs = self.encoder_embedding_dropout(encoder_inputs)

        # [batch, num_tokens, encoder_output_size]
        encoder_outputs = self.encoder(encoder_inputs, mask)
        encoder_outputs = self.encoder_output_dropout(encoder_outputs)

        # A tuple of (state, memory) with shape [num_layers, batch, encoder_output_size]
        encoder_final_states = self.encoder._states
        self.encoder.reset_states()

        return dict(
            memory_bank=encoder_outputs,
            final_states=encoder_final_states
        )

    def decode_for_training(self, tokens, chars, corefs, memory_bank, mask, states):
        # [batch, num_tokens, embedding_size]
        token_embeddings = self.decoder_token_embedding(tokens)
        if self.use_char_cnn:
            char_cnn_output = self._get_decoder_char_cnn_output(chars)
            decoder_inputs = torch.cat([token_embeddings, char_cnn_output], 2)
        else:
            decoder_inputs = token_embeddings

        decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

        decoder_outputs = self.decoder(decoder_inputs, memory_bank, mask, states)

        return dict(
            memory_bank=decoder_outputs['output_sequences'],
            rnn_memory_bank=decoder_outputs['rnn_output_sequences'],
            coref_inputs=decoder_outputs['coref_inputs'],
            coref_attentions=decoder_outputs['coref_attentions'],
            copy_attentions=decoder_outputs['copy_attentions'],
            source_attentions=decoder_outputs['std_attentions'],
            coverage_records=decoder_outputs['coverage_records']
        )

    def graph_decode(self, memory_bank, edge_heads, edge_labels, corefs, mask):
        # Exclude the BOS symbol.
        memory_bank = memory_bank[:, 1:]
        mask = mask[:, 1:]
        return self.graph_decoder(memory_bank, edge_heads, edge_labels, corefs, mask)

    def _get_encoder_char_cnn_output(self, chars):
        # [batch, num_tokens, num_chars, embedding_size]
        char_embeddings = self.encoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.encoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def _get_decoder_char_cnn_output(self, chars):
        # [batch, num_tokens, num_chars, embedding_size]
        char_embeddings = self.decoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.decoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def decode(self, input_dict):
        memory_bank = input_dict['encoder_memory_bank']
        mask = input_dict['encoder_mask']
        states = input_dict['encoder_final_states']
        copy_attention_maps = input_dict['copy_attention_maps']
        copy_vocabs = input_dict['copy_vocabs']

        if self.beam_size == 1:
            generator_outputs = self.decode_with_pointer_generator(
                memory_bank, mask, states, copy_attention_maps, copy_vocabs
            )
            coref_indexes = self.decode_with_coref_scorer(
                generator_outputs['decoder_rnn_memory_bank'],
                generator_outputs['predictions']
            )
            parser_outputs = self.decode_with_graph_parser(
                generator_outputs['decoder_rnn_memory_bank'],
                coref_indexes,
                generator_outputs['decoder_mask']
            )
            return dict(
                nodes=generator_outputs['predictions'],
                heads=parser_outputs['edge_heads'],
                head_labels=parser_outputs['edge_labels'],
                corefs=coref_indexes
            )
        else:
            raise NotImplementedError

    def decode_with_pointer_generator(self, memory_bank, mask, states, copy_attention_maps, copy_vocabs):
        # [batch_size, 1]
        batch_size = memory_bank.size(0)
        tokens = torch.ones(batch_size, 1) * self.vocab.get_token_index(START_SYMBOL, "decoder_token_ids")
        tokens = tokens.type_as(mask).long()

        decoder_outputs = []
        rnn_outputs = []
        copy_attentions = []
        predictions = []
        decoder_mask = []

        input_feed = None
        coref_inputs = None

        coverage = None
        if self.use_coverage:
            coverage = memory_bank.new_zeros(batch_size, 1, memory_bank.size(1))

        for step_i in range(self.max_decode_length):
            # 1. Get the decoder inputs.
            token_embeddings = self.decoder_token_embedding(tokens)
            if self.use_char_cnn:
                # TODO: get chars from tokens.
                # [batch_size, 1, num_chars]
                chars = character_tensor_from_token_tensor(
                    tokens,
                    self.vocab,
                    self.character_tokenizer
                )

                char_cnn_output = self._get_decoder_char_cnn_output(chars)
                decoder_inputs = torch.cat([token_embeddings, char_cnn_output], 2)
            else:
                decoder_inputs = token_embeddings
            decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

            # 2. Decode one step.
            decoder_output_dict = self.decoder(
                decoder_inputs, memory_bank, mask, states, input_feed, coref_inputs, coverage)
            _decoder_outputs = decoder_output_dict['output_sequences']
            _rnn_outputs = decoder_output_dict['rnn_output_sequences']
            _copy_attentions = decoder_output_dict['copy_attentions']
            states = decoder_output_dict['hidden_state']
            input_feed = decoder_output_dict['input_feed']
            coverage = decoder_output_dict['coverage']

            # 3. Run pointer/generator.
            generator_output = self.generator(_decoder_outputs, _copy_attentions, copy_attention_maps)

            # 4. Update maps and get the next token input.
            tokens, _predictions, _mask = self._update_maps_and_get_next_input(
                generator_output['predictions'].squeeze(1), copy_vocabs, decoder_mask)

            # 5. Update variables.
            decoder_outputs += [_decoder_outputs]
            rnn_outputs += [_rnn_outputs]

            copy_attentions += [_copy_attentions]

            predictions += [_predictions]
            # Add the mask for the next input.
            decoder_mask += [_mask]

        # 6. Do the following chunking for the graph decoding input.
        # Exclude the hidden state for BOS.
        decoder_outputs = torch.cat(decoder_outputs[1:], dim=1)
        rnn_outputs = torch.cat(rnn_outputs[1:], dim=1)
        # Exclude coref/mask for EOS.
        # TODO: Answer "What if the last one is not EOS?"
        predictions = torch.cat(predictions[:-1], dim=1)
        decoder_mask = 1 - torch.cat(decoder_mask[:-1], dim=1)

        return dict(
            # [batch_size, max_decode_length]
            predictions=predictions,
            decoder_mask=decoder_mask,
            # [batch_size, max_decode_length, hidden_size]
            decoder_memory_bank=decoder_outputs,
            decoder_rnn_memory_bank=rnn_outputs,
            # [batch_size, max_decode_length, encoder_length]
            copy_attentions=copy_attentions,
        )

    def _update_maps_and_get_next_input(self, predictions, copy_vocabs, masks):
        """Dynamically update/build the maps needed for copying.

        :param predictions: [batch_size]
        :param copy_vocabs: a list of dynamic vocabs.
        :param masks: a list of [batch_size] tensors indicating whether EOS has been generated.
            if EOS has has been generated, then the mask is `1`.
        :return:
        """
        vocab_size = self.generator.vocab_size
        gen_mask = predictions.lt(vocab_size)
        copy_mask = predictions.ge(vocab_size)

        # 1. Compute the next input.
        # If a token is copied from the source side, we look up its index in the gen vocab.
        copy_predictions = (predictions - vocab_size) * copy_mask.long()
        for i, index in enumerate(copy_predictions.tolist()):
            copy_predictions[i] = self.vocab.get_token_index(
                copy_vocabs[i].get_token_from_idx(index), 'decoder_token_ids')

        next_input = copy_predictions * copy_mask.long() + \
                     predictions * gen_mask.long()

        # 2. Get the mask for the current generation.
        has_eos = torch.zeros_like(gen_mask)
        if len(masks) != 0:
            has_eos = torch.cat(masks, 1).long().sum(1).gt(0)
        mask = next_input.eq(self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')) | has_eos

        return next_input.unsqueeze(1), predictions.unsqueeze(1), mask.unsqueeze(1)

    def decode_with_coref_scorer(self, hiddens, tokens):
        # [batch_size, num_tokens]
        predictions = self.coref_scorer(hiddens)['predictions']
        non_sentinel_mask = predictions.ne(0)
        # Offset by 1
        coref_token_indexes = (predictions - 1).mul(non_sentinel_mask.long())

        batch_size, num_tokens = predictions.size()
        coref_candidates = tokens.view(batch_size, 1, num_tokens).expand(batch_size, num_tokens, num_tokens)
        coref_token_ids = coref_candidates.gather(dim=2, index=coref_token_indexes.unsqueeze(2)).squeeze(2)
        valid_coref_mask = coref_token_ids.eq(tokens) & non_sentinel_mask
        predictions = predictions.mul(valid_coref_mask.long())
        defaults = torch.arange(1, num_tokens + 1).type_as(predictions).unsqueeze(0).expand_as(predictions)
        predictions = predictions + defaults.mul((1 - valid_coref_mask).long())
        return predictions

    def decode_with_graph_parser(self, memory_bank, corefs, mask):
        """Predict edges and edge labels between nodes.
        :param memory_bank: [batch_size, node_length, hidden_size]
        :param corefs: [batch_size, node_length]
        :param mask:  [batch_size, node_length]
        :return a dict of edge_heads and edge_labels.
            edge_heads: [batch_size, node_length]
            edge_labels: [batch_size, node_length]
        """
        memory_bank, _, _, corefs, mask = self.graph_decoder._add_head_sentinel(
            memory_bank, None, None, corefs, mask)
        (edge_node_h, edge_node_m), (edge_label_h, edge_label_m) = self.graph_decoder.encode(memory_bank)
        edge_node_scores = self.graph_decoder._get_edge_node_scores(edge_node_h, edge_node_m, mask.float())
        edge_heads, edge_labels = self.graph_decoder.mst_decode(
            edge_label_h, edge_label_m, edge_node_scores, corefs, mask)
        return dict(
            edge_heads=edge_heads,
            edge_labels=edge_labels
        )

    @classmethod
    def from_params(cls, vocab, params):
        logger.info('Building the STOG Model...')

        # Encoder
        encoder_token_embedding = Embedding.from_params(vocab, params['encoder_token_embedding'])
        if params['use_char_cnn']:
            encoder_char_embedding = Embedding.from_params(vocab, params['encoder_char_embedding'])
            encoder_char_cnn = CnnEncoder(
                embedding_dim=params['encoder_char_cnn']['embedding_dim'],
                num_filters=params['encoder_char_cnn']['num_filters'],
                ngram_filter_sizes=params['encoder_char_cnn']['ngram_filter_sizes'],
                conv_layer_activation=torch.tanh
            )
        else:
            encoder_char_embedding = None
            encoder_char_cnn = None

        encoder_embedding_dropout = InputVariationalDropout(p=params['encoder_token_embedding']['dropout'])

        encoder = PytorchSeq2SeqWrapper(
            module=StackedBidirectionalLstm.from_params(params['encoder']),
            stateful=True
        )
        encoder_output_dropout = InputVariationalDropout(p=params['encoder']['dropout'])

        # Decoder
        decoder_token_embedding = Embedding.from_params(vocab, params['decoder_token_embedding'])
        if params['use_char_cnn']:
            decoder_char_embedding = Embedding.from_params(vocab, params['decoder_char_embedding'])
            decoder_char_cnn = CnnEncoder(
                embedding_dim=params['decoder_char_cnn']['embedding_dim'],
                num_filters=params['decoder_char_cnn']['num_filters'],
                ngram_filter_sizes=params['decoder_char_cnn']['ngram_filter_sizes'],
                conv_layer_activation=torch.tanh
            )
        else:
            decoder_char_embedding = None
            decoder_char_cnn = None

        decoder_embedding_dropout = InputVariationalDropout(p=params['decoder_token_embedding']['dropout'])

        # Source attention
        if params['source_attention']['attention_function'] == 'mlp':
            source_attention = MLPAttention(
                decoder_hidden_size=params['decoder']['hidden_size'],
                encoder_hidden_size=params['encoder']['hidden_size'] * 2,
                attention_hidden_size=params['decoder']['hidden_size'],
                coverage=params['source_attention'].get('coverage', False)
            )
        else:
            source_attention = DotProductAttention(
                decoder_hidden_size=params['decoder']['hidden_size'],
                encoder_hidden_size=params['encoder']['hidden_size'] * 2,
                share_linear=params['pointer_attention'].get('share_linear', False)
            )

        source_attention_layer = GlobalAttention(
            decoder_hidden_size=params['decoder']['hidden_size'],
            encoder_hidden_size=params['encoder']['hidden_size'] * 2,
            attention=source_attention
        )

        decoder = InputFeedRNNDecoder(
            rnn_cell=StackedLstm.from_params(params['decoder']),
            attention_layer=source_attention_layer,
            # TODO: modify the dropout so that the dropout mask is unchanged across the steps.
            dropout=InputVariationalDropout(p=params['decoder']['dropout']),
            use_coverage=params['use_coverage']
        )

        switch_input_size = params['encoder']['hidden_size'] * 2
        generator = SimplePointerGenerator(
            input_size=params['decoder']['hidden_size'],
            switch_input_size=switch_input_size,
            vocab_size=vocab.get_vocab_size('decoder_token_ids'),
            force_copy=params['generator'].get('force_copy', True),
            # TODO: Set the following indices.
            vocab_pad_idx=0
        )

        # Coref attention
        coref_attention = MLPAttention(
            decoder_hidden_size=params['decoder']['hidden_size'],
            encoder_hidden_size=params['decoder']['hidden_size'],
            attention_hidden_size=params['decoder']['hidden_size'],
        )
        coref_sentinel = torch.nn.Parameter(torch.randn([1, 1, params['decoder']['hidden_size']]))
        coref_scorer = CorefScorer(coref_sentinel, coref_attention)

        graph_decoder = DeepBiaffineGraphDecoder.from_params(vocab, params['graph_decoder'])

        logger.info('encoder_token: %d' % vocab.get_vocab_size('encoder_token_ids'))
        logger.info('encoder_chars: %d' % vocab.get_vocab_size('encoder_token_characters'))
        logger.info('decoder_token: %d' % vocab.get_vocab_size('decoder_token_ids'))
        logger.info('decoder_chars: %d' % vocab.get_vocab_size('decoder_token_characters'))

        return cls(
            vocab=vocab,
            use_char_cnn=params['use_char_cnn'],
            use_coverage=params['use_coverage'],
            max_decode_length=params.get('max_decode_length', 50),
            encoder_token_embedding=encoder_token_embedding,
            encoder_char_embedding=encoder_char_embedding,
            encoder_char_cnn=encoder_char_cnn,
            encoder_embedding_dropout=encoder_embedding_dropout,
            encoder=encoder,
            encoder_output_dropout=encoder_output_dropout,
            decoder_token_embedding=decoder_token_embedding,
            decoder_char_cnn=decoder_char_cnn,
            decoder_char_embedding=decoder_char_embedding,
            decoder_embedding_dropout=decoder_embedding_dropout,
            decoder=decoder,
            generator=generator,
            coref_scorer=coref_scorer,
            graph_decoder=graph_decoder,
            test_config=params.get('mimick_test', None)
        )

