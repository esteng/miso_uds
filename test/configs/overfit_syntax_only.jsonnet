local data_dir = "dev";
local glove_embeddings = "/exp/estengel/miso/glove.840B.300d.zip";
local synt_method = "encoder-side";

{
  dataset_reader: {
    type: "decomp_syntax_semantics",
    source_token_indexers: {
      source_tokens: {
        type: "single_id",
        namespace: "source_tokens",
      },
      source_token_characters: {
        type: "characters",
        namespace: "source_token_characters",
        min_padding_length: 5,
      },
    },
    target_token_indexers: {
      target_tokens: {
        type: "single_id",
        namespace: "target_tokens",
      },
      target_token_characters: {
        type: "characters",
        namespace: "target_token_characters",
        min_padding_length: 5,
      },
    },
    generation_token_indexers: {
      generation_tokens: {
        type: "single_id",
        namespace: "generation_tokens",
      }
    },
    syntactic_method: synt_method,
    drop_syntax: "true",
    semantics_only: "false",
    line_limit: 2,
    order: "inorder",
    tokenizer: {
                type: "pretrained_transformer_for_amr",
                model_name: "bert-base-cased",
                args: null,
                kwargs: {do_lowercase: 'false'},
                #kwargs: null,
               },
  },
  train_data_path: "dev",
  validation_data_path: "dev",
  #validation_data_path: null,
  test_data_path: null,
  datasets_for_vocab_creation: [
    "train"
  ],

  vocabulary: {
    non_padded_namespaces: [],
    min_count: {
      source_tokens: 1,
      target_tokens: 1,
      generation_tokens: 1,
    },
    max_vocab_size: {
      source_tokens: 1000,
      target_tokens: 1000,
      generation_tokens: 1000,
    },
  },

  model: {
    type: "ud_parser",
    bert_encoder: null,
    #bert_encoder: {
    #                type: "seq2seq_bert_encoder",
    #                config: "bert-base-cased",
    #              },
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
          #pretrained_file: glove_embeddings,
          embedding_dim: 300,
          trainable: true,
        },
        source_token_characters: {
          type: "character_encoding",
          embedding: {
            vocab_namespace: "source_token_characters",
            embedding_dim: 16,
          },
          encoder: {
            type: "cnn",
            embedding_dim: 16,
            num_filters: 50,
            ngram_filter_sizes: [3],
          },
          dropout: 0.00,
        },
      },
    },
    encoder_pos_embedding: {
      vocab_namespace: "pos_tags",
      embedding_dim: 100,
    },
    encoder: {
      type: "miso_stacked_bilstm",
      batch_first: true,
      stateful: true,
      input_size: 300 + 50,
      hidden_size: 64,
      num_layers: 2,
      recurrent_dropout_probability: 0.00,
      use_highway: false,
    },
    biaffine_parser: {
      query_vector_dim: 128,
      key_vector_dim: 128,
      edge_head_vector_dim: 256,
      edge_type_vector_dim: 128,
      num_labels: 16,
      is_syntax: true,
      attention: {
        type: "biaffine",
        query_vector_dim: 256,
        key_vector_dim: 256,
      },
    }, 
    dropout: 0.0,
    syntax_edge_type_namespace: "syn_edge_types",
  },

  iterator: {
    type: "bucket",
    # TODO: try to sort by target tokens.
    sorting_keys: [["source_tokens", "num_tokens"]],
    padding_noise: 0.0,
    batch_size: 64,
  },
  validation_iterator: {
    type: "basic",
    batch_size: 32,
  },

  trainer: {
    type: "decomp_syntax_parsing",
    num_epochs: 250,
    warmup_epochs: 240,
    syntactic_method: synt_method,
    patience: 10000,
    grad_norm: 5.0,
    # TODO: try to use grad clipping.
    grad_clipping: null,
    cuda_device: -1,
    num_serialized_models_to_keep: 5,
    validation_metric: "+syn_las",
    optimizer: {
      type: "adam",
      weight_decay: 3e-9,
      amsgrad: true,
    },
    # learning_rate_scheduler: {
    #   type: "reduce_on_plateau",
    #   patience: 10,
    # },
    no_grad: [],
    # smatch_tool_path: null, # "smatch_tool",
    validation_data_path: "dev",
    #validation_data_path: null,
    validation_prediction_path: "decomp_validation.txt",
    semantics_only: "false",
    drop_syntax: "true",
  },
  random_seed: 12,
  numpy_seed: 12,
  pytorch_seed: 12,
}
