local data_dir = "data/AMR/amr_2.0/";
local glove_embeddings = "/export/ssd/sheng/data/glove/glove.840B.300d.zip";

{
  dataset_reader: {
    type: "amr",
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
  },
  train_data_path: data_dir + "dev_amr.txt.features.preproc",
  validation_data_path: data_dir + "dev_amr.txt.features.preproc",
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
      source_tokens: 18000,
      target_tokens: 12200,
      generation_tokens: 12200,
    },
  },

  model: {
    type: "transductive_parser",
    bert_encoder: null,
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
          # pretrained_file: glove_embeddings,
          embedding_dim: 200,
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
          dropout: 0.0,
        },
      },
    },
    encoder_pos_embedding: {
      vocab_namespace: "pos_tags",
      embedding_dim: 50,
    },
    encoder_anonymization_embedding: {
      vocab_namespace: "anonymization_tags",
      # num_embeddings: 2,
      embedding_dim: 30,
    },
    encoder: {
      type: "miso_stacked_bilstm",
      batch_first: true,
      stateful: true,
      input_size: 200 + 50 + 50 + 30,
      hidden_size: 256,
      num_layers: 2,
      recurrent_dropout_probability: 0.0,
      use_highway: false,
    },
    decoder_token_embedder: {
      token_embedders: {
        target_tokens: {
          type: "embedding",
          vocab_namespace: "target_tokens",
          # pretrained_file: glove_embeddings,
          embedding_dim: 200,
          trainable: true,
        },
        target_token_characters: {
          type: "character_encoding",
          embedding: {
            vocab_namespace: "target_token_characters",
            embedding_dim: 16,
          },
          encoder: {
            type: "cnn",
            embedding_dim: 16,
            num_filters: 50,
            ngram_filter_sizes: [3],
          },
          dropout: 0.0,
        },
      },
    },
    decoder_node_index_embedding: {
      # vocab_namespace: "node_indices",
      num_embeddings: 200,
      embedding_dim: 50,
    },
    decoder_pos_embedding: {
      vocab_namespace: "pos_tags",
      embedding_dim: 50,
    },
    decoder: {
      rnn_cell: {
        input_size: 200 + 50 + 50 + 50 + 512,
        hidden_size: 512,
        num_layers: 2,
        recurrent_dropout_probability: 0.0,
        use_highway: false,
      },
      source_attention_layer: {
        type: "global",
        query_vector_dim: 512,
        key_vector_dim: 512,
        output_vector_dim: 512,
        attention: {
          type: "mlp",
          # TODO: try to use smaller dims.
          query_vector_dim: 512,
          key_vector_dim: 512,
          hidden_vector_dim: 128,
          use_coverage: false,
        },
      },
      target_attention_layer: {
        type: "global",
        query_vector_dim: 512,
        key_vector_dim: 512,
        output_vector_dim: 512,
        attention: {
          type: "mlp",
          query_vector_dim: 512,
          key_vector_dim: 512,
          hidden_vector_dim: 128,
          use_coverage: false,
        },
      },
      dropout: 0.0,
    },
    extended_pointer_generator: {
      input_vector_dim: 512,
      source_copy: true,
      target_copy: true,
    },
    tree_parser: {
      query_vector_dim: 512,
      key_vector_dim: 512,
      edge_head_vector_dim: 128,
      edge_type_vector_dim: 64,
      attention: {
        type: "biaffine",
        query_vector_dim: 128,
        key_vector_dim: 128,
      },
    },
    label_smoothing: {
        smoothing: 0.0,
    },
    dropout: 0.0,
    beam_size: 1,
    max_decoding_steps: 50,
    target_output_namespace: "generation_tokens",
    pos_tag_namespace: "pos_tags",
    edge_type_namespace: "edge_types",
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
    batch_size: 64,
  },

  trainer: {
    type: "amr_parsing",
    num_epochs: 120,
    patience: 20,
    grad_norm: 5.0,
    # TODO: try to use grad clipping.
    grad_clipping: null,
    cuda_device: 0,
    num_serialized_models_to_keep: 5,
    validation_metric: "-loss",
    optimizer: {
      type: "adam",
      weight_decay: 3e-9,
      amsgrad: true,
    },
    learning_rate_scheduler: {
      type: "reduce_on_plateau",
      patience: 10,
    },
    no_grad: [],
    evaluation_script_path: "scripts/amr_parsing_evaluation.sh",
    smatch_tool_path: "smatch_tool",
    validation_data_path: data_dir + "dev_amr.txt.features.preproc",
    validation_prediction_path: "amr_validation.txt",
  },
  random_seed: 1,
  numpy_seed: 1,
  pytorch_seed: 1,
}
