local data_dir = "dev";
local glove_embeddings = "/exp/estengel/miso/glove.840B.300d.zip";

{
  dataset_reader: {
    type: "decomp",
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
    drop_syntax: "true",
    semantics_only: "false",
    line_limit: 16,
    order: "inorder",
    #tokenizer: {
    #            type: "pretrained_transformer_for_amr",
    #            model_name: "bert-base-cased",
    #            args: null,
    #            kwargs: {do_lowercase: 'false'},
    #            #kwargs: null,
    #           },
  },
  train_data_path: data_dir,
  #validation_data_path: "dev",
  validation_data_path: null, 
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
    type: "decomp_transformer_parser",
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
          dropout: 0.01,
        },
      },
    },
    encoder_pos_embedding: {
      vocab_namespace: "pos_tags",
      embedding_dim: 100,
    },
    encoder: {
      type: "stacked_self_attention",
      input_dim: 300 + 50,
      feedforward_hidden_dim: 512,
      num_attention_heads: 2,
      hidden_dim: 64,
      projection_dim: 64, 
      num_layers: 4,
      #recurrent_dropout_probability: 0.33,
      #use_highway: false,
    },
    decoder_token_embedder: {
      token_embedders: {
        target_tokens: {
          type: "embedding",
          vocab_namespace: "target_tokens",
          #pretrained_file: glove_embeddings,
          embedding_dim: 300,
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
          dropout: 0.01,
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
      input_size: 300 + 50 + 50,
      hidden_size: 64,
      num_layers: 4,
      use_coverage: true,
      decoder_layer: {
        type: "pre_norm",
        d_model: 64, 
        n_head: 4, 
        norm: {type: "scale_norm",
               dim: 64},
        dim_feedforward: 128,
        dropout: 0.2, 
        init_scale: 128,
      },
      source_attention_layer: {
        type: "global",
        query_vector_dim: 64,
        key_vector_dim: 64,
        output_vector_dim: 64,
        attention: {
          type: "mlp",
          # TODO: try to use smaller dims.
          query_vector_dim: 64,
          key_vector_dim: 64,
          hidden_vector_dim: 64, 
          use_coverage: true,
        },
      },
      target_attention_layer: {
        type: "global",
        query_vector_dim: 64,
        key_vector_dim: 64,
        output_vector_dim: 64,
        attention: {
          type: "mlp",
          query_vector_dim: 64,
          key_vector_dim: 64,
          hidden_vector_dim: 64,
        },
      },
    },
    extended_pointer_generator: {
      input_vector_dim: 64,
      source_copy: true,
      target_copy: true,
    },
    tree_parser: {
      query_vector_dim: 64,
      key_vector_dim: 64,
      edge_head_vector_dim: 64,
      edge_type_vector_dim: 32,
      attention: {
        type: "biaffine",
        query_vector_dim: 64,
        key_vector_dim: 64,
      },
    },
    node_attribute_module: {
        input_dim: 64,
        hidden_dim: 128,
        output_dim: 44,
        n_layers: 4, 
        loss_multiplier: 10,
    },
    edge_attribute_module: {
        h_input_dim: 32,
        hidden_dim: 200,
        output_dim: 14,
        n_layers: 4, 
        loss_multiplier: 10,
    },
    label_smoothing: {
        smoothing: 0.0,
    },
    dropout: 0.0,
    beam_size: 2,
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
    batch_size: 32,
  },

  trainer: {
    type: "decomp_parsing",
    num_epochs: 250,
    patience: 1000,
    grad_norm: 5.0,
    # TODO: try to use grad clipping.
    grad_clipping: null,
    cuda_device: -1,
    num_serialized_models_to_keep: 5,
    validation_metric: "+s_f1",
    optimizer: {
      type: "adam",
      betas: [0.9, 0.98],
      eps: 1e-9,
      lr: 1e-3, 
      amsgrad: true,
    },
     learning_rate_scheduler: {
       type: "noam",
       model_size: 64, 
       warmup_steps: 1000,
     },
    no_grad: [],
    # smatch_tool_path: null, # "smatch_tool",
    validation_data_path: null,
    #validation_data_path: "dev",
    #validation_prediction_path: "decomp_validation.txt",
    semantics_only: "false",
    drop_syntax: "true",
  },
  random_seed: 1,
  numpy_seed: 1,
  pytorch_seed: 1,
}
