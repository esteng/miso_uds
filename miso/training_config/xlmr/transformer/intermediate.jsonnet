local data_dir = "train";
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
    drop_syntax: true,
    semantics_only: false,
    syntactic_method: synt_method,
    order: "inorder",
    tokenizer: {
                type: "pretrained_xlmr",
                model_name: "xlm-roberta-base",
               },
  },
  train_data_path: data_dir,
  validation_data_path: "dev",
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
      source_tokens: 19700,
      target_tokens: 19700,
      generation_tokens: 19700,
    },
  },

  model: {
    type: "decomp_transformer_syntax_parser",
    syntactic_method: synt_method,
    intermediate_graph: true,
    bert_encoder: {
                    type: "seq2seq_xlmr_encoder",
                    config: "xlm-roberta-base",
    },
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
            embedding_dim: 100,
          },
          encoder: {
            type: "cnn",
            embedding_dim: 100,
            num_filters: 50,
            ngram_filter_sizes: [3],
          },
          dropout: 0.33,
        },
      },
    },
    encoder_pos_embedding: {
      vocab_namespace: "pos_tags",
      embedding_dim: 100,
    },
    encoder: {
      type: "transformer_encoder",
      input_size: 300 + 50 + 768,
      hidden_size: 256,
      num_layers: 6,
      encoder_layer: {
          type: "pre_norm",
          d_model: 256,
          n_head: 8,
          norm: {type: "scale_norm",
                dim: 256},
          dim_feedforward: 2048,
          init_scale: 512,
          },
      dropout: 0.20,
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
            embedding_dim: 100,
          },
          encoder: {
            type: "cnn",
            embedding_dim: 100,
            num_filters: 50,
            ngram_filter_sizes: [3],
          },
          dropout: 0.33,
        },
      },
    },
    decoder_node_index_embedding: {
      # vocab_namespace: "node_indices",
      num_embeddings: 500,
      embedding_dim: 50,
    },
    decoder_pos_embedding: {
      vocab_namespace: "pos_tags",
      embedding_dim: 50,
    },
    biaffine_parser: {
      query_vector_dim: 256,
      key_vector_dim: 256,
      edge_head_vector_dim: 256,
      edge_type_vector_dim: 256,
      num_labels: 49,
      is_syntax: true,
      dropout: 0.2,
      attention: {
        type: "biaffine",
        query_vector_dim: 256,
        key_vector_dim: 256,
      },
    }, 
    decoder: {
      input_size: 300 + 50 + 50,
      hidden_size: 768,
      num_layers: 6,
      use_coverage: true,
      type: "transformer_decoder", 
      decoder_layer: {
        type: "pre_norm",
        d_model: 768, 
        n_head: 8,
        norm: {type: "scale_norm",
               dim: 512},
        dim_feedforward: 2048,
        dropout: 0.20,
        init_scale: 512,
      },
      source_attention_layer: {
        type: "global",
        query_vector_dim: 768,
        key_vector_dim: 768,
        output_vector_dim: 768,
        attention: {
          type: "mlp",
          # TODO: try to use smaller dims.
          query_vector_dim: 768,
          key_vector_dim: 768,
          hidden_vector_dim: 128, 
          use_coverage: true,
        },
      },
      target_attention_layer: {
        type: "global",
        query_vector_dim: 768,
        key_vector_dim: 768,
        output_vector_dim: 768,
        attention: {
          type: "mlp",
          query_vector_dim: 768,
          key_vector_dim: 768,
          hidden_vector_dim: 128,
        },
      },
    },
    extended_pointer_generator: {
      input_vector_dim: 768,
      source_copy: true,
      target_copy: true,
    },
    tree_parser: {
      query_vector_dim: 768,
      key_vector_dim: 768, 
      edge_head_vector_dim: 768,
      edge_type_vector_dim: 128,
      dropout: 0.2,
      attention: {
        type: "biaffine",
        query_vector_dim: 768,
        key_vector_dim: 768,
      },
    },
    node_attribute_module: {
        input_dim: 768,
        hidden_dim: 1024,
        output_dim: 44,
        n_layers: 4, 
        dropout: 0.2,
        loss_multiplier: 10,
    },
    edge_attribute_module: {
        h_input_dim: 128,
        hidden_dim: 1024,
        output_dim: 14,
        n_layers: 4, 
        dropout: 0.2,
        loss_multiplier: 10,
    },
    label_smoothing: {
        smoothing: 0.0,
    },
    dropout: 0.20,
    beam_size: 2,
    max_decoding_steps: 60,
    target_output_namespace: "generation_tokens",
    pos_tag_namespace: "pos_tags",
    edge_type_namespace: "edge_types",
    syntax_edge_type_namespace: "syn_edge_types",
    loss_mixer: {type:"static-syntax-heavy",
                 weight: 2},
  },

  iterator: {
    type: "bucket",
    # TODO: try to sort by target tokens.
    sorting_keys: [["source_tokens", "num_tokens"]],
    padding_noise: 0.0,
    batch_size: 30,
  },
  validation_iterator: {
    type: "basic",
    batch_size: 64,
  },

  trainer: {
    type: "decomp_syntax_parsing",
    num_epochs: 450,
    patience: 50,
    warmup_epochs: 0,
    grad_norm: 5.0,
    # TODO: try to use grad clipping.
    grad_clipping: null,
    cuda_device: 0,
    num_serialized_models_to_keep: 5,
    validation_metric: "+syn_uas",
    optimizer: {
      type: "adam",
      betas: [0.9, 0.999],
      eps: 1e-9,
      lr: 0.0000, 
      weight_decay: 3e-9, 
      amsgrad: true,
    },
    bert_optimizer: {
        type: "adam",
        lr: 1e-5,
    },
    bert_tune_layer: 5,
     learning_rate_scheduler: {
       type: "noam",
       model_size: 512, 
       warmup_steps: 8000,
     },
    no_grad: [],
    # smatch_tool_path: null, # "smatch_tool",
    validation_data_path: "dev",
    validation_prediction_path: "decomp_validation.txt",
    semantics_only: false,
    drop_syntax: true,
    syntactic_method: synt_method,
  },
  random_seed: 12,
  numpy_seed: 12,
  pytorch_seed: 12,
}
