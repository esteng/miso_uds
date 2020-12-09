local data_dir = "/exp/estengel/ud_data/all_data/";

{
  dataset_reader: {
    type: "ud-syntax",
    languages: ["hy"],
    alternate: false,
    instances_per_file: 32,
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
    tokenizer: {
                type: "pretrained_xlmr",
                model_name: "xlm-roberta-base",
               },
  },
  train_data_path: data_dir + "/train/*",
  validation_data_path: data_dir + "/dev/*",
  #test_data_path: null,
  #line_limit: 2,
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
    type: "ud_parser",
    bert_encoder: {
                    type: "seq2seq_xlmr_encoder",
                    config: "xlm-roberta-base",
    },
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
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
      hidden_size: 512,
      num_layers: 7,
      encoder_layer: {
          type: "pre_norm",
          d_model: 512,
          n_head: 16,
          norm: {type: "scale_norm",
                dim: 512},
          dim_feedforward: 2048,
          init_scale: 128,
          },
      dropout: 0.20,
    },
    biaffine_parser: {
      query_vector_dim: 512,
      key_vector_dim: 512,
      edge_head_vector_dim: 512,
      edge_type_vector_dim: 512,
      num_labels: 52,
      is_syntax: true,
      attention: {
        type: "biaffine",
        query_vector_dim: 512,
        key_vector_dim: 512,
      },
    }, 
    dropout: 0.2,
    syntax_edge_type_namespace: "syn_edge_types",
    pretrained_weights: "/exp/estengel/miso_res/xlmr_models/decomp_transformer_intermediate_no_positional/best.th"
  },
  iterator: {
    type: "bucket",
    # TODO: try to sort by target tokens.
    sorting_keys: [["source_tokens", "num_tokens"]],
    padding_noise: 0.0,
    batch_size: 30,
    instances_per_epoch: 20000,
  },
  validation_iterator: {
    type: "basic",
    batch_size: 64,
  },

  trainer: {
    type: "decomp_syntax_parsing",
    num_epochs: 200,
    #warmup_epochs: 200,
    patience: 40,
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
     learning_rate_scheduler: {
       type: "noam",
       model_size: 512, 
       warmup_steps: 4000,
     },
    no_grad: [],
    # smatch_tool_path: null, # "smatch_tool",
    validation_data_path: "dev",
    validation_prediction_path: "ud_validation.txt",
    semantics_only: false,
    syntactic_method: "encoder-side",
    drop_syntax: true,
  },
  random_seed: 12,
  numpy_seed: 12,
  pytorch_seed: 12,
}
