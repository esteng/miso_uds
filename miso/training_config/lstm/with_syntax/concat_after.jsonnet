{
    "dataset_reader": {
        "type": "decomp_syntax_semantics",
        "drop_syntax": "true",
        "generation_token_indexers": {
            "generation_tokens": {
                "type": "single_id",
                "namespace": "generation_tokens"
            }
        },
        "order": "inorder",
        "semantics_only": "false",
        "source_token_indexers": {
            "source_token_characters": {
                "type": "characters",
                "min_padding_length": 5,
                "namespace": "source_token_characters"
            },
            "source_tokens": {
                "type": "single_id",
                "namespace": "source_tokens"
            }
        },
        "syntactic_method": "concat-after",
        "target_token_indexers": {
            "target_token_characters": {
                "type": "characters",
                "min_padding_length": 5,
                "namespace": "target_token_characters"
            },
            "target_tokens": {
                "type": "single_id",
                "namespace": "target_tokens"
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer_for_amr",
            "args": null,
            "kwargs": {
                "do_lowercase": "false"
            },
            "model_name": "bert-base-cased"
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 16,
        "padding_noise": 0,
        "sorting_keys": [
            [
                "source_tokens",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "decomp_syntax_parser",
        "beam_size": 2,
        "bert_encoder": {
            "type": "seq2seq_bert_encoder",
            "config": "bert-base-cased"
        },
        "decoder": {
            "dropout": 0.33,
            "rnn_cell": {
                "hidden_size": 1024,
                "input_size": 1424,
                "num_layers": 2,
                "recurrent_dropout_probability": 0.33,
                "use_highway": false
            },
            "source_attention_layer": {
                "type": "global",
                "attention": {
                    "type": "mlp",
                    "hidden_vector_dim": 256,
                    "key_vector_dim": 1024,
                    "query_vector_dim": 1024,
                    "use_coverage": true
                },
                "key_vector_dim": 1024,
                "output_vector_dim": 1024,
                "query_vector_dim": 1024
            },
            "target_attention_layer": {
                "type": "global",
                "attention": {
                    "type": "mlp",
                    "hidden_vector_dim": 256,
                    "key_vector_dim": 1024,
                    "query_vector_dim": 1024,
                    "use_coverage": false
                },
                "key_vector_dim": 1024,
                "output_vector_dim": 1024,
                "query_vector_dim": 1024
            }
        },
        "decoder_node_index_embedding": {
            "embedding_dim": 50,
            "num_embeddings": 500
        },
        "decoder_pos_embedding": {
            "embedding_dim": 50,
            "vocab_namespace": "pos_tags"
        },
        "decoder_token_embedder": {
            "token_embedders": {
                "target_token_characters": {
                    "type": "character_encoding",
                    "dropout": 0.33,
                    "embedding": {
                        "embedding_dim": 100,
                        "vocab_namespace": "target_token_characters"
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 100,
                        "ngram_filter_sizes": [
                            3
                        ],
                        "num_filters": 50
                    }
                },
                "target_tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "/exp/estengel/miso/glove.840B.300d.zip",
                    "trainable": true,
                    "vocab_namespace": "target_tokens"
                }
            }
        },
        "dropout": 0.2,
        "edge_attribute_module": {
            "h_input_dim": 256,
            "hidden_dim": 1024,
            "loss_multiplier": 10,
            "n_layers": 4,
            "output_dim": 14
        },
        "edge_type_namespace": "edge_types",
        "encoder": {
            "type": "miso_stacked_bilstm",
            "batch_first": true,
            "hidden_size": 512,
            "input_size": 1118,
            "num_layers": 2,
            "recurrent_dropout_probability": 0.33,
            "stateful": true,
            "use_highway": false
        },
        "encoder_pos_embedding": {
            "embedding_dim": 100,
            "vocab_namespace": "pos_tags"
        },
        "encoder_token_embedder": {
            "token_embedders": {
                "source_token_characters": {
                    "type": "character_encoding",
                    "dropout": 0.33,
                    "embedding": {
                        "embedding_dim": 100,
                        "vocab_namespace": "source_token_characters"
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 100,
                        "ngram_filter_sizes": [
                            3
                        ],
                        "num_filters": 50
                    }
                },
                "source_tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "/exp/estengel/miso/glove.840B.300d.zip",
                    "trainable": true,
                    "vocab_namespace": "source_tokens"
                }
            }
        },
        "extended_pointer_generator": {
            "input_vector_dim": 1024,
            "source_copy": true,
            "target_copy": true
        },
        "label_smoothing": {
            "smoothing": 0
        },
        "max_decoding_steps": 100,
        "node_attribute_module": {
            "hidden_dim": 1024,
            "input_dim": 1024,
            "loss_multiplier": 10,
            "n_layers": 4,
            "output_dim": 44
        },
        "pos_tag_namespace": "pos_tags",
        "syntactic_method": "concat-after",
        "target_output_namespace": "generation_tokens",
        "tree_parser": {
            "attention": {
                "type": "biaffine",
                "key_vector_dim": 256,
                "query_vector_dim": 256
            },
            "dropout": 0.2,
            "edge_head_vector_dim": 256,
            "edge_type_vector_dim": 256,
            "key_vector_dim": 1024,
            "query_vector_dim": 1024
        }
    },
    "train_data_path": "train",
    "validation_data_path": "dev",
    "test_data_path": null,
    "trainer": {
        "validation_data_path": "dev",
        "type": "decomp_syntax_parsing",
        "cuda_device": 0,
        "drop_syntax": "true",
        "grad_clipping": null,
        "grad_norm": 5,
        "no_grad": [],
        "num_epochs": 250,
        "num_serialized_models_to_keep": 5,
        "optimizer": {
            "type": "adam",
            "amsgrad": true,
            "weight_decay": 3e-09
        },
        "patience": 40,
        "semantics_only": "false",
        "syntactic_method": "concat-after",
        "validation_metric": "+s_f1",
        "validation_prediction_path": "decomp_validation.txt"
    },
    "vocabulary": {
        "max_vocab_size": {
            "generation_tokens": 19700,
            "source_tokens": 19700,
            "target_tokens": 19700
        },
        "min_count": {
            "generation_tokens": 1,
            "source_tokens": 1,
            "target_tokens": 1
        },
        "non_padded_namespaces": []
    },
    "datasets_for_vocab_creation": [
        "train"
    ],
    "validation_iterator": {
        "type": "basic",
        "batch_size": 16
    }
}