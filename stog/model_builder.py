from torchtext.vocab import Vectors
from stog.utils import logger
from stog.models import DeepBiaffineParser

def load_embedding(opt, train_data):
    # pretrained embeddings
    if opt.pretrain_token_emb:
        logger.info("Reading pretrained token embeddings from {} ...".format(opt.pretrain_token_emb))
        train_data.fields['tokens'].vocab.load_vectors(Vectors(opt.pretrain_token_emb))
        logger.info("Done.")

    if opt.pretrain_char_emb:
        logger.info("Reading pretrained char embeddings from {} ...".format(opt.pretrain_char_emb))
        train_data.fields['tokens'].vocab.load_vectors(Vectors(opt.pretrain_char_emb))
        logger.info("Done.")


def build_model(opt, train_data):

    if opt.model_type not in ["DeepBiaffineParser"]:
        raise NotImplementedError

    load_embedding(opt, train_data)

    if opt.model_type == "DeepBiaffineParser":
        model = DeepBiaffineParser(
            num_token_embeddings=len(train_data.fields["tokens"].vocab),
            token_embedding_dim=opt.token_emb_size,
            token_embedding_weight=train_data.fields["tokens"].vocab.vectors,
            num_char_embeddings=len(train_data.fields["chars"].vocab),
            char_embedding_dim=opt.char_emb_size,
            char_embedding_weight=train_data.fields["chars"].vocab.vectors,
            embedding_dropout_rate=opt.emb_dropout,
            hidden_state_dropout_rate=opt.hidden_dropout,
            use_char_conv=opt.use_char_conv,
            num_filters=opt.num_filters,
            kernel_size=opt.kernel_size,
            encoder_input_size=opt.encoder_size,
            encoder_hidden_size=opt.encoder_size,
            num_encoder_layers=opt.encoder_layers,
            encoder_dropout_rate=opt.encoder_dropout,
            edge_hidden_size=opt.edge_hidden_size,
            type_hidden_size=opt.type_hidden_size,
            num_labels=opt.num_labels
        )
        return model

