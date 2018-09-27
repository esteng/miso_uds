from stog.utils import logging


logger = logging.init_logger()



class Trainer:

    def __init__(self, model, optim):
        self.model = model
        self.optim = optim

    def train(self, training_iterator, dev_iterator, train_epoches, dev_steps):
        logger.info('Start training...')

        step = self.optim._step + 1
        epoch = self.optim._epoch + 1
        while epoch <= train_epoches:
            logger.info('Epoch {}'.format(epoch))
            train_loss = 0.0

            for i, batch in enumerate(training_iterator):
                tokens, chars, headers, mask = batch
                self.optim.zero_grad()

                edge_scores = self.model(tokens, chars, mask)
                loss = self.model.loss(edge_scores, headers)

                loss.backward()
                self.optim.step()

                num_tokens = mask.data.sum() - mask.size(0)
                train_loss += loss.item()

                step += 1

            epoch += 1


