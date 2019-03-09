import os
class DataWriter:

    def __init__(self):
        pass

    def set_dir(self, path):
        self.dir = path

    def reset_file_epoch(self, epoch):
        self.file_path = os.path.join(self.dir, "pred.{}".format(epoch))
        if os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                pass


    def predict_instance(self, list_dict):
        raise NotImplementedError

    def predict_instance_batch(self, torch_dict, batch):
        raise NotImplementedError

    def set_vocab(self, vocab):
        self.vocab = vocab

    def write_instance_batch(self, torch_dict, batch):
        for tree in self.predict_instance_batch(torch_dict, batch):
            self.write_instance(tree)

    def write_instance(self, tree):
        with open(self.file_path, 'a') as f:
            f.write(tree.pretty_str() + '\n')
