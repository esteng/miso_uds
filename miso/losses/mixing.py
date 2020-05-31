from overrides import overrides

import torch
import torch.nn.functional as F
from allennlp.common.registrable import Registrable

class LossMixer(torch.nn.Module, Registrable):
    def __init__(self):
        super().__init__()
        # convention: semantics loss, syntax loss
        self.loss_weights = [1,1]

    def forward(self, sem_loss, syn_loss): 
        # convention: semantics loss, syntax loss
        return self.loss_weights[0] * sem_loss +\
                self.loss_weights[1] * syn_loss

    def update_weights(self, curr_epoch, total_epochs):
        raise NotImplementedError

@LossMixer.register("alternating") 
class AlternatingLossMixer(LossMixer):
    """
    Alternate between all syntax or all semantics loss 
    """
    def __init__(self):
        super().__init__() 
        self.syn_loss_weights = [0,1]
        self.sem_loss_weights = [1,0]
        self.loss_weights = self.syn_loss_weights

    def update_weights(self, curr_epoch, total_epochs): 
        print(f"updating loss weights with curr_epoch {curr_epoch} and total_epochs {total_epochs}" )
        if curr_epoch % 2 == 0:
            self.loss_weights = self.syn_loss_weights
        else:
            self.loss_weights = self.sem_loss_weights

@LossMixer.register("syntax->semantics") 
class SyntaxSemanticsLossMixer(LossMixer):
    """
    Start with all syntax loss, move to all semantics loss 
    """
    def __init__(self):
        super().__init__() 
        self.loss_weights = [0,1]

    def update_weights(self, curr_epoch, total_epochs): 
        # take steps towards all semantics loss s.t. by the end of training, 
        # semantics loss weight is 1
        step_size = 1/total_epochs
        syn_weight = 1 - step_size * curr_epoch
        self.loss_weights[1] = syn_weight
        self.loss_weights[0] = 1 - syn_weight

@LossMixer.register("semantics->syntax") 
class SemanticsSyntaxLossMixer(LossMixer):
    """
    Start with all semantics loss, move to all syntax loss 
    """
    def __init__(self):
        super().__init__() 
        self.loss_weights = [1,0]

    def update_weights(self, curr_epoch, total_epochs): 
        # take steps towards all syntax loss s.t. by the end of training, 
        # syntax loss weight is 1
        step_size = 1/total_epochs
        sem_weight = 1 - step_size * curr_epoch
        self.loss_weights[0] = sem_weight
        self.loss_weights[1] = 1 - sem_weight

@LossMixer.register("semantics-only") 
class SemanticsOnlyLossMixer(LossMixer):
    """
    Start with all semantics loss, move to all syntax loss 
    """
    def __init__(self):
        super().__init__() 
        self.loss_weights = [1,0]

    def update_weights(self, curr_epoch, total_epochs): 
        pass

@LossMixer.register("static-semantics-heavy") 
class SemanticsHeavyLossMixer(LossMixer):
    """
    Downweight syntactic loss so that it's roughly the same magnitude as semantic loss 
    based on observed ratio of losses 
    """
    def __init__(self):
        super().__init__() 
        self.loss_weights = [1, 0.003]

    def update_weights(self, curr_epoch, total_epochs): 
        pass

@LossMixer.register("learned") 
class LearnedLossMixer(LossMixer):
    """
    Downweight syntactic loss so that it's roughly the same magnitude as semantic loss 
    based on observed ratio of losses 
    """
    def __init__(self):
        super().__init__() 
        # placeholder 
        self.loss_weights = [0.5, 0.5]
        # start at 50-50
        self.semantics_raw_weight = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))

    @overrides
    def forward(self, sem_loss, syn_loss): 
        sem_weight = F.sigmoid(self.semantics_raw_weight)
        syn_weight = 1 - sem_weight

        # convention: semantics loss, syntax loss
        return sem_weight * sem_loss +\
               syn_weight * syn_loss

    def update_weights(self, curr_epoch, total_epochs): 
        pass


