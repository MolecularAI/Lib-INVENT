from models.actions import Action
import torch.utils.data as tud
import models.dataset as md


class CalculateNLLsFromModel(Action):

    def __init__(self, model, batch_size, logger=None):
        """
        Creates an instance of CalculateNLLsFromModel.
        :param model: A model instance.
        :param batch_size: Batch size to use.
        :return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.batch_size = batch_size

    def run(self, scaffold_decoration_list):
        """
        Calculates the NLL for a set of SMILES strings.
        :param scaffold_decoration_list: List with pairs of (scaffold, decoration) SMILES.
        :return: An iterator with each NLLs in the same order as the list.
        """
        dataset = md.DecoratorDataset(scaffold_decoration_list, self.model.vocabulary)
        dataloader = tud.DataLoader(dataset, batch_size=self.batch_size, collate_fn=md.DecoratorDataset.collate_fn,
                                    shuffle=False)
        for scaffold_batch, decorator_batch in dataloader:
            for nll in self.model.likelihood(*scaffold_batch, *decorator_batch).data.cpu().numpy():
                yield nll