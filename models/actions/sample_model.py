from models.actions import Action
import torch.utils.data as tud
import models.dataset as md


class SampleModel(Action):

    def __init__(self, model, batch_size, logger=None):
        """
        Creates an instance of SampleModel.
        :params model: A model instance (better in scaffold_decorating mode).
        :params batch_size: Batch size to use.
        :return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.batch_size = batch_size

    def run(self, scaffold_list):
        """
        Samples the model for the given number of SMILES.
        :params scaffold_list: A list of scaffold SMILES.
        :return: An iterator with each of the batches sampled in (scaffold, decoration, nll) triplets.
        """
        dataset = md.Dataset(scaffold_list, self.model.vocabulary.scaffold_vocabulary,
                             self.model.vocabulary.scaffold_tokenizer)
        dataloader = tud.DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=False, collate_fn=md.Dataset.collate_fn)
        for batch in dataloader:
            for scaff, dec, nll in self.model.sample_decorations(*batch):
                yield scaff, dec, nll
