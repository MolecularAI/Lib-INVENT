import unittest

from models.model import DecoratorModel
from running_modes.enums import GenerativeModelRegimeEnum
from tests.unit_tests.fixtures.paths import MODEL_PATH
from tests.unit_tests.fixtures import SCAFFOLD_SUZUKI

import torch.utils.data as tud
import models.dataset as md


class TestDecoratorModel(unittest.TestCase):
    def setUp(self):
        input_scaffold = SCAFFOLD_SUZUKI
        scaffold_list_1 = [input_scaffold]
        scaffold_list_2 = [input_scaffold, input_scaffold]
        scaffold_list_3 = [input_scaffold, input_scaffold, input_scaffold]
        self._model_regime = GenerativeModelRegimeEnum()
        self._decorator = DecoratorModel.load_from_file(MODEL_PATH, mode=self._model_regime.INFERENCE)

        dataset_1 = md.Dataset(scaffold_list_1*2, self._decorator.vocabulary.scaffold_vocabulary,
                               self._decorator.vocabulary.scaffold_tokenizer)
        self.dataloader_1 = tud.DataLoader(dataset_1, batch_size=32, shuffle=False, collate_fn=md.Dataset.collate_fn)

        dataset_2 = md.Dataset(scaffold_list_2, self._decorator.vocabulary.scaffold_vocabulary,
                               self._decorator.vocabulary.scaffold_tokenizer)
        self.dataloader_2 = tud.DataLoader(dataset_2, batch_size=32, shuffle=False, collate_fn=md.Dataset.collate_fn)

        dataset_3 = md.Dataset(scaffold_list_3, self._decorator.vocabulary.scaffold_vocabulary,
                               self._decorator.vocabulary.scaffold_tokenizer)
        self.dataloader_3 = tud.DataLoader(dataset_3, batch_size=32, shuffle=False, collate_fn=md.Dataset.collate_fn)

    def test_single_scaffold_input(self):
        results = []
        for batch in self.dataloader_1:
            for scaff, decorations, nll in self._decorator.sample_decorations(*batch):
                results.append(decorations)
        self.assertEqual(2, len(results))

    def test_double_scaffold_input(self):
        results = []
        for batch in self.dataloader_2:
            for scaff, decorations, nll in self._decorator.sample_decorations(*batch):
                results.append(decorations)
        self.assertEqual(2, len(results))

    def test_triple_scaffold_input(self):
        results = []
        for batch in self.dataloader_3:
            for scaff, decorations, nll in self._decorator.sample_decorations(*batch):
                results.append(decorations)
        self.assertEqual(3, len(results))
