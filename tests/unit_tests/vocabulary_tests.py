import unittest

import models.model as mm
from tests.unit_tests.fixtures.paths import MODEL_PATH


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.smiles = "c1ccccc1CC0C"
        self.actor = mm.DecoratorModel.load_from_file(MODEL_PATH, mode="train")

    def test_something(self):
        tokenized = self.actor.vocabulary.scaffold_tokenizer.tokenize(self.smiles)
        # encoded = self.actor.vocabulary.encode(tokenized)
        self.assertEqual(14, len(tokenized))
