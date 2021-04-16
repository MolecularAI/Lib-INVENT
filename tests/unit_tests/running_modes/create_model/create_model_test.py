import os
import unittest

from running_modes.configurations.create_model_configuration import CreateModelConfiguration
from running_modes.create_model.create_model import CreateModel
from tests.unit_tests.fixtures import MAIN_TEST_PATH, DATA_PATH


class TestCreateModel(unittest.TestCase):
    def setUp(self):
        self.testfolder = os.path.join(MAIN_TEST_PATH, "scaffold_decorating")
        self.input_smiles_path = os.path.join(DATA_PATH, "train.smi")
        self.output_model_path = os.path.join(self.testfolder, "empty.model.unit.test")
        if not os.path.isdir(self.testfolder):
            os.makedirs(self.testfolder)

        # Leaving all the model parameters default. Can be changed as required,
        self.config = CreateModelConfiguration(self.input_smiles_path, self.output_model_path)

    def test_create_model_run(self):
        runner = CreateModel(configuration=self.config)
        runner.run()
        self.assertEqual(os.path.exists(self.output_model_path), True)

