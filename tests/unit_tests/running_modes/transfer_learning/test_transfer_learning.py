import os
import shutil
import unittest
import torch

from running_modes.configurations import TransferLearningConfiguration
from running_modes.configurations.transfer_learning_configuration import LearningRate
from running_modes.transfer_learning.transfer_learning import LargeScaleTransferLearning
from tests.unit_tests.fixtures import MAIN_TEST_PATH, CHEMBL_TL_MODEL_PATH, DATA_PATH


class TestTransferLearning(unittest.TestCase):
    def setUp(self):
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        self.testfolder = os.path.join(MAIN_TEST_PATH, "transfer_learning")
        self.training_path = os.path.join(DATA_PATH, "train.smi")
        self.val_path = os.path.join(DATA_PATH, "val.smi")
        self.model_path = CHEMBL_TL_MODEL_PATH
        self.output = os.path.join(self.testfolder, "tl_output")
        self.logdir = os.path.join(self.testfolder, "log.dir")
        _lr = LearningRate()

        if not os.path.isdir(self.testfolder):
            os.makedirs(self.testfolder)

        self.config = TransferLearningConfiguration(model_path=self.model_path,
                                                    output_path=self.output,
                                                    training_set_path=self.training_path,
                                                    validation_sets_path=self.val_path,
                                                    logging_path=self.logdir, sample_size=2, learning_rate=_lr,
                                                    epochs=1, collect_stats_frequency=1, batch_size=2)
        self.tl = LargeScaleTransferLearning(self.config)

    def tearDown(self):
        if os.path.isdir(self.testfolder):
            shutil.rmtree(self.testfolder)


    def test_load_sets(self):
        test_set = self.tl.load_sets(self.training_path)
        compound_list = next(test_set)
        # check the right format of the entries: [scaffold, decorations]
        self.assertEqual(len(compound_list[0]), 2)

    def test_transfer_learning_run(self):
        self.tl.run()
        # check the model was saved and logs ouputted
        self.assertEqual(os.path.exists(f"{self.output}/trained.1"), True)
        self.assertEqual(os.path.exists(self.logdir), True)


