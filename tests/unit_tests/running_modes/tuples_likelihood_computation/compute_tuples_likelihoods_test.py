import os
import shutil
import unittest

from running_modes.configurations.tuples_likelihood_computation_configuration import \
    TuplesLikelihoodComputationConfiguration
from running_modes.tuples_likelihood_computation.tuples_likelihood_computation import \
    ComputeScaffoldDecorationLikelihoods
from tests.unit_tests.fixtures import DATA_PATH, MAIN_TEST_PATH, MODEL_PATH


class TestComputeScaffoldDecorationLikelihoods(unittest.TestCase):
    def setUp(self):
        self.testfolder = os.path.join(MAIN_TEST_PATH, "tuples_likelihood_computation")
        if not os.path.isdir(self.testfolder):
            os.makedirs(self.testfolder)

        self.config = TuplesLikelihoodComputationConfiguration(
            input_csv_path=f"{DATA_PATH}/scaffold_decoration_pairs.smi",
            output_csv_path=f"{self.testfolder}/likelihoods.smi",
            model_path=MODEL_PATH)

    def test_run(self):
        runner = ComputeScaffoldDecorationLikelihoods(self.config)
        runner.run()
        self.assertEqual(os.path.isfile(self.config.output_csv_path), True)

    def tearDown(self):
        if os.path.isdir(self.testfolder):
            shutil.rmtree(self.testfolder)
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)
