import os
import shutil
import unittest

from running_modes.configurations import ScaffoldDecoratingConfiguration
from running_modes.scaffold_decorating.logging.scaffold_decorating_logger import ScaffoldDecoratingLogger
from running_modes.scaffold_decorating.scaffold_decoration import ScaffoldDecorator
from tests.unit_tests.fixtures.compounds import SCAFFOLD_TO_DECORATE, CELECOXIB_SCAFFOLD
from tests.unit_tests.fixtures.paths import MAIN_TEST_PATH, MODEL_PATH


class TestSingleScaffoldDecorating(unittest.TestCase):

    def setUp(self):
        self.testfolder = os.path.join(MAIN_TEST_PATH, "scaffold_decorating")
        self.input = os.path.join(self.testfolder, "input.smi")
        self.output = os.path.join(self.testfolder, "scaffold_decoration.smi")
        self.logdir = os.path.join(self.testfolder, "log")
        if not os.path.isdir(self.testfolder):
            os.makedirs(self.testfolder)
        self.prepare_input()
        self.config = ScaffoldDecoratingConfiguration(model_path=MODEL_PATH,
                                                      input_scaffold_path=self.input, output_path=self.output,
                                                      logging_path=self.logdir, batch_size=1,
                                                      number_of_decorations_per_scaffold=32,
                                                      randomize=True)

    def prepare_input(self):
        with open(self.input, "w+") as writer:
            writer.write(f"{SCAFFOLD_TO_DECORATE}\n")
            writer.close()

    def tearDown(self):
        if os.path.isdir(self.testfolder):
            shutil.rmtree(self.testfolder)
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_scaffold_decorating(self):
        logger = ScaffoldDecoratingLogger(self.logdir)
        runner = ScaffoldDecorator(self.config, logger)
        runner.run()
        self.assertEqual(os.path.isfile(self.output), True)
        self.assertEqual(os.path.isdir(self.logdir), True)


class TestMultipleScaffoldDecorating(unittest.TestCase):

    def setUp(self):
        self.testfolder = os.path.join(MAIN_TEST_PATH, "multiple_scaffold_decorating")
        self.input = os.path.join(self.testfolder, "input.smi")
        self.output = os.path.join(self.testfolder, "scaffold_decoration.smi")
        self.logdir = os.path.join(self.testfolder, "log")
        if not os.path.isdir(MAIN_TEST_PATH):
            os.makedirs(MAIN_TEST_PATH)
        if not os.path.isdir(self.testfolder):
            os.makedirs(self.testfolder)
        self.prepare_input()
        self.config = ScaffoldDecoratingConfiguration(model_path=MODEL_PATH,
                                                      input_scaffold_path=self.input, output_path=self.output,
                                                      logging_path=self.logdir, batch_size=1,
                                                      number_of_decorations_per_scaffold=32,
                                                      randomize=True, sample_uniquely=False)

    def prepare_input(self):
        with open(self.input, "w+") as writer:
            writer.write(f"{SCAFFOLD_TO_DECORATE}\n{CELECOXIB_SCAFFOLD}\n")
            writer.close()

    def tearDown(self):
        if os.path.isdir(self.testfolder):
            shutil.rmtree(self.testfolder)
        if os.path.isdir(MAIN_TEST_PATH):
            shutil.rmtree(MAIN_TEST_PATH)

    def test_scaffold_decorating(self):
        logger = ScaffoldDecoratingLogger(self.logdir)
        runner = ScaffoldDecorator(self.config, logger)
        runner.run()
        self.assertEqual(os.path.isfile(self.output), True)
        self.assertEqual(os.path.isdir(self.logdir), True)
