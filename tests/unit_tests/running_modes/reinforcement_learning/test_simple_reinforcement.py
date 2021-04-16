import os
import shutil
import unittest
from reaction_filters.reaction_filter_enum import ReactionFiltersEnum
import torch
from reinvent_scoring import ScoringFunctionNameEnum, ScoringFunctionComponentNameEnum, ScoringFuncionParameters, \
    ComponentParameters

from diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from models.model import DecoratorModel

from running_modes.configurations import ReactionFilterConfiguration, ReinforcementLearningConfiguration, \
    LearningStrategyConfiguration
from running_modes.configurations.log_configuration import LogConfiguration
from running_modes.configurations.scoring_strategy_configuration import ScoringStrategyConfiguration
from running_modes.enums import GenerativeModelRegimeEnum, LearningStrategyEnum
from running_modes.reinforcement_learning.logging import ReinforcementLogger
from running_modes.reinforcement_learning.reinforcement_learning import ReinforcementLearning
from tests.unit_tests.fixtures.compounds import CELECOXIB_SCAFFOLD, CELECOXIB, REACTION_SUZUKI
from tests.unit_tests.fixtures.paths import MAIN_TEST_PATH, MODEL_PATH


class TestSimpleReinforcement(unittest.TestCase):

    def setUp(self):
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)

        _model_regime = GenerativeModelRegimeEnum
        reaction_filter_enum = ReactionFiltersEnum()
        sf_enum = ScoringFunctionNameEnum()
        sf_component_enum = ScoringFunctionComponentNameEnum()
        learning_strategy_enum = LearningStrategyEnum()
        scaffolds = [CELECOXIB_SCAFFOLD]
        self.workfolder = MAIN_TEST_PATH
        smiles = [CELECOXIB]
        ts_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 smiles=smiles, model_path="",
                                                 component_type=sf_component_enum.TANIMOTO_SIMILARITY,
                                                 specific_parameters={}))

        sf_parameters = {"name": sf_enum.CUSTOM_SUM,
                         "parameters": [ts_parameters]}
        sf_config = ScoringFuncionParameters(**sf_parameters)

        diversity_config = {"name": "NoFilter", "minscore": 0.05, "bucket_size": 25, "minsimilarity": 0.4}
        diversity_parameters = DiversityFilterParameters(**diversity_config)
        reactions = {"0": [REACTION_SUZUKI]}
        reaction_config = ReactionFilterConfiguration(reaction_filter_enum.SELECTIVE, reactions)

        learning_strategy_config = LearningStrategyConfiguration(name=learning_strategy_enum.DAP, parameters={})

        scoring_strategy = ScoringStrategyConfiguration(reaction_filter=reaction_config,
                                                        diversity_filter=diversity_parameters,
                                                        scoring_function=sf_config, name="standard")

        rl_config = ReinforcementLearningConfiguration(actor=MODEL_PATH, critic=MODEL_PATH,
                                                       scaffolds=scaffolds,
                                                       n_steps=3, batch_size=64,
                                                       learning_strategy=learning_strategy_config,
                                                       scoring_strategy=scoring_strategy)

        critic = DecoratorModel.load_from_file(rl_config.critic, mode=_model_regime.INFERENCE)
        actor = DecoratorModel.load_from_file(rl_config.actor, mode=_model_regime.TRAINING)

        log_config = LogConfiguration(logging_path=f"{self.workfolder}/log", recipient="local")
        logger = ReinforcementLogger(log_config)

        self.runner = ReinforcementLearning(critic, actor, rl_config, logger)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_simple_reinforcement(self):
        self.runner.run()
        self.assertEqual(os.path.isdir(f"{self.workfolder}/log"), True)


class TestMAULI(unittest.TestCase):

    def setUp(self):
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)

        _model_regime = GenerativeModelRegimeEnum
        reaction_filter_enum = ReactionFiltersEnum()
        sf_enum = ScoringFunctionNameEnum()
        sf_component_enum = ScoringFunctionComponentNameEnum()
        learning_strategy_enum = LearningStrategyEnum()
        scaffolds = [CELECOXIB_SCAFFOLD]
        self.workfolder = MAIN_TEST_PATH
        smiles = [CELECOXIB]
        ts_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 smiles=smiles, model_path="",
                                                 component_type=sf_component_enum.TANIMOTO_SIMILARITY,
                                                 specific_parameters={}))

        sf_parameters = {"name": sf_enum.CUSTOM_SUM,
                         "parameters": [ts_parameters]}
        sf_config = ScoringFuncionParameters(**sf_parameters)

        diversity_config = {"name": "NoFilter", "minscore": 0.05, "bucket_size": 25, "minsimilarity": 0.4}
        diversity_parameters = DiversityFilterParameters(**diversity_config)
        reactions = {"0": [REACTION_SUZUKI]}
        reaction_config = ReactionFilterConfiguration(reaction_filter_enum.SELECTIVE, reactions)

        learning_strategy_config = LearningStrategyConfiguration(name=learning_strategy_enum.MAULI,
                                                                 parameters={"sigma": 100})

        scoring_strategy = ScoringStrategyConfiguration(reaction_filter=reaction_config,
                                                        diversity_filter=diversity_parameters,
                                                        scoring_function=sf_config, name="standard")

        rl_config = ReinforcementLearningConfiguration(actor=MODEL_PATH, critic=MODEL_PATH,
                                                       scaffolds=scaffolds,
                                                       n_steps=3, batch_size=64,
                                                       learning_strategy=learning_strategy_config,
                                                       scoring_strategy=scoring_strategy)

        critic = DecoratorModel.load_from_file(rl_config.critic, mode=_model_regime.INFERENCE)
        actor = DecoratorModel.load_from_file(rl_config.actor, mode=_model_regime.TRAINING)

        log_config = LogConfiguration(logging_path=f"{self.workfolder}/log", recipient="local")
        logger = ReinforcementLogger(log_config)

        self.runner = ReinforcementLearning(critic, actor, rl_config, logger)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_mauli(self):
        self.runner.run()
        self.assertEqual(os.path.isdir(f"{self.workfolder}/log"), True)


class TestSDAP(unittest.TestCase):

    def setUp(self):
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)

        _model_regime = GenerativeModelRegimeEnum
        reaction_filter_enum = ReactionFiltersEnum()
        sf_enum = ScoringFunctionNameEnum()
        sf_component_enum = ScoringFunctionComponentNameEnum()
        learning_strategy_enum = LearningStrategyEnum()
        scaffolds = [CELECOXIB_SCAFFOLD]
        self.workfolder = MAIN_TEST_PATH
        smiles = [CELECOXIB]
        ts_parameters = vars(ComponentParameters(name="tanimoto similarity", weight=1,
                                                 smiles=smiles, model_path="",
                                                 component_type=sf_component_enum.TANIMOTO_SIMILARITY,
                                                 specific_parameters={}))

        sf_parameters = {"name": sf_enum.CUSTOM_SUM,
                         "parameters": [ts_parameters]}
        sf_config = ScoringFuncionParameters(**sf_parameters)

        diversity_config = {"name": "NoFilter", "minscore": 0.05, "bucket_size": 25, "minsimilarity": 0.4}
        diversity_parameters = DiversityFilterParameters(**diversity_config)
        reactions = {"0": [REACTION_SUZUKI]}
        reaction_config = ReactionFilterConfiguration(reaction_filter_enum.SELECTIVE, reactions)

        learning_strategy_config = LearningStrategyConfiguration(name=learning_strategy_enum.SDAP,
                                                                 parameters={"sigma": 120})

        scoring_strategy = ScoringStrategyConfiguration(reaction_filter=reaction_config,
                                                        diversity_filter=diversity_parameters,
                                                        scoring_function=sf_config, name="standard")

        rl_config = ReinforcementLearningConfiguration(actor=MODEL_PATH, critic=MODEL_PATH,
                                                       scaffolds=scaffolds,
                                                       n_steps=3, batch_size=64,
                                                       learning_strategy=learning_strategy_config,
                                                       scoring_strategy=scoring_strategy)

        critic = DecoratorModel.load_from_file(rl_config.critic, mode=_model_regime.INFERENCE)
        actor = DecoratorModel.load_from_file(rl_config.actor, mode=_model_regime.TRAINING)

        log_config = LogConfiguration(logging_path=f"{self.workfolder}/log", recipient="local")
        logger = ReinforcementLogger(log_config)

        self.runner = ReinforcementLearning(critic, actor, rl_config, logger)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

    def test_sdap(self):
        self.runner.run()
        self.assertEqual(os.path.isdir(f"{self.workfolder}/log"), True)


