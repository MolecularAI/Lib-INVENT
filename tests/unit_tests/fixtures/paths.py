import os
import json

from definitions import ROOT_DIR

with open(os.path.join(ROOT_DIR, 'configurations/config.json'), 'r') as f:
    config = json.load(f)


MAIN_TEST_PATH = config["MAIN_TEST_PATH"]
MODEL_PATH = f'{ROOT_DIR}/trained_models/recap.model'
CHEMBL_TL_MODEL_PATH = f'{ROOT_DIR}/trained_models/reaction_based.model'
DATA_PATH = f'{ROOT_DIR}/tests/unit_tests/unit.test.data'