import random
import scipy.stats as sps

import numpy as np
import torch
from rdkit import Chem
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints

from models.actions.calculate_nlls_from_model import CalculateNLLsFromModel
from models.actions.sample_model import SampleModel


class CollectStatsFromModel:
    """Collects stats from an existing RNN model."""

    def __init__(self, model, epoch, training_set, validation_set, sample_size,
                 decoration_type="all", with_weights=False, other_values=None):
        """
        Creates an instance of CollectStatsFromModel.
        : param model: A model instance initialized as sampling_mode.
        : param epoch: Epoch number to be sampled(informative purposes).
        : param training_set: Iterator with the training set.
        : param validation_set: Iterator with the validation set.
        : param writer: Writer object(Tensorboard writer).
        : param other_values: Other values to save for the epoch.
        : param sample_size: Number of molecules to sample from the training / validation / sample set.
        : param decoration_type: Kind of decorations (single or all).
        : param with_weights: To calculate or not the weights.
        : return:
        """

        self.model = model
        self.epoch = epoch
        self.sample_size = sample_size
        self.training_set = training_set
        self.validation_set = validation_set
        self.other_values = other_values

        self.decoration_type = decoration_type
        self.with_weights = with_weights
        self.sample_size = max(sample_size, 1)

        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._calc_nlls_action = CalculateNLLsFromModel(self.model, 128)
        self._sample_model_action = SampleModel(self.model, 128)

    @torch.no_grad()
    def run(self):
        """
        Collects stats for a specific model object, epoch, validation set, training set and writer object.
        : return: A dictionary with all the data saved for that given epoch.
        """
        data = {}
        sliced_training_set = list(random.sample(self.training_set, self.sample_size))
        sliced_validation_set = list(random.sample(self.validation_set, self.sample_size))

        sampled_training_mols, sampled_training_nlls = self._sample_decorations(next(zip(*sliced_training_set)))
        sampled_validation_mols, sampled_validation_nlls = self._sample_decorations(next(zip(*sliced_validation_set)))

        training_nlls = np.array(list(self._calc_nlls_action.run(sliced_training_set)))
        validation_nlls = np.array(list(self._calc_nlls_action.run(sliced_validation_set)))

        data.update({"sampled_training_mols": sampled_training_mols, "sampled_validation_mols": sampled_validation_mols,
                     "training_nlls": training_nlls, "validation_nlls": validation_nlls,
                     "binned_jsd": self.jsd([sampled_training_nlls, sampled_validation_nlls,
                                             training_nlls, validation_nlls], binned=True),
                     "unbinned_jsd": self.jsd([sampled_training_nlls, sampled_validation_nlls,
                                               training_nlls, validation_nlls], binned=False)
                     })
        return data

    def _sample_decorations(self, scaffold_list):
        mols = []
        nlls = []
        for scaff, decoration, nll in self._sample_model_action.run(scaffold_list):
            labeled_scaffold = self._attachment_points.add_attachment_point_numbers(scaff, canonicalize=False)
            molecule = self._bond_maker.join_scaffolds_and_decorations(labeled_scaffold, decoration)
            if molecule:
                mols.append(Chem.MolToSmiles(molecule))
            nlls.append(nll)
        return mols, np.array(nlls)

    def bin_dist(self, dist, bins=1000, dist_range=(0, 100)):
        bins = np.histogram(dist, bins=bins, range=dist_range, density=False)[0]
        bins[bins == 0] = 1
        return bins / bins.sum()

    def jsd(self, dists, binned=False):
        min_size = min(len(dist) for dist in dists)
        dists = [dist[:min_size] for dist in dists]
        if binned:
            dists = [self.bin_dist(dist) for dist in dists]
        num_dists = len(dists)
        avg_dist = np.sum(dists, axis=0) / num_dists
        return sum((sps.entropy(dist, avg_dist) for dist in dists)) / num_dists
