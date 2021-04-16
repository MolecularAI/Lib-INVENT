import unittest

from rdkit import Chem
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints

from reaction_filters.reaction_filter_enum import ReactionFiltersEnum
from reaction_filters.reaction_filter import ReactionFilter
from running_modes.configurations import ReactionFilterConfiguration
from tests.unit_tests.fixtures.compounds import REACTION_SUZUKI, DECORATION_SUZUKI, SCAFFOLD_SUZUKI, SCAFFOLD_NO_SUZUKI, \
    DECORATION_NO_SUZUKI


class TestNonSelectiveReactionFilters(unittest.TestCase):
    def setUp(self):
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._enum = ReactionFiltersEnum()
        reactions = {"0": [REACTION_SUZUKI]}
        configuration = ReactionFilterConfiguration(type=self._enum.NON_SELECTIVE, reactions=reactions)
        self.reaction_filter = ReactionFilter(configuration)

    def test_with_suzuki_reagents(self):
        scaffold = SCAFFOLD_SUZUKI
        decoration = DECORATION_SUZUKI
        scaffold = self._attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule: Chem.Mol = self._bond_maker.join_scaffolds_and_decorations(scaffold, decoration)
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(1.0, score)

    def test_with_non_suzuki_reagents(self):
        scaffold = SCAFFOLD_NO_SUZUKI
        decoration = DECORATION_NO_SUZUKI
        scaffold = self._attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule: Chem.Mol = self._bond_maker.join_scaffolds_and_decorations(scaffold, decoration)
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(0.0, score)


class TestNonSelectiveReactionFiltersNoReaction(unittest.TestCase):
    def setUp(self):
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._enum = ReactionFiltersEnum()
        reactions = {"1": []}
        configuration = ReactionFilterConfiguration(type=self._enum.NON_SELECTIVE, reactions=reactions)
        self.reaction_filter = ReactionFilter(configuration)

    def test_with_suzuki_reagents(self):
        scaffold = SCAFFOLD_SUZUKI
        decoration = DECORATION_SUZUKI
        scaffold = self._attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule: Chem.Mol = self._bond_maker.join_scaffolds_and_decorations(scaffold, decoration)
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(1.0, score)

    def test_with_any_reagents(self):
        scaffold = SCAFFOLD_NO_SUZUKI
        decoration = DECORATION_NO_SUZUKI
        scaffold = self._attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule: Chem.Mol = self._bond_maker.join_scaffolds_and_decorations(scaffold, decoration)
        score = self.reaction_filter.evaluate(molecule)
        self.assertEqual(1.0, score)