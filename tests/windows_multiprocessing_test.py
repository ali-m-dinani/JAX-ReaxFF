import os
from pathlib import Path
import unittest

import numpy as onp

from jax_md.reaxff.reaxff_forcefield import ForceField
from jax_md.reaxff.reaxff_helper import read_force_field
from jaxreaxff.helper import count_inter_list_sizes, move_dataclass, read_geo_file
from jaxreaxff.inter_list_counter import calculate_inter_list_sizes


@unittest.skipUnless(os.name == "nt", "Windows multiprocessing regression")
class WindowsMultiprocessingTest(unittest.TestCase):
  def test_spawn_matches_serial_counts(self):
    dataset = Path(__file__).resolve().parents[1] / "Datasets" / "cobalt"
    force_field = read_force_field(str(dataset / "ffield_lit"))
    force_field = ForceField.fill_off_diag(force_field)
    force_field = ForceField.fill_symm(force_field)
    force_field = move_dataclass(force_field, onp)
    systems = read_geo_file(str(dataset / "geo"), force_field.name_to_index, 10.0)[:2]
    self.assertEqual(len(systems), 2)
    expected = [calculate_inter_list_sizes(system, force_field) for system in systems]
    actual = count_inter_list_sizes(systems, force_field, num_threads=2, chunksize=1)
    self.assertEqual(actual, expected)


if __name__ == "__main__":
  unittest.main()
