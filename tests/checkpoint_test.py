from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import numpy as onp

from jax_md.reaxff.reaxff_helper import read_force_field
from jaxreaxff.checkpoint import ForceFieldCheckpoint
from jaxreaxff import optimizer


jax.config.update('jax_enable_x64', True)


class CheckpointTest(unittest.TestCase):
  def setUp(self):
    self.folder = tempfile.TemporaryDirectory()
    self.addCleanup(self.folder.cleanup)
    self.output = Path(self.folder.name)
    self.initial = str(Path(__file__).resolve().parents[1] /
                       'Datasets' / 'cobalt' / 'ffield_lit')
    self.force_field = read_force_field(self.initial, dtype=onp.float64)
    self.indices = (('over_coord1', (0,)),)
    self.initial_params = onp.array([float(self.force_field.over_coord1[0])])

  def make_checkpoint(self, save_opt):
    return ForceFieldCheckpoint(self.initial, self.output, self.force_field,
                                self.indices, save_opt)

  def assert_saved_params(self, filename, expected):
    restored = read_force_field(str(self.output / filename))
    self.assertAlmostEqual(float(restored.over_coord1[0]), float(expected[0]), places=4)

  def test_all_keeps_trial_bests_and_global_best(self):
    checkpoint = self.make_checkpoint('all')
    checkpoint(1, 0, self.initial_params, 10.0)
    checkpoint(1, 1, self.initial_params + 0.25, 5.0)
    checkpoint(2, 0, self.initial_params + 0.5, 8.0)
    self.assert_saved_params('checkpoint_FF_best', self.initial_params + 0.25)
    self.assert_saved_params('checkpoint_FF_trial_1', self.initial_params + 0.25)
    self.assert_saved_params('checkpoint_FF_trial_2', self.initial_params + 0.5)

  def test_best_does_not_regress_on_later_trial_or_nonfinite_loss(self):
    checkpoint = self.make_checkpoint('best')
    checkpoint(1, 0, self.initial_params, 5.0)
    checkpoint(2, 0, self.initial_params + 0.5, 8.0)
    checkpoint(2, 1, self.initial_params + 1.0, float('nan'))
    self.assert_saved_params('checkpoint_FF_best', self.initial_params)
    self.assertEqual([path.name for path in self.output.iterdir()], ['checkpoint_FF_best'])

  def test_interrupted_write_preserves_previous_checkpoint(self):
    checkpoint = self.make_checkpoint('best')
    checkpoint(1, 0, self.initial_params, 10.0)
    previous = (self.output / 'checkpoint_FF_best').read_bytes()

    def interrupt_write(initial, destination, force_field):
      Path(destination).write_text('partial output')
      raise KeyboardInterrupt()

    with mock.patch('jaxreaxff.checkpoint.parse_and_save_force_field',
                    side_effect=interrupt_write):
      with self.assertRaises(KeyboardInterrupt):
        checkpoint(1, 1, self.initial_params + 0.25, 5.0)
    self.assertEqual((self.output / 'checkpoint_FF_best').read_bytes(), previous)
    self.assertEqual(checkpoint.best_loss, 10.0)
    self.assertEqual(list(self.output.glob('*.tmp')), [])

  def test_optimizer_interruption_keeps_completed_improvement(self):
    checkpoint = self.make_checkpoint('best')
    loss = mock.Mock(side_effect=[(10.0, onp.zeros(1)), (5.0, onp.zeros(1))])
    improved = self.initial_params + 0.25
    first_step = SimpleNamespace(x=improved, fun=5.0, nfev=1)
    with mock.patch.object(optimizer, 'minimize',
                           side_effect=[first_step, KeyboardInterrupt()]):
      with self.assertRaises(KeyboardInterrupt):
        optimizer.train_FF(
            self.initial_params, self.indices, [(0.0, 100.0)], self.force_field,
            [], [], None, None, 2, False, 'L-BFGS-B', {},
            {'perc_err_change_thr': 0.0}, loss, None, None,
            checkpoint_callback=lambda iteration, params, value:
                checkpoint(1, iteration, params, value))
    self.assert_saved_params('checkpoint_FF_best', improved)


if __name__ == '__main__':
  unittest.main()
