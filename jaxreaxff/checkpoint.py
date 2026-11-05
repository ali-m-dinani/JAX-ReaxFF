import os
from pathlib import Path
import tempfile

import numpy as onp

from jaxreaxff.helper import move_dataclass, parse_and_save_force_field, set_params


class ForceFieldCheckpoint:
  def __init__(self, initial_file, output_folder, force_field, param_indices, save_opt):
    self.initial_file = initial_file
    self.output_folder = Path(output_folder)
    self.force_field = force_field
    self.param_indices = param_indices
    self.save_opt = save_opt
    self.best_loss = float('inf')
    self.output_folder.mkdir(parents=True, exist_ok=True)

  def _save(self, filename, force_field):
    descriptor, temporary_file = tempfile.mkstemp(
        prefix='.checkpoint_', suffix='.tmp', dir=self.output_folder)
    os.close(descriptor)
    try:
      parse_and_save_force_field(self.initial_file, temporary_file, force_field)
      with open(temporary_file, 'rb+') as checkpoint_file:
        os.fsync(checkpoint_file.fileno())
      os.replace(temporary_file, self.output_folder / filename)
    finally:
      if os.path.exists(temporary_file):
        os.unlink(temporary_file)

  def __call__(self, trial, iteration, params, loss):
    if not onp.isfinite(loss):
      return
    if self.save_opt != 'all' and loss >= self.best_loss:
      return
    force_field = set_params(self.force_field, self.param_indices, params)
    force_field = move_dataclass(force_field, onp)
    if loss < self.best_loss:
      self._save('checkpoint_FF_best', force_field)
      self.best_loss = loss
    if self.save_opt == 'all':
      self._save('checkpoint_FF_trial_{}'.format(trial), force_field)
    print('[INFO] Checkpoint saved: trial {}, iteration {}, loss {:.6g}'.format(
        trial, iteration, loss))
