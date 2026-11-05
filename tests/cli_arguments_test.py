import contextlib
import io
import unittest
from unittest import mock

from jaxreaxff import driver
from jaxreaxff.driver import create_parser


class ValidationArgumentsTest(unittest.TestCase):
  def test_validation_defaults_to_disabled(self):
    self.assertIs(create_parser().parse_args([]).use_valid, False)

  def test_explicit_validation_values(self):
    for value, expected in [('True', True), ('False', False),
                            ('true', True), ('false', False),
                            ('TRUE', True), ('FALSE', False)]:
      with self.subTest(value=value):
        args = create_parser().parse_args(['--use_valid', value])
        self.assertIs(args.use_valid, expected)

  def test_invalid_validation_value_is_rejected(self):
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors):
      with self.assertRaises(SystemExit) as raised:
        create_parser().parse_args(['--use_valid', 'invalid'])
    self.assertEqual(raised.exception.code, 2)
    self.assertIn('Expected True or False', errors.getvalue())


class SaveOptionArgumentsTest(unittest.TestCase):
  def test_driver_preserves_save_option(self):
    for arguments, expected in [([], 'best'),
                                (['--save_opt', 'best'], 'best'),
                                (['--save_opt', 'all'], 'all')]:
      with self.subTest(arguments=arguments):
        parser = create_parser()
        args = parser.parse_args(arguments)
        with mock.patch.object(driver, 'create_parser', return_value=parser), \
             mock.patch.object(parser, 'parse_args', return_value=args), \
             mock.patch.object(driver.jax, 'default_backend', side_effect=RuntimeError('stop before computation')):
          with self.assertRaisesRegex(RuntimeError, 'stop before computation'):
            driver.main()
        self.assertEqual(args.save_opt, expected)


if __name__ == '__main__':
  unittest.main()
