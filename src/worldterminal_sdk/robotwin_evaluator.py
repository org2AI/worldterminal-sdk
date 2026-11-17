"""Explicit observation override for the pinned RoboTwin evaluator.

This is a different launcher, not support for the unmodified standard launcher.
It changes one runtime conversion boundary while retaining the official source file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

from .robotwin import observation

OFFICIAL_REVISION = 'ea8b21121ebb3cd201ff5b3fe361944ac94eda3f'
OFFICIAL_SHA256 = '78821388d58daf8aa65db55c5952df3742486dd93e717ce6a96374a3795dfc7a'


def measured_observation(raw, *, instruction=None, env_idx=0, frequency=30, task_env=None):
    """Use the evaluator's captured RGB and its actual loaded robot metadata."""
    if task_env is None:
        raise ValueError('Measured RoboTwin conversion requires the actual task_env')
    text = instruction or raw.get('language') or ''
    result = observation(task_env, instruction=text, raw=raw)
    result.update(data_format_version='v1.0', instructions=[text] if text else [],
                  env_idx=int(env_idx))
    result['additional_info']['frequency'] = int(frequency)
    return result


def load_evaluator(path):
    """Load the byte-pinned official module without running its __main__ block."""
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != OFFICIAL_SHA256:
        raise ValueError(f'Expected unchanged RoboTwin {OFFICIAL_REVISION} evaluator: {path}')
    name = '_worldterminal_official_robotwin_evaluator'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load RoboTwin evaluator: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_evaluator(path, official_args):
    """Override the named converter, then call official preflight, parser and main."""
    module = load_evaluator(path)
    module.robotwin_obs_to_xpolicylab = measured_observation
    previous_argv = sys.argv
    sys.argv = [str(path), *official_args]
    try:
        importlib.import_module('test_render').Sapien_TEST()
        parsed = module.parse_args()
        if module.parse_bool(parsed.get('eval_batch', False)):
            raise ValueError('This explicit adapter supports sequential eval only; '
                             'spawned batch workers do not inherit the converter override')
        return module.main(parsed)
    finally:
        sys.argv = previous_argv
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != OFFICIAL_SHA256:
            raise RuntimeError('Official evaluator source changed during adapter execution')


def main(argv=None):
    """Separate adapter options from untouched official evaluator arguments."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', required=True, type=Path)
    if '--' not in argv:
        parser.parse_args(argv)  # Preserve ordinary --help behavior.
        parser.error('Separate the official evaluator arguments with --')
    boundary = argv.index('--')
    args = parser.parse_args(argv[:boundary])
    path = args.robotwin_root / 'scripts' / 'eval_policy_xpolicylab.py'
    print('WorldTerminal explicit adapter: overriding robotwin_obs_to_xpolicylab in memory; '
          'official evaluator source is unchanged. Sequential mode only.', file=sys.stderr)
    return run_evaluator(path, argv[boundary + 1:])


if __name__ == '__main__':
    main()
