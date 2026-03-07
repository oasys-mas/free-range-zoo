import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from free_range_zoo.envs.cybersecurity.env.v0.cybersecurity import raw_env as cybersecurity_raw_env
from free_range_zoo.envs.rideshare.env.v0.rideshare import raw_env as rideshare_raw_env
from free_range_zoo.envs.wildfire.env.v0.wildfire import raw_env as wildfire_raw_env
from free_range_zoo.utils.yaml_configs import load_environment, load_yaml, write_environment
from free_range_zoo.wrappers.action_task import ActionTaskMappingWrapperModifier
from free_range_zoo.wrappers.space_validator import ActionSpaceValidatorModifier
from free_range_zoo.wrappers.utils import list_wrappers
from free_range_zoo.wrappers.wrapper_util import shared_wrapper
from tests.utils.cybersecurity_configs import non_stochastic as cybersecurity_non_stochastic
from tests.utils.rideshare_configs import non_stochastic as rideshare_non_stochastic
from tests.utils.wildfire_configs import non_stochastic as wildfire_non_stochastic

sys.path.append('.')


class _NamedStringIO(io.StringIO):
    """StringIO wrapper with a stable name attribute for YAML loader compatibility."""

    def __init__(self, initial_value: str, name: str):
        super().__init__(initial_value)
        self.name = name


class _WriteBackStringIO(_NamedStringIO):
    """Write-mode in-memory file that stores contents back into the file store on close."""

    def __init__(self, store: dict[str, str], name: str):
        super().__init__("", name)
        self._store = store

    def close(self) -> None:
        if not self.closed:
            self._store[self.name] = self.getvalue()
        super().close()


class _InMemoryFileStore:
    """Simple path-keyed in-memory filesystem for text read/write operations."""

    def __init__(self):
        self._files: dict[str, str] = {}

    def open(self, path_obj, *args, **kwargs):
        mode = kwargs.get('mode', args[0] if args else 'r')

        # Normalize path strings (e.g., "./a.yaml" and "a.yaml") to the same key.
        path_key = str(Path(path_obj))

        if 'r' in mode:
            if path_key not in self._files:
                raise FileNotFoundError(path_key)
            return _NamedStringIO(self._files[path_key], path_key)

        if 'w' in mode:
            return _WriteBackStringIO(self._files, path_key)

        raise ValueError(f"Unsupported open mode for in-memory file store: {mode}")


class _YamlRoundTripBase(unittest.TestCase):

    def _build_wrapped_env(self):
        env = self.env_class(
            parallel_envs=2,
            max_steps=15,
            configuration=self.configuration,
            device=self.device,
        )

        # Use two wrappers to test wrapper recon.
        env = shared_wrapper(env, ActionSpaceValidatorModifier, allow_flexible_task_tags=False)
        env = shared_wrapper(env, ActionTaskMappingWrapperModifier)
        return env

    def _assert_wrappers_reinitialized(self, original_env, loaded_env) -> None:
        original_wrappers = list_wrappers(original_env, return_modifiers=True)
        loaded_wrappers = list_wrappers(loaded_env, return_modifiers=True)

        self.assertEqual(
            len(original_wrappers),
            len(loaded_wrappers),
            f"Wrapper count mismatch after round-trip for {self.env_name}",
        )

        for original_modifier_map, loaded_modifier_map in zip(original_wrappers, loaded_wrappers):
            self.assertEqual(
                set(original_modifier_map.keys()),
                set(loaded_modifier_map.keys()),
                f"Wrapped agent keys changed after round-trip for {self.env_name}",
            )

            first_agent = next(iter(original_modifier_map.keys()))
            original_modifier = original_modifier_map[first_agent]
            loaded_modifier = loaded_modifier_map[first_agent]

            self.assertEqual(
                type(original_modifier),
                type(loaded_modifier),
                f"Wrapper type mismatch after round-trip for {self.env_name}",
            )

            self.assertEqual(
                original_modifier.__dict__.get('allow_flexible_task_tags'),
                loaded_modifier.__dict__.get('allow_flexible_task_tags'),
                f"Wrapper parameters were not reconstructed for {self.env_name}",
            )

    def _round_trip_environment(self) -> None:
        original_env = self._build_wrapped_env()

        original_yaml = Path(f"./{self.env_name}_original.yaml")
        loaded_yaml = Path(f"./{self.env_name}_loaded.yaml")

        # Use a plain function for monkeypatching Path.open so descriptor binding stays correct.
        def patched_path_open(path_obj, *args, **kwargs):
            return self.file_store.open(path_obj, *args, **kwargs)

        with patch('free_range_zoo.utils.yaml_configs.Path.open', new=patched_path_open):
            write_environment(original_env, original_yaml)
            loaded_env = load_environment(original_yaml)
            write_environment(loaded_env, loaded_yaml)

            original_dict = load_yaml(original_yaml)
            loaded_dict = load_yaml(loaded_yaml)

        self._assert_wrappers_reinitialized(original_env, loaded_env)

        self.assertEqual(
            original_dict,
            loaded_dict,
            f"YAML round-trip did not preserve environment definition for {self.env_name}",
        )


class TestWildfireYamlRoundTrip(_YamlRoundTripBase):

    def setUp(self) -> None:
        self.device = torch.device('cpu')
        self.file_store = _InMemoryFileStore()
        self.env_class = wildfire_raw_env
        self.configuration = wildfire_non_stochastic()
        self.env_name = 'wildfire'

    def test_wildfire_yaml_round_trip(self) -> None:
        self._round_trip_environment()


class TestRideshareYamlRoundTrip(_YamlRoundTripBase):

    def setUp(self) -> None:
        self.device = torch.device('cpu')
        self.file_store = _InMemoryFileStore()
        self.env_class = rideshare_raw_env
        self.configuration = rideshare_non_stochastic()
        self.env_name = 'rideshare'

    def test_rideshare_yaml_round_trip(self) -> None:
        self._round_trip_environment()


class TestCybersecurityYamlRoundTrip(_YamlRoundTripBase):

    def setUp(self) -> None:
        self.device = torch.device('cpu')
        self.file_store = _InMemoryFileStore()
        self.env_class = cybersecurity_raw_env
        self.configuration = cybersecurity_non_stochastic()
        self.env_name = 'cybersecurity'

    def test_cybersecurity_yaml_round_trip(self) -> None:
        self._round_trip_environment()


if __name__ == '__main__':
    unittest.main()