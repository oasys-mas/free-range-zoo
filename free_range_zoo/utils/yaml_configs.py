import copy
import operator
import importlib
from typing import Any, Type
from pathlib import Path
from functools import reduce
from dataclasses import asdict, fields, MISSING

import yaml
import attrs
import numpy as np
import ruamel.yaml as ry
from attrs import Attribute, define
from yaml.nodes import ScalarNode
from yaml.composer import Composer
from yaml.resolver import BaseResolver

import torch

from free_range_zoo.wrappers.wrapper_util import shared_wrapper
from free_range_zoo.wrappers.utils import list_wrappers, unwrap
from free_range_zoo.envs._base.v0.env import BatchedAECEnv
from free_range_zoo.envs._base.configuration import Configuration
#?Load all environments and configurations into subclass registry
from free_range_zoo.utils.all_modules import all_environments


#?Adapting YAML loading and validation from H2Integrate
#?https://github.com/RHammond2/H2Integrate/blob/33bc7316730c873b2878fed10915171e7d87f8b2/h2integrate/core/utilities.py
def merge_shared_inputs(config, input_type):
    """
    Merges two dictionaries from a configuration object and resolves potential conflicts.

    This function combines the dictionaries associated with `shared_parameters` and
    `performance_parameters`, `cost_parameters`, or `finance_parameters` in the provided
    `config` dictionary. If both dictionaries contain the same keys,
    a ValueError is raised to prevent duplicate parameter definitions.

    Parameters:
        config (dict): A dictionary containing configuration data. It must include keys
                       like `shared_parameters` and `{input_type}_parameters`.
        input_type (str): The type of input parameters to merge. Valid values are
                          'performance', 'control', 'cost', or 'finance'.

    Returns:
        dict: A merged dictionary containing parameters from both `shared_parameters`
              and `{input_type}_parameters`. If one of the dictionaries is missing,
              the function returns the existing dictionary.

    Raises:
        ValueError: If duplicate keys are found in `shared_parameters` and
                    `{input_type}_parameters`.
    """

    if f"{input_type}_parameters" in config.keys() and "shared_parameters" in config.keys():
        common_keys = config[f"{input_type}_parameters"].keys() & config["shared_parameters"].keys()
        if common_keys:
            raise ValueError(f"Duplicate parameters found: {', '.join(common_keys)}. "
                             f"Please define parameters only once in the shared and {input_type} dictionaries.")
        return {**config[f"{input_type}_parameters"], **config["shared_parameters"]}
    elif "shared_parameters" not in config.keys():
        return config[f"{input_type}_parameters"]
    else:
        return config["shared_parameters"]


@define(kw_only=True)
class BaseConfig:
    """
    A Mixin class to allow for kwargs overloading when a data class doesn't
    have a specific parameter defined. This allows passing of larger dictionaries
    to a data class without throwing an error.
    """

    @classmethod
    def from_dict(cls, data: dict, strict=True, additional_cls_name: str | None = None):
        """Maps a data dictionary to an `attr`-defined class.

        Args:
            data : dict
                The data dictionary to be mapped.
            strict: bool
                A flag enabling strict parameter processing, meaning that no extra parameters
                    may be passed in or an AttributeError will be raised.
            additional_cls_name (str | None): The name of the model class creating the configuration
                data class. Provides an easier to diagnose error message for end users when
                the class name is provided.
        Returns:
            cls
                The `attr`-defined class.
        """
        # Check for any inputs that aren't part of the class definition
        if strict is True:
            class_attr_names = [a.name for a in cls.__attrs_attrs__]
            extra_args = [d for d in data if d not in class_attr_names]
            if len(extra_args):
                if additional_cls_name is not None:
                    msg = (f"{additional_cls_name} setup failed as a result of {cls.__name__}"
                           f" receiving extraneous inputs: {extra_args}")
                else:
                    msg = (f"The initialization for {cls.__name__} was given extraneous "
                           f"inputs: {extra_args}")
                raise AttributeError(msg)

        kwargs = {a.name: data[a.name] for a in cls.__attrs_attrs__ if a.name in data and a.init}

        # Map the inputs must be provided: 1) must be initialized, 2) no default value defined
        required_inputs = [a.name for a in cls.__attrs_attrs__ if a.init and a.default is attrs.NOTHING]
        undefined = sorted(set(required_inputs) - set(kwargs))

        if undefined:
            if additional_cls_name is not None:
                msg = (f"{additional_cls_name} setup failed as a result of {cls.__name__}"
                       f" missing the following inputs: {undefined}")
            else:
                msg = (f"The class definition for {cls.__name__} is missing the following inputs: "
                       f"{undefined}")
            raise AttributeError(msg)
        return cls(**kwargs)

    def as_dict(self) -> dict:
        """Creates a JSON and YAML friendly dictionary that can be save for future reloading.
        This dictionary will contain only `Python` types that can later be converted to their
        proper `Turbine` formats.

        Returns:
            dict: All key, value pairs required for class re-creation.
        """
        return attrs.asdict(self, filter=attr_filter, value_serializer=attr_serializer)


def attr_serializer(inst: type, field: Attribute, value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def attr_filter(inst: Attribute, value: Any) -> bool:
    if inst.init is False:
        return False
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
    return True


def check_pysam_input_params(user_dict, pysam_options):
    """Checks for different values provided in two dictionaries that have the general format::

        value = input_dict[group][group_param]

    Args:
        user_dict (dict): top-level performance model inputs formatted to align with
            the corresponding PySAM module.
        pysam_options (dict): additional PySAM module options.

    Raises:
        ValueError: if there are two different values provided for the same key.

    """
    for group, group_params in user_dict.items():
        if group in pysam_options:
            for key in group_params.keys():
                if key in pysam_options:
                    if pysam_options[group][key] != user_dict[group][key]:
                        msg = (f"Inconsistent values provided for parameter {key} in {group} Group."
                               f"pysam_options has value of {pysam_options[group][key]} "
                               f"but user also specified value of {user_dict[group][key]}. ")
                        raise ValueError(msg)
    return


def dict_to_yaml_formatting(orig_dict):
    """Recursive method to convert arrays to lists and numerical entries to floats.
    This is primarily used before writing a dictionary to a YAML file to ensure
    proper output formatting.

    Args:
        orig_dict (dict): input dictionary

    Returns:
        dict: input dictionary with reformatted values.
    """
    for key, val in orig_dict.items():
        if isinstance(val, dict):
            tmp = dict_to_yaml_formatting(orig_dict.get(key, {}))
            orig_dict[key] = tmp
        else:
            if isinstance(key, list):
                for i, k in enumerate(key):
                    if isinstance(orig_dict[k], str | bool | int):
                        orig_dict[k] = orig_dict.get(k, []) + val[i]
                    elif isinstance(orig_dict[k], list | np.ndarray):
                        orig_dict[k] = np.array(val, dtype=float).tolist()
                    else:
                        orig_dict[k] = float(val[i])
            elif isinstance(key, str):
                if isinstance(orig_dict[key], str | bool | int):
                    continue
                if orig_dict[key] is None:
                    continue
                if isinstance(orig_dict[key], list | np.ndarray):
                    if any(isinstance(v, dict) for v in val):
                        for vii, v in enumerate(val):
                            if isinstance(v, dict):
                                new_val = dict_to_yaml_formatting(v)
                            else:
                                new_val = v if isinstance(v, str | bool | int) else float(v)
                            orig_dict[key][vii] = new_val
                    else:
                        new_val = [v if isinstance(v, str | bool | int) else float(v) for v in val]
                        orig_dict[key] = new_val
                else:
                    orig_dict[key] = float(val)
    return orig_dict


def get_path(path: str | Path) -> Path:
    """
    Convert a string or Path object to an absolute Path object, prioritizing different locations.

    This function attempts to find the existence of a path in the following order:
    1. As an absolute path.
    2. Relative to the current working directory.
    3. Relative to the H2Integrate package.

    Args:
        path (str | Path): The input path, either as a string or a Path object.

    Raises:
        FileNotFoundError: If the path is not found in any of the locations.

    Returns:
        Path: The absolute path to the file.
    """
    # Store the original path for reference in error messages.
    original_path = path

    # If the input is a string, convert it to a Path object.
    if isinstance(path, str):
        path = Path(path)

    # Check if the path exists as an absolute path.
    if path.exists():
        return path.absolute()

    # If not, try finding the path relative to the current working directory.
    relative_path = Path.cwd() / path
    path = relative_path

    # If the path still doesn't exist, attempt to find it relative to the H2Integrate package.
    if path.exists():
        return path.absolute()

    # Determine the path relative to the H2Integrate package.
    h2i_based_path = ROOT_DIR.parent / Path(original_path)

    path = h2i_based_path

    if path.exists():
        return path.absolute()

    # If the path still doesn't exist in any of the prioritized locations, raise an error.
    raise FileNotFoundError(f"File not found in absolute path: {original_path}, relative path: "
                            f"{relative_path}, or H2Integrate-based path: "
                            f"{h2i_based_path}")


def find_file(filename: str | Path, root_folder: str | Path | None = None):
    """
    This function attempts to find a filepath matching `filename` from a variety of locations
    in the following order:

    1. Relative to the root_folder (if provided)
    2. Relative to the current working directory.
    3. Relative to the H2Integrate package.
    4. As an absolute path if `filename` is already absolute

    Args:
        filename (str | Path): Input filepath
        root_folder (str | Path, optional): Root directory to search for filename in.
            Defaults to None.

    Raises:
        FileNotFoundError: If the path is not found in any of the locations.

    Returns:
        Path: The absolute path to the file.
    """

    # 1. check for file in the root directory
    files = []
    if root_folder is not None:
        root_folder = Path(root_folder)
        # if the file exists in the root directory, return full path
        if Path(root_folder, filename).exists():
            return Path(root_folder, filename).resolve().absolute()

        # check for files within root directory
        files = list(Path(root_folder).glob(f"**/{filename}"))

        if len(files) == 1:
            return files[0].absolute()
        if len(files) > 1:
            raise FileNotFoundError(f"Found {len(files)} files in the root directory ({root_folder}) that have "
                                    f"filename {filename}")

        filename_no_rel = "/".join(p for p in Path(root_folder, filename).resolve(strict=False).parts
                                   if p not in Path(root_folder).parts)
        files = list(Path(root_folder).glob(f"**/{filename_no_rel}"))
        if len(files) == 1:
            return files[0].absolute()

    # 2. check for file relative to the current working directory
    files_cwd = list(Path.cwd().glob(f"**/{filename}"))
    if len(files_cwd) == 1:
        return files_cwd[0].absolute()

    # 3. check for file relative to the H2Integrate package root
    files_h2i = list(ROOT_DIR.parent.glob(f"**/{filename}"))
    files_h2i = [file for file in files_h2i if "build" not in file.parts]
    if len(files_h2i) == 1:
        return files_h2i[0].absolute()

    # 4. check for as absolute path
    if Path(filename).is_absolute():
        return Path(filename)

    if len(files_cwd) == 0 and len(files_h2i) == 0:
        raise FileNotFoundError(f"Did not find any files matching {filename} in the current working directory "
                                f"{Path.cwd()} or relative to the H2Integrate package {ROOT_DIR.parent}")
    if root_folder is not None and len(files) == 0:
        raise FileNotFoundError(f"Did not find any files matching {filename} in the current working directory "
                                f"{Path.cwd()}, relative to the H2Integrate package {ROOT_DIR.parent}, or relative to "
                                f"the root directory {root_folder}.")
    raise ValueError(f"Cannot find unique file: found {len(files_cwd)} files relative to cwd, "
                     f"{len(files_h2i)} files relative to H2Integrate root directory, "
                     f"{len(files)} files relative to the root folder.")


def remove_numpy(fst_vt: dict) -> dict:
    """
    Recursively converts numpy array elements within a nested dictionary to lists and ensures
    all values are simple types (float, int, dict, bool, str) for writing to a YAML file.

    Args:
        fst_vt (dict): The dictionary to process.

    Returns:
        dict: The processed dictionary with numpy arrays converted to lists
            and unsupported types to simple types.
    """

    def get_dict(vartree, branch):
        return reduce(operator.getitem, branch, vartree)

    # Define conversion dictionary for numpy types
    conversions = {
        np.int_: int,
        np.intc: int,
        np.intp: int,
        np.int8: int,
        np.int16: int,
        np.int32: int,
        np.int64: int,
        np.uint8: int,
        np.uint16: int,
        np.uint32: int,
        np.uint64: int,
        np.single: float,
        np.double: float,
        np.longdouble: float,
        np.csingle: float,
        np.cdouble: float,
        np.float16: float,
        np.float32: float,
        np.float64: float,
        np.complex64: float,
        np.complex128: float,
        np.bool_: bool,
        np.ndarray: lambda x: x.tolist(),
        torch.Tensor: lambda x: x.numpy().tolist()
    }

    def loop_dict(vartree, branch):
        if not isinstance(vartree, dict):
            return fst_vt
        for var in vartree.keys():
            branch_i = copy.copy(branch)
            branch_i.append(var)
            if isinstance(vartree[var], dict):
                loop_dict(vartree[var], branch_i)
            else:
                current_value = get_dict(fst_vt, branch_i[:-1])[branch_i[-1]]
                data_type = type(current_value)
                if data_type in conversions:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = conversions[data_type](current_value)
                elif isinstance(current_value, list | tuple):
                    for i, item in enumerate(current_value):
                        current_value[i] = remove_numpy(item)

    # set fast variables to update values
    loop_dict(fst_vt, [])
    return fst_vt


class DuplicateKeyError(Exception):
    """Exception raised when a duplicate YAML key is found.

    Args:
        message (:obj:str): The duplicate key error message to be displayed.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Loader(yaml.SafeLoader):

    def __init__(self, stream):
        # root is the parent directory of the parent yaml file
        self._root = get_path(Path(stream.name).parent)

        super().__init__(stream)

    def include(self, node):
        filename = find_file(node.value, self._root)

        return load_yaml(filename)

    def compose_node(self, parent, index):
        """Custom implementation to include line numbers that account for all lines, including
        blank spaces that align with user anticipated 1-indexing.
        """
        line = self.line
        node = Composer.compose_node(self, parent, index)
        node.__line__ = line + 1
        return node

    def construct_mapping(self, node, deep=False):
        """Hooks into the ``yaml.SafeLoader.construct_mapping`` routine to create line number
        mappings for all keys and values, which enables duplicate key error handling.

        Two copies of node are created to avoid errors when run through the validation schema as
        the ``__line__{key}`` and ``__line__`` keys in the key and value nodes are not represented
        by the schema, and therefore raise an error during validation.
        """
        numbered_node = copy.deepcopy(node)
        numbered_nodes = []
        for key_node, _ in numbered_node.value:
            shadow_key_node = ScalarNode(tag=BaseResolver.DEFAULT_SCALAR_TAG, value="__line__" + key_node.value)
            shadow_value_node = ScalarNode(tag=BaseResolver.DEFAULT_SCALAR_TAG, value=key_node.__line__)
            numbered_nodes.append((shadow_key_node, shadow_value_node))

        numbered_node.value += numbered_nodes
        mapping = self.check_duplicate_keys(numbered_node, node, deep)
        return mapping

    def check_duplicate_keys(self, numbered_node, node, deep=False):
        """Raises an error for duplicate keys and calls the ``SafeLoader.construct_mapping()``
        routine to create the final dictionary mappings.
        """
        unique_keys = set()
        for key_node, _ in numbered_node.value:
            if ":merge" in key_node.tag:
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in unique_keys:
                raise DuplicateKeyError(f"Duplicate '{key}' key found at line {key_node.__line__}.")
            unique_keys.add(key)

        mapping = super().construct_mapping(node, deep)
        return mapping


Loader.add_constructor("!include", Loader.include)


def load_yaml(filename, loader=Loader) -> dict:
    if isinstance(filename, dict):
        return filename  # filename already yaml dict
    with Path.open(filename) as fid:
        try:
            return yaml.load(fid, loader)
        except DuplicateKeyError as e:
            raise ValueError(f"Duplicate key found in {filename}: {e.message}") from e


def write_yaml(instance: dict, foutput: str, convert_np: bool = True, check_formatting: bool = False) -> None:
    """
    Writes a dictionary to a YAML file using the ruamel.yaml library.

    Args:
        instance (dict): Dictionary to be written to the YAML file.
        foutput (str): Path to the output YAML file.
        convert_np (bool): Whether to convert numpy objects to simple types. Defaults to True.
        check_formatting (bool): Whether to check formatting to convert numpy arrays to lists.
            Defaults to False.

    Returns:
        None
    """

    if convert_np:
        instance = remove_numpy(instance)
    if check_formatting:
        instance = dict_to_yaml_formatting(instance)
    # Write yaml with updated values
    yaml = ry.YAML()
    yaml.default_flow_style = None
    yaml.width = float("inf")
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.allow_unicode = False
    with Path(foutput).open("w", encoding="utf-8") as f:
        yaml.dump(instance, f)


def write_readable_yaml(instance: dict, foutput: str | Path):
    """
    Writes a dictionary to a YAML file using the yaml library.

    Args:
        instance (dict): Dictionary to be written to the YAML file.
        foutput (str | Path): Path to the output YAML file.

    Returns:
        None
    """
    instance = dict_to_yaml_formatting(instance)

    with Path(foutput).open("w", encoding="utf-8") as f:
        yaml.dump(instance, f, sort_keys=False, encoding=None, default_flow_style=False)


#? FRZ loading, validation, and writing using H2Integrate's initial untilities.
default_typecast = {
    torch.BoolTensor: lambda x: torch.tensor(x, dtype=torch.bool),
    torch.FloatTensor: lambda x: torch.tensor(x, dtype=torch.float),
    torch.IntTensor: lambda x: torch.tensor(x, dtype=torch.int),
    torch.DoubleTensor: lambda x: torch.tensor(x, dtype=torch.double),
    torch.LongTensor: lambda x: torch.tensor(x, dtype=torch.long),
    torch.Tensor: lambda x: torch.tensor(x),
    int: int,
    float: float,
    bool: bool
}
default_typecast.update({
    c.__name__ if 'Tensor' not in c.__name__ else 'torch.' + c.__name__: v
    for c,v in default_typecast.items()
})

#parse to find all configuration classes in frz


class ConfigError(Exception):
    """Exception raised for errors in the configuration loading and validation process.

    Args:
        message (:obj:str): The error message to be displayed.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


def alternate_path_colors(line: str) -> str:
    """Color `->` path segments in alternating black/blue ANSI colors."""
    if ":" not in line:
        return line

    path, message = line.split(":", 1)
    if "->" not in path:
        return line

    has_leading_arrow = path.startswith("->")
    segments = [segment for segment in path.split("->") if segment]
    if not segments:
        return line

    black = "\033[30m"
    blue = "\033[34m"
    reset = "\033[0m"

    colored_segments = []
    for index, segment in enumerate(segments):
        color = black if index % 2 == 0 else blue
        colored_segments.append(f"{color}{segment}{reset}")

    colored_path = "->".join(colored_segments)
    if has_leading_arrow:
        colored_path = "->" + colored_path

    return f"{colored_path}:{message}"


def write_config(config: Type[Configuration], foutput: str | Path) -> None:
    """
    Writes a Configuration object to a YAML file.

    Args:
        config (Configuration): The Configuration object to be written to the YAML file.
        foutput (str | Path): The path to the output YAML file.

    Returns:
        None
    """
    config_dict = asdict(config)
    config_dict = {config.__class__.__name__: config_dict}

    write_yaml(config_dict, foutput)


def load_config(config: dict, config_class: Type[Configuration], _config_path: str = '', _name: str = '') -> Configuration:
    """
    Instantiates a free-range-zoo loaded configuration file against a provided Configuration class.

    Args:
        config (dict): The configuration dictionary to validate.
        config_class (Type[Configuration]): The Configuration class to validate against.
        _config_path (str): The path from config->config for nested configs to provide informative debug messages.
        _name (str): The name of the cur parameter be it subconfig or variable
        _module (str): The name of the module of the base config class for diambiguation
    Returns:
        Configuration: An instance of the Configuration class initialized with the validated config.
    """

    warnings = []
    config_params = {}
    _name = f"{config_class.__module__}:{config_class.__name__}" if _name == '' else _name
    _module = config_class.__module__

    try:
        config = config[_name]
    except Exception as e:
        raise ConfigError(str(e))

    for field in fields(config_class):

        name = field.name
        class_name = f'{_module}:{field.type.__name__ if hasattr(field.type, "__name__") else field.type}'

        is_here = name in config.keys()
        is_subconfig = class_name in Configuration.get_subclasses()

        #clean error message to identify missing config parameters
        if not is_here and field.default is MISSING and field.default_factory is MISSING:
            warnings.append(_config_path + "->" + _name + ": " + name + " (missing from config)")
            continue

        if is_subconfig:
            try:
                subconfig = Configuration.get_subclasses()[class_name]
                config_params[name] = load_config(config,
                                                  config_class=subconfig,
                                                  _config_path=_config_path + "->" + _name,
                                                  _name=name)
            except ConfigError as e:
                warnings.append(str(e))
        else:
            try:
                convert_function = config_class.type_caster().get((config_class, name), default_typecast[field.type])
                config_params[name] = convert_function(config[name])

            except Exception as e:
                warnings.append(str(e))

    if len(warnings) > 0:
        colored_warnings = [alternate_path_colors(warning) for warning in warnings]
        raise ConfigError('\n'.join(colored_warnings) if _config_path != '' else 'Configuration failure, failed to intialize:\n' +
                          '\n'.join(colored_warnings))

    return config_class(**config_params)


def load_environment(finput: str | Path, **kwargs) -> BatchedAECEnv:
    """
    Loads a free-range-zoo environment from a YAML file.

    Args:
        finput (str | Path): The path to the input YAML file.
        **kwargs: Additional keyword arguments to overwrite environment initialization parameters.
    Returns:
        BatchedAECEnv: The loaded free-range-zoo environment.
    """
    env_dict = load_yaml(finput)
    env_class = env_dict['env']
    env_init_kwargs = env_dict['init_kwargs']
    env_config_kwargs = env_dict['config']
    env_wrappers = env_dict['wrappers']

    #?Load config
    config_class_name, _ = list(env_config_kwargs.items())[0]
    config_class = Configuration.get_subclasses()[config_class_name]
    config = load_config(env_config_kwargs, config_class)

    #?Load environment
    env_module_name, env_class_name = env_class.rsplit(":", 1)
    env_module = importlib.import_module(env_module_name)
    env_cls = getattr(env_module, env_class_name)
    env = env_cls(configuration=config, **(env_init_kwargs | kwargs))
    env.reset()

    #?Load wrappers
    for wrapper in env_wrappers:
        wrapper_name, wrapper_kwargs = wrapper['name'], wrapper['params']
        wrapper_module_name, wrapper_class_name = wrapper_name.rsplit(":", 1)
        wrapper_module = importlib.import_module(wrapper_module_name)
        wrapper_cls = getattr(wrapper_module, wrapper_class_name)
        env = shared_wrapper(env, wrapper_cls, **wrapper_kwargs)
    return env


def write_environment(env: BatchedAECEnv, foutput: str | Path):
    """
    Write a frz environment, meaning env_v?, wrappers, and configuration, to a YAML file.

    Args:
        env (BatchedAECEnv): The frz environment to be written to the YAML file.
        foutput (str | Path): The path to the output YAML file.
    """

    wrappers = list_wrappers(env, return_modifiers=True)

    #?Extract wrapper parameters for shared wrappers
    ignore = set(['subject_agent', 'cur_obs', 'cur_observation', 'env', 'observe', 'observation_space', 'action_space','return'])
    pruned_params = [{k: {_k: _v for _k, _v in v.__dict__.items() if _k not in ignore} for k, v in e.items()} for e in wrappers]
    wrapper_names = [{k: f'{type(v).__module__}:{type(v).__name__}' for k,v in e.items()} for e in wrappers]

    for wrap_params, wrap in zip(pruned_params, wrapper_names):
        n0 = list(wrap.values())[0]
        assert all([n==n0 for n in wrap.values()]),\
            "All agents must share the same wrappers."

        v0 = list(wrap_params.values())[0]
        assert all([v==v0 for v in wrap_params.values()]),\
            "To make reconstructable wrappers, all agents must share wrapper hyperparameters for now."

    #?assemble wrapper parameters
    wrapper_kwargs = [
        {
            'name': list(wrapper_name.values())[0],
            'params': list(wrapper_param.values())[0]
        }
        for wrapper_name, wrapper_param in zip(reversed(wrapper_names), reversed(pruned_params))
    ]

    #?get env hyperparameters
    unwrapped_env = unwrap(env)
    env_spec_kwargs = unwrapped_env.__init__.__annotations__ 
    init_kwargs = {
        'max_steps': unwrapped_env.max_steps,
        'parallel_envs': unwrapped_env.parallel_envs,
        **{k: getattr(unwrapped_env, k) for k in env_spec_kwargs if hasattr(unwrapped_env, k) and k not in ignore}
    }

    #?get config hyperparameters
    config_kwargs = asdict(unwrapped_env.config)
    config_kwargs = {f"{unwrapped_env.config.__class__.__module__}:{unwrapped_env.config.__class__.__name__}": config_kwargs}

    env_dict = {
        'env': f'{type(unwrapped_env).__module__}:{type(unwrapped_env).__name__}',
        'init_kwargs': init_kwargs,
        'config': config_kwargs,
        'wrappers': wrapper_kwargs
    }

    write_yaml(env_dict, foutput)

