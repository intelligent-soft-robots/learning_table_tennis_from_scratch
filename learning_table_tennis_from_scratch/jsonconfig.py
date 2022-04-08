import os
import json
import typing
import pathlib

class TooManyFilesError(Exception):
    def __init__(self, files):
        super().__init__()
        self._files = files

    def __str__(self):
        return str("found too many files: {}".format(",".join(self._files)))


def find_json_file(search_directory: str) -> str:
    """Search the given directory for a JSON file.

    The directory is expected to contain a single JSON file, recognised by its
    file extension (".json").
    If the directory contains no or multiple JSON files, an exception is
    raised.

    Args:
        search_directory: Directory in which the JSON file is searched (search
            is non-recursive, i.e. subdirectories are not considered).

    Returns:
        Path to the JSON file.

    Raises:
        FileNotFoundError:  If search_directory does not contain a JSON file.
        TooManyFilesError:  If search_directory contains more than one JSON
            files.
    """
    # all files in search_directory
    files = [
        f
        for f in os.listdir(search_directory)
        if os.path.isfile(os.path.join(search_directory, f))
    ]

    # json files
    json_files = [f for f in files if f.endswith(".json")]

    # ! no json file
    if not json_files:
        raise FileNotFoundError("failed to find a json file in the current directory")

    # too many json files !
    if len(json_files) != 1:
        raise TooManyFilesError(json_files)

    # the config file used
    return json_files[0]


def get_json_config(
    expected_keys: typing.List[str] = [], config_file=None
) -> typing.Dict[str, str]:
    """
    Searches for a json file in the current folder
    and returns the corresponding dict.
    It assumes values of the dict are either relative or
    absolute path of existing files. If not, a FileNotFoundError
    is thrown.
    Takes an optional list of expected keys, and a
    KeyError will be raised if the json file does not contain
    all specified keys.
    Raises a FileNotFoundError if no json file is found.
    Raises a TooManyFilesError if more than one file is found.
    Raises a JSONDecodeError if json fails to parse the file.
    """

    # current folder
    current_folder = os.getcwd()

    # If no config file is explicitly given, search in the current folder
    if not config_file:
        config_file = find_json_file(current_folder)

    # reading the file
    with open(config_file) as f:
        conf = json.load(f)

    # checking all expected keys are there
    for key in expected_keys:
        if key not in conf.keys():
            raise KeyError(
                "{} does not have the required key {}".format(config_file, key)
            )

    # checking all values are existing files.
    # if exist as relative path, fix to absolute path.
    # if does not exist, raise error
    def _exists(path):
        if path.startswith('~'):
            path = str(pathlib.Path.home())+path[1:]
        fixed_path = os.path.join(current_folder, path)
        if os.path.isfile(fixed_path):
            return fixed_path
        if os.path.isfile(path):
            return path
        return None

    fixed_conf = {}
    for key, value in conf.items():
        fixed_value = _exists(value)
        if not fixed_value:
            raise FileNotFoundError("{} (for key: {})".format(value,key))
        fixed_conf[key] = fixed_value

    # done
    return fixed_conf
