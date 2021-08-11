import os, json


class TooManyFilesError(Exception):
    def __init__(self, files):
        super.__init__()
        self._files = files

    def __str__(self):
        return str("found too many files: {}".format(",".join(self._files)))


def get_json_config(expected_keys: list = []):
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

    # files in it
    files = [
        f
        for f in os.listdir(current_folder)
        if os.path.isfile(os.path.join(current_folder, f))
    ]

    # json files
    json_files = [f for f in files if f.endswith(".json")]

    # ! no json file
    if not json_files:
        raise FileNotFoundError(
            str("failed to find a json file in the current directory")
        )

    # too many json files !
    if len(json_files) != 1:
        raise TooManyFilesError(json_files)

    # the config file used
    config_file = json_files[0]

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
            raise FileNotFoundError(value)
        fixed_conf[key] = fixed_value

    # done
    return fixed_conf
