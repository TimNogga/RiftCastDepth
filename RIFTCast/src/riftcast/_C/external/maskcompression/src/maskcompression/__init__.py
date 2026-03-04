import sys

required_version = (3, 8)

if sys.version_info[:2] < required_version:
    msg = "%s requires Python %d.%d+" % (__package__, *required_version)
    raise RuntimeError(msg)

del required_version
del sys

import pathlib
import charonload

PROJECT_ROOT_DIRECTORY = pathlib.Path(__file__).parents[2]

VSCODE_STUBS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "typings"


charonload.module_config["_c_maskcompression"] = charonload.Config(
    pathlib.Path(__file__).parent / "_C",
    stubs_directory=VSCODE_STUBS_DIRECTORY,
    verbose=False,
)

from _c_maskcompression import compress, decompress  # noqa: F401
