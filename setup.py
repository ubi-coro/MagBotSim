import os
import pathlib

from setuptools import setup

CWD = pathlib.Path(__file__).absolute().parent

with open(os.path.join(CWD, 'magbotsim', '__init__.py')) as f:
    content_str = f.read()
    version_start_idx = content_str.find('__version__') + len('__version__ = ') + 1
    version_stop_idx = version_start_idx + content_str[version_start_idx:].find('\n')
    __version__ = content_str[version_start_idx : version_stop_idx - 1]

setup(
    name='magbotsim',
    version=__version__,
)
