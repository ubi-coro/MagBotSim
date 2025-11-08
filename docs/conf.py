# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import inspect
import os
import sys
from pathlib import Path

import magbotsim

# sphinx-autobuild: source code directory
sys.path.insert(0, os.path.abspath('..'))

# version
with open(os.path.join(os.path.dirname(__file__), '../magbotsim', '__init__.py')) as f:
    content_str = f.read()
    version_start_idx = content_str.find('__version__') + len('__version__ = ') + 1
    version_stop_idx = version_start_idx + content_str[version_start_idx:].find('\n')
    __version__ = content_str[version_start_idx : version_stop_idx - 1]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MagBotSim'
year = datetime.date.today().year
if year == 2024:
    copyright = '2024, Lara Bergmann'
else:
    copyright = f'2024-{year}, Lara Bergmann'
author = 'Lara Bergmann'
release = f'v{__version__}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # include documentation from docstrings
    'sphinxcontrib.spelling',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
    'sphinxcontrib.video',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

spelling_lang = 'en_US'
spelling_show_suggestions = True
spelling_warning = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['css/custom_theme.css']
html_title = 'MagBotSim\nDocumentation'
html_favicon = '_static/favicon.png'

html_theme_options = {
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/ubi-coro/MagBotSim',
            'icon': 'fa-brands fa-github',
        },
    ],
    'logo': {
        'image_light': '_static/logo-light.png',
        'image_dark': '_static/logo-dark.png',
    },
    'primary_sidebar_end': ['sidebar-ethical-ads'],
    'show_toc_level': 1,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'secondary_sidebar_items': ['page-toc'],
    'navbar_center': ['navbar-nav'],
    'navbar_persistent': ['search-button'],
    'navbar_align': 'right',
    'pygment_light_style': 'default',
    'pygment_dark_style': 'monokai',
}


def skip_private_callbacks(app, what, name, obj, skip, options):
    # print docstrings for private methods whose names end with 'callback'
    if what != 'class':
        return None
    is_private = name.startswith('_') and not (name.startswith('__') and name.endswith('__'))
    is_callback = name.endswith('callback')
    is_method = inspect.isroutine(obj)

    if skip and is_private and is_callback and is_method:
        return False

    # None -> default decision
    return None


def list_stl_files(app):
    stl_dir = Path(magbotsim.__file__).parent.joinpath('assets', 'meshes', 'mover_and_bumper')
    stl_files = sorted(f.stem for f in stl_dir.glob('*.stl'))

    beckhoff_mover_files = [stl_file for stl_file in stl_files if stl_file.startswith('beckhoff') and stl_file.endswith('_mover')]
    beckhoff_bumper_files = [stl_file for stl_file in stl_files if stl_file.startswith('beckhoff') and stl_file.endswith('_bumper')]
    planar_motor_files = [stl_file for stl_file in stl_files if stl_file.startswith('planar_motor')]

    with open(Path(app.srcdir) / 'stl_list.rst', 'w') as f:
        f.write('- Beckhoff XPlanar Mover Files\n')
        for name in beckhoff_mover_files:
            f.write(f'    - ``{name}``\n')

        f.write('\n- Beckhoff XPlanar Bumper Files\n')
        for name in beckhoff_bumper_files:
            f.write(f'    - ``{name}``\n')

        f.write('\n- Planar Motor XBot Files\n')
        for name in planar_motor_files:
            f.write(f'    - ``{name}``\n')


def setup(app):
    app.connect('autodoc-skip-member', skip_private_callbacks)
    app.connect('builder-inited', list_stl_files)
