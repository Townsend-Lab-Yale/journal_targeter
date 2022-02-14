import os
import yaml
import pickle
import shutil
import logging
import unicodedata
from typing import Union, Dict

from . import helpers
from .paths import DEMO_DIR
from .plot import get_bokeh_components
from .mapping import run_queries


_logger = logging.getLogger(__name__)


def get_demo_data_with_prefix(file_prefix) -> Union[Dict, None]:
    """Load data dictionary for specified demo name prefix.

    Args:
        file_prefix (str): Filename without extension, e.g. prefix.yaml.

    Returns:
        Dictionary containing title, abstract, ris_name, j, a, jf, af, refs_df,
            bokeh_js, bokeh_divs.
    """
    pickle_path = os.path.join(DEMO_DIR, f'{file_prefix}.pickle')
    if not os.path.exists(pickle_path):
        return
    with open(pickle_path, 'rb') as infile:
        data = pickle.load(infile)
    return data


def update_demo_plot(file_prefix, use_jane_tables=True, pref_metric=None):
    """Update bokeh plot js and html for specified demo file.

    Plot components can become out of date. This function updates them.
    Does not alter refs_df table of citations matched to journals.
    """
    pickle_path = os.path.join(DEMO_DIR, f'{file_prefix}.pickle')
    with open(pickle_path, 'rb') as infile:
        data = pickle.load(infile)
    refs_df = data['refs_df']
    if use_jane_tables:
        jf = data['jf']
        af = data['af']
    else:
        jf, af, _ = run_queries(data['title'], query_abstract=data['abstract'],
                                refs_df=refs_df)
        data.update({'jf': jf,
                     'af': af})
    js, divs = get_bokeh_components(jf, af, refs_df, pref_metric=pref_metric)
    data.update({'bokeh_js': js,
                 'bokeh_divs': divs
                 })
    with open(pickle_path, 'wb') as out:
        pickle.dump(data, out)
    _logger.info(f"Updated demo data written to '{pickle_path}'.")
    return pickle_path


def create_demo_data_from_yaml(yaml_path, ris_path, prefix=None, save_yaml=True):
    """Use yaml file with title + abstract, ris_path to create demo example."""
    demo_dict = helpers.get_queries_from_yaml(yaml_path)
    if not os.path.exists(ris_path):
        raise FileNotFoundError('Invalid query YAML path.')
    ris_name = os.path.basename(ris_path)
    demo_dict.update({'ris_name': ris_name})
    dest_ris_path = os.path.join(DEMO_DIR, ris_name)
    if os.path.exists(dest_ris_path):
        _logger.info("Using existing ris file in demo directory.")
    else:
        shutil.copy2(ris_path, dest_ris_path)
        _logger.info("Copied ris path to demo directory.")
    if prefix is None:
        yaml_basename = os.path.basename(yaml_path)
        prefix = os.path.splitext(yaml_basename)[0]
    _logger.info(f"Using prefix '{prefix}'.")
    _create_demo_data(file_prefix=prefix, save_yaml=save_yaml, **demo_dict)


def create_demo_data_from_args(title=None, abstract=None, ris_path=None,
                               prefix=None, save_yaml=True, overwrite_ris=False):
    """Create demo data from title + abstract, ris_path."""
    if not os.path.exists(ris_path):
        raise FileNotFoundError('Invalid query YAML path.')
    if None in {title, abstract, prefix}:
        raise ValueError('Title, abstract and prefix are required.')
    demo_dict = {'title': title, 'abstract': abstract}
    if ris_path is not None:
        ris_name = os.path.basename(ris_path)
        demo_dict.update({'ris_name': ris_name})
        dest_ris_path = os.path.join(DEMO_DIR, ris_name)
        if os.path.exists(dest_ris_path) and not overwrite_ris:
            _logger.info("Using existing ris file in demo directory.")
        else:
            shutil.copy2(ris_path, dest_ris_path)
            _logger.info("Copied ris path to demo directory.")
    else:
        _logger.info("Skipping citation as no RIS file supplied.")
    _logger.info(f"Using prefix '{prefix}'.")
    _create_demo_data(file_prefix=prefix, save_yaml=save_yaml, **demo_dict)


def init_demo(prefix, overwrite=True):
    """Create demo pickle file from inputs YAML and RIS file in demo dir."""
    yaml_path = os.path.join(DEMO_DIR, f'{prefix}.yaml')
    ris_path = os.path.join(DEMO_DIR, f'{prefix}.ris')
    output_path = os.path.join(DEMO_DIR, f'{prefix}.pickle')
    if not overwrite and os.path.exists(output_path) and \
            helpers.pickle_seems_ok(output_path):
        return
    create_demo_data_from_yaml(yaml_path, ris_path, prefix=prefix, save_yaml=False)


def _create_demo_data(title=None, abstract=None, ris_name=None,
                      file_prefix=None, save_yaml=True):
    """Create inputs YAML file and data pickle for query.

    Args:
        title (str): Title query.
        abstract (str): Abstract query.
        ris_name (str): filename of RIS file in DEMO_DIR.
        file_prefix (str): prefix for output files, for e.g. prefix.yaml, prefix.pickle.
        save_yaml (bool): save yaml file with title, abstract, ris_name.

    Returns:
        output paths ([yaml_path, pickle_path]): saved file paths.
    """
    if file_prefix is None:
        raise ValueError(f"file_prefix is required to generate {file_prefix}.yaml")
    yaml_filename = f"{file_prefix}.yaml"
    if save_yaml:
        _save_inputs_yaml(title=title, abstract=abstract, ris_name=ris_name,
                          yaml_filename=yaml_filename)
    _build_demo_pickle(yaml_filename)


def _build_demo_pickle(yaml_filename):
    """Build dataframes of Jane title+abstract lookups and reference parsing.

    Args:
        yaml_filename: file basename in demo directory with input data,
            inc title, abstract, ris_name.

    Returns:
        pickle_basename (str): pickle filename with processed data tables.
    """

    yaml_path = os.path.join(DEMO_DIR, yaml_filename)
    with open(yaml_path) as infile:
        demo = yaml.load(infile, yaml.SafeLoader)
    ris_path = os.path.join(DEMO_DIR, demo['ris_name'])

    jf, af, refs_df = run_queries(
        query_title=demo['title'], query_abstract=demo['abstract'],
        ris_path=ris_path)
    bokeh_js, bokeh_divs = get_bokeh_components(jf, af, refs_df)

    data = {
        'title': demo['title'],
        'abstract': demo['abstract'],
        'ris_name': demo['ris_name'],
        'jf': jf,
        'af': af,
        'refs_df': refs_df,
        'bokeh_js': bokeh_js,
        'bokeh_divs': bokeh_divs,
    }
    prefix = os.path.splitext(yaml_filename)[0]
    pickle_path = os.path.join(DEMO_DIR, f'{prefix}.pickle')
    with open(pickle_path, 'wb') as out:
        pickle.dump(data, out)
    _logger.info(f"Pickled data written to '{pickle_path}'.")
    return pickle_path


def _save_inputs_yaml(title=None, abstract=None, ris_name=None,
                      yaml_filename=None):
    """Save reference inputs to YAML file in demo directory.

    Args:
        title (str): Title query.
        abstract (str): Abstract query.
        ris_name (str): filename of RIS file in DEMO_DIR.
        yaml_filename: yaml filename for output.

    Returns:
        inputs_path (str): full path to inputs yaml file.
    """

    inputs_path = os.path.join(DEMO_DIR, yaml_filename)
    abstract = unicodedata.normalize('NFKC', abstract)
    data = {
        'title': title,
        'abstract': abstract,
        'ris_name': ris_name,
    }

    with open(inputs_path, 'w') as out:
        yaml.dump(data, out, sort_keys=False)
    _logger.info(f"Inputs yaml written to '{inputs_path}'.")
    return inputs_path
