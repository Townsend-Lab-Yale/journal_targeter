import os
import sys
import yaml
import pickle
import shutil
import logging
import unicodedata

from . import DEMO_DIR
from .plot import get_bokeh_components
from .mapping import run_queries
from .helpers import get_queries_from_yaml


logging.basicConfig(format='%(levelname)s: %(message)s',  # %(asctime)-15s
                    level=logging.INFO, stream=sys.stdout)
_logger = logging.getLogger(__name__)


def get_demo_data(file_prefix):
    """Load data dictionary for specified demo name prefix.

    Args:
        file_prefix (str): Filename without extension, e.g. prefix.yaml.

    Returns:
        dict with keys: title, abstract, ris_name, j, a, jf, af, refs_df,
            bokeh_js, bokeh_icats.
    """
    pickle_path = os.path.join(DEMO_DIR, f'{file_prefix}.pickle')
    with open(pickle_path, 'rb') as infile:
        data = pickle.load(infile)
    return data


def update_demo_plot(file_prefix, use_pickle=True):
    """Update bokeh plot js and html for specified demo file.

    Plot components can become out of date. This function updates them.
    """
    pickle_path = os.path.join(DEMO_DIR, f'{file_prefix}.pickle')
    with open(pickle_path, 'rb') as infile:
        data = pickle.load(infile)
    if use_pickle:
        jf = data['jf']
        af = data['af']
        refs_df = data['refs_df']
    else:
        jf, af, refs_df = run_queries(data['title'],
                                      query_abstract=data['abstract'],
                                      refs_df=data['refs_df'])
        data.update({'jf': jf,
                     'af': af})
    js, divs = get_bokeh_components(jf, af, refs_df)
    data.update({'bokeh_js': js,
                 'bokeh_divs': divs
                 })
    with open(pickle_path, 'wb') as out:
        pickle.dump(data, out)
    _logger.info(f"Updated demo data written to {pickle_path}")
    return pickle_path


def create_demo_data(title=None, abstract=None, ris_name=None,
                     file_prefix=None):
    """Create inputs YAML file and data pickle for query.

    Args:
        title (str): Title query.
        abstract (str): Abstract query.
        ris_name (str): filename of RIS file in DEMO_DIR.
        file_prefix: prefix for output files, for e.g. prefix.yaml, prefix.pickle.

    Returns:
        output paths ([yaml_path, pickle_path]): saved file paths.
    """
    yaml_filename = f"{file_prefix}.yaml"
    _save_inputs_yaml(title=title, abstract=abstract, ris_name=ris_name,
                      yaml_filename=yaml_filename)
    _build_demo_pickle(yaml_filename)


def create_demo_data_from_yaml(yaml_path, ris_path, prefix=None):
    """Use yaml file with title + abstract, ris_path to create demo example."""
    demo_dict = get_queries_from_yaml(yaml_path)
    if not os.path.exists(ris_path):
        raise FileNotFoundError('Invalid query YAML path.')
    ris_name = os.path.basename(ris_path)
    dest_ris_path = os.path.join(DEMO_DIR, ris_name)
    if os.path.exists(dest_ris_path):
        _logger.info("Using existing ris file in demo directory")
    else:
        shutil.copy2(ris_path, dest_ris_path)
        _logger.info("Copied ris path to demo directory.")
    if prefix is None:
        yaml_basename = os.path.basename(yaml_path)
        prefix = os.path.splitext(yaml_basename)[0]
    _logger.info(f"Using prefix '{prefix}'.")
    create_demo_data(file_prefix=prefix, **demo_dict)


def _build_demo_pickle(yaml_filename):
    """Build dataframes of Jane title+abstract lookups and reference parsing.

    Args:
        yaml_filename: file basename in demo directory with input data,
            inc title, abstract, ris_name.

    Returns:
        pickle_basename (str): pickle filename with processed data tables.
    """

    inputs_path = os.path.join(DEMO_DIR, yaml_filename)
    with open(inputs_path) as infile:
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
        'bokeh_icats': bokeh_divs['icats'],
        'bokeh_table': bokeh_divs['table'],
    }
    prefix = os.path.splitext(yaml_filename)[0]
    pickle_path = os.path.join(DEMO_DIR, f'{prefix}.pickle')
    with open(pickle_path, 'wb') as out:
        pickle.dump(data, out)
    _logger.info(f"Pickled data written to {pickle_path}")
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
    _logger.info(f"Inputs yaml written to {inputs_path}")
    return inputs_path
