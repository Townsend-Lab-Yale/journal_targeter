import os
import gzip
import uuid
import pickle


def get_new_pickle_path(parent_dir):
    """Generate a new, unique path for saving pickle data."""
    while True:
        test_name = f"{uuid.uuid4().hex}.pickle.gz"
        full_path = os.path.join(parent_dir, test_name)
        if not os.path.exists(full_path):
            return full_path


def write_tables_to_pickle(j, a, jf, af, refs_df, pickle_path):
    tables = {
        'j': j,
        'a': a,
        'jf': jf,
        'af': af,
        'refs_df': refs_df,
    }
    with gzip.open(pickle_path, 'wb') as out:
        pickle.dump(tables, out)
