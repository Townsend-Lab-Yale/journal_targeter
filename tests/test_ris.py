import pathlib

import pytest

from journal_targeter.ref_loading import identify_user_references, BadRisException
from journal_targeter.reference import init_reference_data_from_cache

init_reference_data_from_cache()


def test_empty_ris():
    empty_path = get_ris_path('empty.ris')
    with pytest.raises(BadRisException):
        identify_user_references(empty_path)


# def test_missing_journal_ris():
#     for file_name in ['no_journal_one.ris', 'no_journal_two.ris']:
#         path = get_ris_path(file_name)
#         with pytest.raises(BadRisException):
#             identify_user_references(path)


def test_misc_ris():
    for file_name in ['missing_type.ris']:
        path = get_ris_path(file_name)
        with pytest.raises(BadRisException):
            identify_user_references(path)


def test_valid_ris():
    for file_name in ['endnote_export.ris']:
        path = get_ris_path(file_name)
        refs_df = identify_user_references(path)
        assert(len(refs_df)), "No records identified."
        assert(refs_df.uid.isnull().sum() == 0), "Some journals not matched."


def get_ris_path(basename):
    return pathlib.Path(__file__).parent.joinpath(f'data/{basename}')
