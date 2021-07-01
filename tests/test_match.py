from io import StringIO

import pandas as pd

from journal_targeter import mapping


def test_title_issn_match():
    """Verify proper matching against NLM catalog."""
    # 20 title/issn values with corresponding NLM UID
    data_str = StringIO(
        "input_title	issn	uid\n"
        "CA-A CANCER JOURNAL FOR CLINICIANS	0007-9235	370647\n"
        "NEW ENGLAND JOURNAL OF MEDICINE	0028-4793	255562\n"
        "Nature Reviews Materials	2058-8437	101692128\n"
        "NATURE REVIEWS DRUG DISCOVERY	1474-1776	101124171\n"
        "LANCET	0140-6736	302294\n"
        "NATURE REVIEWS MOLECULAR CELL BIOLOGY	1471-0072	100962782\n"
        "Nature Reviews Clinical Oncology	1759-4774	101500077\n"
        "NATURE REVIEWS CANCER	1474-175X	101124168\n"
        "CHEMICAL REVIEWS	0009-2665	302219\n"
        "Nature Energy	2058-7546	101734042\n"
        "JAMA-JOURNAL OF THE AMERICAN MEDICAL ASSOCIATION	0098-7484	7501160\n"
        "REVIEWS OF MODERN PHYSICS	0034-6861	401307\n"
        "CHEMICAL SOCIETY REVIEWS	0306-0012	335405\n"
        "NATURE	0028-0836	410462\n"
        "SCIENCE	0036-8075	404511\n"
        "Nature Reviews Disease Primers	2056-676X	101672103\n"
        "World Psychiatry	1723-8617	101189643\n"
        "NATURE REVIEWS IMMUNOLOGY	1474-1733	101124169\n"
        "NATURE MATERIALS	1476-1122	101155473\n"
        "CELL	0092-8674	413066\n")

    data = pd.read_csv(data_str, sep='\t', dtype=str)
    match = mapping.lookup_uids_from_title_issn(titles=data['input_title'], issn_print=data['issn'])
    assert match['uid'].equals(data['uid'])
