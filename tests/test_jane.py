import pytest_check as check

from journal_targeter import lookup

text = 'We estimate the distribution of serial intervals for 468 confirmed ' \
       'cases of coronavirus disease reported in China as of February 8, 2020. ' \
       'The mean interval was 3.96 days (95% CI 3.53–4.39 days), SD 4.75 days ' \
       '(95% CI 4.46–5.07 days); 12.6% of case reports indicated presymptomatic ' \
       'transmission.'


def test_api_matches_scrape():
    ja, aa = lookup.fetch_jane_results_via_api(text)
    jh, ah = lookup.fetch_jane_results_via_scrape(text)

    # tables not empty
    check.greater_equal(len(ja), 1)
    check.greater_equal(len(aa), 1)

    # matching table lengths
    check.equal(len(ja), len(jh))
    check.equal(len(aa), len(ah))

    # journal names match
    check.is_true(ja['journal_name'].equals(jh['journal_name']))

    # valid range
    check.less_equal(ja['confidence'].max(), 100)
    check.greater_equal(ja['confidence'].min(), 0)

    # no unexpected nulls
    check.is_false(ja['journal_name'].isnull().any())
    check.is_false(ja['confidence'].isnull().any())
    check.is_false(aa['sim'].isnull().any())






