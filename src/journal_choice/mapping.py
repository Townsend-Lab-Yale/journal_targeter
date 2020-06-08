
def get_pubmed_mapping():

    testt = pd.merge(pub[pub.is_unique_title_safe], scop[scop.is_unique_title_safe],
                     on='title_safe', how='inner', suffixes=['_pub', '_scop'])
    testt.rename(columns={'title_safe': 'title_safe_scop'}, inplace=True)

    testp = pd.merge(
        pub[pub.unique_isoabbr_nodots].dropna(axis=0, subset=['issn_print']),
        scop.dropna(axis=0, subset=['issn_print']),
        on='issn_print', how='inner', suffixes=['_pub', '_scop'])

    testo = pd.merge(
        pub[pub.unique_isoabbr_nodots].dropna(axis=0, subset=['issn_online']),
        scop.dropna(axis=0, subset=['issn_online']),
        on='issn_online', how='inner', suffixes=['_pub', '_scop'])

    testc = pd.merge(
        pub[pub.unique_isoabbr_nodots].dropna(axis=0, subset=['issn_comb']),
        scop.dropna(axis=0, subset=['issn_comb']),
        on='issn_comb', how='inner', suffixes=['_pub', '_scop'])

    test1 = pd.merge(
        pub[pub.unique_isoabbr_nodots].dropna(axis=0, subset=['issn1']),
        scop.dropna(axis=0, subset=['issn1']),
        on='issn1', how='inner', suffixes=['_pub', '_scop'])

    safe_to_abbr_dict = pub[pub.is_unique_title_safe & pub.unique_isoabbr_nodots].set_index('title_safe')['isoabbr_nodots'].to_dict()

    ambiguous_abbrvs = set()

    for merged, method in [(testc, 'issn_combined'),
                           (testo, 'issn_print'),
                           (testp, 'issn_online'),
                           (testt, 'safe_title'),
                          ]:
        print(f"Linking Scopus to Pubmed table via {method}")
        updates = merged.set_index('title_safe_scop')['isoabbr_nodots'].to_dict()
        n_additions = 0
        for key in updates:
            if key in safe_to_abbr_dict:
                # Handle contradictory update
                if safe_to_abbr_dict[key] != updates[key]:
                    print(f"CLASH {key!r}: PUBMED {safe_to_abbr_dict[key]} | SCOPUS: {updates[key]}")
                    ambiguous_abbrvs.add(key)
            else:
                n_additions += 1
                safe_to_abbr_dict[key] = updates[key]
        print(f"{n_additions} via {method}.\n")