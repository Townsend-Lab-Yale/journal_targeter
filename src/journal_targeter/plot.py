from collections import OrderedDict

import numpy as np
import pandas as pd
import bokeh as bk
from bokeh import io as bkio
from bokeh import embed as bke
from bokeh import models as bkm
from bokeh import layouts as bkl
from bokeh import plotting as bkp
from bokeh import transform as bkt

from .colors import CATEG_HEX
from .reference import MT

_URL_NLM_BK = "https://www.ncbi.nlm.nih.gov/nlmcatalog/?term=@uid[nlmid]"
_URL_NLM_USCORE = "https://www.ncbi.nlm.nih.gov/nlmcatalog/?term=<%= uid %>[nlmid]"
_URL_PUBMED = "https://pubmed.ncbi.nlm.nih.gov/@PMID/"
_DEFAULT_IMPACT = "CiteScore"
_DEFAULT_MATCH = 'sim_max'


def get_bokeh_components(jf, af, refs_df):
    """Returns bokeh_js, bokeh_divs."""
    source_j, source_a, source_c = build_bokeh_sources(jf, af, refs_df)

    filter_dict = {
        'is_open': bkm.BooleanFilter(jf['is_open'].eq(True)),
        'is_closed': bkm.BooleanFilter(~jf['is_open'].eq(True)),
        'known_metric': bkm.CustomJSFilter(code="""
            var indices = [];
            for (var i = 0; i < source.get_length(); i++){
                if (source.data['ax_impact'][i] < 0){
                    indices.push(false);
                } else {
                    indices.push(true);
                }
            }
            return indices;
            """),
    }

    plots = OrderedDict()
    plots['icats'] = plot_icats(source_j, source_a, source_c, filter_dict=filter_dict)
    plots['table'] = plot_datatable(source_j)
    plots['fit'] = plot_fit_scatter(source_j, filter_dict=filter_dict)
    plots['prospect'] = plot_prospect_scatter(source_j, filter_dict=filter_dict)
    bokeh_js, bokeh_divs = bke.components(plots)
    return bokeh_js, bokeh_divs


def build_bokeh_sources(jf, af, refs_df):
    """Return source_j, source_a, source_c."""
    # JOURNALS
    jfs = jf.copy()
    for metric in MT.metric_list:
        # Replace nans with -1 as workaround for bokeh nan sorting
        jfs[metric] = jfs[metric].fillna(-1)
        jfs[f'p_{metric}'] = jfs[f'p_{metric}'].fillna(-1)
        # Create column for hovertool values
        jfs[f'{metric}_str'] = jfs[metric].map(lambda v: 'unknown' if v < 0 else f"{v:0.1f}")
        _mark_dominant_journals(jfs, metric)
    jfs['doaj_seal'] = jfs['doaj_seal'].map({'Yes': 1, 'No': 0, np.nan: -1})
    jfs['apc'] = jfs['apc'].map({'Yes': 1, 'No': 0, np.nan: -1})

    # checkmark columns
    jfs['is_oa_str'] = jfs['is_open'].map({True: '✔', False: '', np.nan: '?'})
    jfs['in_ml_str'] = jfs['in_medline'].map({True: '✔', False: '', np.nan: '?'})
    jfs['in_pmc_str'] = jfs['in_pmc'].map({True: '✔', False: '', np.nan: '?'})
    # fill jane metric columns
    for col in ['conf_pc', 'sim_sum', 'sim_max']:
        jfs[col].fillna(-1, inplace=True)

    jfs['loc_cited'] = 'cited'
    jfs['loc_abstract'] = 'abstract'
    jfs['loc_title'] = 'title'
    jfs['ax_impact'] = jfs[_DEFAULT_IMPACT]  # redundant column for metric toggling
    jfs['dominant'] = jfs[f'dominant_{_DEFAULT_IMPACT}']
    jfs['ax_impact_bg'] = (jfs[_DEFAULT_IMPACT] < 1).map({True: 'whitesmoke', False: 'white'})
    jfs['impact_max'] = jfs[_DEFAULT_IMPACT].max()
    jfs['ax_match'] = jfs[_DEFAULT_MATCH]  # redundant column for suitability toggling
    jfs['prospect'] = jfs[f"p_{_DEFAULT_IMPACT}"]
    jfs['label_metric'] = jfs[f"label_{_DEFAULT_IMPACT}"]

    source_j = bkm.ColumnDataSource(jfs)

    # ARTICLES
    afs = af[af.jid.isin(jfs.jid)].copy()
    afs['loc_abstract'] = 'abstract'
    afs['loc_title'] = 'title'
    source_a = bkm.ColumnDataSource(afs)

    # CITATIONS. user cited articles that overlap jane journal results
    cited = refs_df.copy()
    cited['loc_cited'] = 'cited'
    source_c = bkm.ColumnDataSource(cited)

    return source_j, source_a, source_c


def plot_prospect_scatter(source_j, show_plot=False, filter_dict=None):
    TOOLS = "pan,wheel_zoom,box_select,reset,tap"
    plot_width, plot_height = 800, 400
    default_metric_label = _DEFAULT_IMPACT

    # IMPACT VS PROSPECT FIGURE (p1)
    p1 = bkp.figure(tools=TOOLS, plot_width=plot_width, plot_height=plot_height,
                    y_axis_label=default_metric_label, x_axis_label='Prospect',
                    x_range=(0, 1), active_scroll='wheel_zoom')
    labels = bkm.LabelSet(x='prospect', y='ax_impact', text='label_metric',
                          source=source_j, x_offset=4, y_offset=2,
                          text_font_size='10pt', background_fill_color='white',
                          background_fill_alpha=0.5)
    labels.level = 'underlay'
    p1.add_layout(labels)
    p1.toolbar.logo = None
    impact_kws = dict(y='ax_impact', x='prospect')
    _add_scatter(fig=p1, source=source_j, filter_dict=filter_dict, **impact_kws)

    # WIDGETS
    option_dict = {i: i for i in MT.metric_list}
    select_kws = dict(width=150, width_policy='fixed', margin=(5, 5, 5, 45))
    select1 = bkm.widgets.Select(title="Impact metric:",
                                 value=default_metric_label,
                                 options=list(option_dict),
                                 **select_kws)

    def get_prospect_js():
        code = """
        const option = select.value;
        const option_dict = %s;
        const new_data = Object.assign({}, source.data);
        new_data.prospect = source.data['p_'.concat(option_dict[option])];
        new_data.ax_impact = source.data[option_dict[option]];
        new_data.label_metric = source.data['label_'.concat(option_dict[option])];
        new_data.dominant = source.data['dominant_'.concat(option_dict[option])];
        ax[0].axis_label = option;
        source.data = new_data;
        slider.value = 1;
        p.x_range.reset();
        p.y_range.reset();
        """ % option_dict
        return code

    slider = bkm.widgets.Slider(start=0.05, end=5, value=1, step=0.05, title="Weight")

    select1.js_on_change('value', bkm.callbacks.CustomJS(args=dict(
        select=select1, ax=p1.yaxis, source=source_j, slider=slider, p=p1),
        code=get_prospect_js()))
    slider.js_on_change('value', bkm.CustomJS(args=dict(source=source_j, select=select1), code="""
        const new_data = Object.assign({}, source.data);
        const col_names = %s;
        const weight = cb_obj.value;
        const impact_col = col_names[select.value];
        const impact_vals = source.data[impact_col];
        const cat_vals = source.data['CAT']
        let prospects = [];
        for (let ind = 0; ind < impact_vals.length; ind++){
            let impact = impact_vals[ind];
            if (impact >= 0){
                let cat = cat_vals[ind];
                let p = cat / (weight * impact + cat);
                prospects.push(p);
            }
            else {
                prospects.push(-1);
            }
        }
        new_data.prospect = prospects;
        source.data = new_data;
    """ % option_dict))

    grid = bkl.gridplot([[bkl.row(select1, slider)], [p1]], merge_tools=False)
    if show_plot:
        bk.io.show(grid)
    return grid


def plot_fit_scatter(source_j, show_plot=False, filter_dict=None):
    """Scatter plot: CAT vs CiteScore."""
    TOOLS = "pan,wheel_zoom,box_select,reset"
    plot_width, plot_height = 400, 400
    label_dict = {i: i for i in MT.metric_list}
    label_dict.update({
        'CAT': 'CAT (Citations + Abstract hits + Title hits)',
        'sim_max': 'Max Similarity',
        'sim_sum': 'Sum of Similarities',
        })
    match_cols = ['sim_sum', 'sim_max']
    max_cat = max(max(source_j.data['CAT']), 1)
    range_cat = bkm.Range1d(start=0, end=max_cat + 0.5, bounds=[0, max_cat + 0.5])
    ticker_cat = bkm.FixedTicker(ticks=list([i + 1 for i in range(max_cat)]))
    cat_jitter = bkt.jitter('CAT', 0.3, range=range_cat)

    fig_kws = dict(tools=TOOLS, plot_width=plot_width, plot_height=plot_height,
                   y_axis_label=label_dict[_DEFAULT_IMPACT])

    # CAT VS IMPACT FIGURE (p1)
    p1 = bkp.figure(x_range=range_cat, x_axis_label=label_dict['CAT'],
                    **fig_kws)
    p1.xaxis.ticker = ticker_cat
    impact_kws = dict(x=cat_jitter, y='ax_impact')
    _add_scatter(p1, source=source_j, filter_dict=filter_dict, **impact_kws)

    # SIM VS IMPACT FIGURE (p2)
    p2 = bkp.figure(x_axis_label=label_dict[_DEFAULT_MATCH], **fig_kws)
    match_kws = dict(x='ax_match', y='ax_impact')
    _add_scatter(fig=p2, source=source_j, filter_dict=filter_dict, **match_kws)

    # WIDGETS
    option_dict = {label_dict[i]: i for i in label_dict}
    select_kws = dict(width=150, width_policy='fixed', margin=(5, 5, 5, 45))
    select1 = bkm.widgets.Select(title="Impact metric:",
                                 value=label_dict[_DEFAULT_IMPACT],
                                 options=[label_dict[i] for i in MT.metric_list],
                                 **select_kws)
    select2 = bkm.widgets.Select(title="Similarity metric:",
                                 value=label_dict[_DEFAULT_MATCH],
                                 options=[label_dict[i] for i in match_cols],
                                 **select_kws)

    def get_select_js(col_name, impact_changed=True):
        # impact_changed = True
        # figures [p1, p2]
        # impact axes [ax1, ax2]
        code_data = """
        const option_dict = %s
        const option = select.value;
        const new_data = Object.assign({}, source.data);
        new_data.%s = source.data[option_dict[option]];
        source.data = new_data;
        """ % (option_dict, col_name)

        if impact_changed:
            code_axis = """
            axy1[0].axis_label = option;
            axy2[0].axis_label = option;
            """
        else:
            code_axis = """
            axis[0].axis_label = option;
            """
        return code_data + code_axis

    select1.js_on_change('value', bkm.callbacks.CustomJS(
        args=dict(select=select1, axy1=p1.yaxis, axy2=p2.yaxis, source=source_j),
        code=get_select_js('ax_impact', impact_changed=True)))
    select2.js_on_change('value', bkm.callbacks.CustomJS(
        args=dict(select=select2, axis=p2.xaxis, source=source_j),
        code=get_select_js('ax_match', impact_changed=False)))

    # column = bkl.column([select1, p1])
    grid = bkl.gridplot([[select1, select2], [p1, p2]], toolbar_location='left',
                        toolbar_options={'logo': None})
    if show_plot:
        bk.io.show(grid)
    return grid


def _add_scatter(fig=None, source=None, filter_dict=None, **kwargs):
    """Add circle renderers for open and closed journals, inc hover + tap."""
    cmap_oa = bkt.factor_cmap('is_oa_str', palette=['red', 'blue'], factors=['✔', ''])
    view_closed = bkm.CDSView(source=source,
                              filters=[filter_dict['is_closed'],
                                       filter_dict['known_metric']])
    view_open = bkm.CDSView(source=source,
                            filters=[filter_dict['is_open'],
                                     filter_dict['known_metric']])
    scatter_kws = dict(size=10, color=cmap_oa, source=source)
    closed_kws = dict(legend_label='closed', fill_alpha=0.3, view=view_closed)
    open_kws = dict(legend_label='open', fill_alpha=0.6, view=view_open)

    r1_open = fig.circle(**open_kws, **scatter_kws, **kwargs)
    r1_closed = fig.circle(**closed_kws, **scatter_kws, **kwargs)

    tooltips = [('Journal', '@journal_name')]
    tooltips.extend([(i, f'@{i + "_str"}') for i in MT.metric_list])
    tooltips.append(('CAT', '@CAT (@cited, @abstract, @title)'))

    fig.add_tools(bkm.HoverTool(renderers=[r1_open, r1_closed], tooltips=tooltips))  # , callback=cb_hover
    taptool = fig.select(type=bkm.TapTool)
    taptool.callback = bkm.OpenURL(url=_URL_NLM_BK)
    fig.legend.click_policy = 'hide'


def plot_datatable(source_j, show_plot=False, table_kws=None):
    if not table_kws:
        table_kws = {}
    col_kws = {
        'default_sort': 'descending',
    }
    w_journal = 300
    w_doaj = 110
    w_md = 60
    w_sm = 40
    w_xs = 30

    metric_dict = OrderedDict({
        f"{i}": (i, w_sm) for i in MT.metric_list
    })
    metric_dict['CiteScore'] = ('CiteScore', w_md)
    # metric_dict['Influence'] = ('Inf', w_sm)
    col_param_dict = OrderedDict({
        'journal_name': ('Journal', w_journal),
        'CAT': ('CAT', w_sm),
        'cited': ('C', w_xs),
        'abstract': ('A', w_xs),
        'title': ('T', w_xs),
        'both': ('A&T', w_xs),
        'doaj_seal': ('DOAJ', w_doaj),
    })
    col_param_dict.update(metric_dict)
    col_param_dict.update({
        'prospect': ('P', w_sm),
        # 'is_oa_str': ('OA', w_sm),
        'in_ml_str': ('ML', w_sm),
        'in_pmc_str': ('PMC', w_sm),
        # 'conf_title': ('cT', w_sm),
        # 'conf_abstract': ('cA', w_sm),
        'conf_pc': ('conf', w_sm),
        'sim_sum': ('∑sim', w_sm),
        'sim_max': ('⤒sim', w_sm),
    })
    index_width = 0  # setting index_position to None
    table_width = sum([i[1] for i in col_param_dict.values()]) + index_width

    # url_template = """<a href="<%= value %>" target="_blank"><%= value %></a>"""

    sr_template = """'<a href="https://v2.sherpa.ac.uk/id/publication/' + sr_id """ \
        """+ '" target="_blank"><i class="fas fa-registered fa-fw pr-1"></i></a>'"""

    cc_template = """'<a href="' + url_doaj + '" target="_blank">""" \
                  f"""<i class="fab fa-creative-commons fa-fw pr-1"></i></a>'"""

    title_template = (
        f"""<a href="{_URL_NLM_USCORE}" target="_blank">"""  # <i class="fas fa-book pr-1"></i>
        """<i class="fas fa-landmark fa-fw pr-1"></i></a>"""
        """<%= sr_id == 'NaN' ? '<i class="fas fa-registered fa-fw pr-1 """
        f"""icon-empty"></i>' : {sr_template} %><%= url_doaj == 'NaN' ? """
        f"""'<i class="fab fa-creative-commons fa-fw pr-1 icon-empty"></i>' : {cc_template} %>"""
        """<span class="journal-cell" data-toggle="tooltip" title="<%= value %>">"""
        """<%= value %></span>"""
    )

    _fragment_seal = """<span data-toggle="tooltip" title="Awarded DOAJ Seal">""" \
        """<i class="fas fa-award fa-fw fa-fw pr-1"></i></span>"""
    _fragment_tick = """<span data-toggle="tooltip" title="DOAJ compliant">""" \
        """<a href="' + url_doaj + '" target="_blank">""" \
        """<i class="far fa-check-circle fa-fw pr-1"></i></a></span>"""

    _fragment_auth_y = '<span data-toggle="tooltip" title="Author holds copyright">' \
                       '<i class="fas fa-user fa-fw pr-1"></i></span>'
    _fragment_auth_n = '<span data-toggle="tooltip" title="Author does not hold copyright">' \
                       '<i class="fas fa-user-slash fa-fw pr-1"></i></span>'

    _fragment_apc_1 = """'<span data-toggle="tooltip" title="Max APC: ' + apc_val + '">""" \
                      """<i class="fas fa-dollar-sign fa-fw pr-1"></i></span>'"""
    _fragment_apc_0 = '<span data-toggle="tooltip" title="No processing charge">' \
                      '<i class="fas fa-gift fa-fw pr-1"></i></span>'

    _fragment_preserve = """'<span data-toggle="tooltip" title="Archived in: ' + preservation + '">""" \
                         """<i class="fas fa-archive fa-fw pr-1"></i></span>'"""
    _fragment_weeks = """'<span data-toggle="tooltip" title="Avg weeks to publication">""" \
                      """' + n_weeks_avg + 'w</span>'"""

    doaj_template = (
        f"""<%= doaj_compliant == 'Yes' ? '{_fragment_tick}' : '' %>"""
        f"""<%= doaj_seal == 1 ? '{_fragment_seal}' : '' %>"""
        f"""<%= author_copyright == 'Yes' ? '{_fragment_auth_y}' : '' %>"""
        f"""<%= author_copyright == 'No' ? '{_fragment_auth_n}' : '' %>"""
        f"""<%= apc == 1 ? {_fragment_apc_1} : '' %>"""
        f"""<%= apc == 0 ? '{_fragment_apc_0}' : '' %>"""
        f"""<%= preservation == 'NaN' ? '' : {_fragment_preserve} %>"""
        f"""<%= isNaN(n_weeks_avg) ? '' : {_fragment_weeks} %>"""
    )

    format_dict = {
        'journal_name': bkm.widgets.HTMLTemplateFormatter(template=title_template),
        'doaj_seal': bkm.widgets.HTMLTemplateFormatter(template=doaj_template),
        'is_oa_str': bkm.widgets.StringFormatter(),
        'in_ml_str': bkm.widgets.StringFormatter(),
        'in_pmc_str': bkm.widgets.StringFormatter(),
    }
    format_dict.update({i: _get_formatter_mark_blank_round_dp(dp=0) for i in
                        ['sim_sum', 'sim_max']})
    format_dict.update({i: _get_formatter_mark_blank_round_dp(dp=1) for i in
                        list(metric_dict) + ['conf_pc']})
    format_dict.update({'prospect': _get_formatter_mark_blank_round_dp(dp=2)})
    table_cols = OrderedDict({
        col: dict(width=col_param_dict[col][1],
                  formatter=format_dict.get(col,
                                            bkm.widgets.NumberFormatter(format='0')),
                  **col_kws) for col in col_param_dict})
    # jfp = jf[col_param_dict].rename(columns=col_names)

    # LINK_COLS = []
    columns = []  # FOR DataTable
    for col in col_param_dict:
        columns.append(bkm.widgets.TableColumn(
            field=col, title=col_param_dict[col][0],
            **table_cols[col]))
    n_journals = len(source_j.data['index'])
    row_height = 25  # pixels
    table_height = (n_journals + 1) * row_height  # add 1 for header
    data_table = bkm.widgets.DataTable(source=source_j, columns=columns,
                                       width=table_width, height=table_height,
                                       row_height=row_height,
                                       index_position=None, fit_columns=False,
                                       **table_kws)
    if show_plot:
        bkio.show(data_table)
    return data_table


def _get_formatter_mark_blank_round_dp(dp=1):
    scalar = 10 ** dp
    return bkm.widgets.HTMLTemplateFormatter(
        template=f"""<span class="col-metric">"""
                 f"""<%= value < 0 ? '' : Math.round(value * {scalar}) / {scalar} %></span>""")


def plot_icats(source_j, source_a, source_c, show_plot=False, filter_dict=None):
    """Create interactive ICATS scatter plot.

    Returns:
         (js, div) Bokeh javascript and html div elements if
         as_components=True, else notebook handle.
    """
    # if n_journals is None:
    #     n_journals = len(jf)
    width_l, width_m, width_r = 300, 120, 300
    TOOLS = "ypan,ywheel_zoom,reset,tap"

    text_props = {"text_align": "center", "text_baseline": "middle",
                  'text_color': '#000000', 'text_font_size': '10pt'}
    factors = ['cited', 'abstract', 'title']
    stack_factors = ['cited', 'title_only', 'abstract_only', 'both']
    # categ_colors = dict(zip(stack_factors, bk.palettes.Colorblind4))
    a_colors = [CATEG_HEX[i] for i in stack_factors]
    # box_colors = [CATEG_HEX[i] for i in factors]

    n_journals = source_j.data['index'].size
    plot_height = 36 * n_journals + 38

    view_aa = bkm.CDSView(source=source_a, filters=[bkm.BooleanFilter(source_a.data['in_abstract'])])
    view_at = bkm.CDSView(source=source_a, filters=[bkm.BooleanFilter(source_a.data['in_title'])])

    # JID / NAME TUPLES FOR Y RANGES
    jfs = source_j.to_df()  # temporary dataframe to simplify calculations
    factor_dict = _get_journal_factor_dict(jfs)
    factors_default = [tuple(i) for i in factor_dict['CAT']]
    jname_factors = bkm.FactorRange(factors=factors_default,  # bounds=(-200, 50),
                                    factor_padding=0, group_padding=0)

    # MIDDLE SQUARES
    p = bkp.figure(
        tools=TOOLS,
        x_range=factors,
        y_range=jname_factors,
        plot_width=width_m, plot_height=plot_height,
        x_axis_location="above")
    # INDIVIDUAL ARTICLE RECT GLYPHS
    r_ac = p.rect(y='jid', x='loc_cited', color=CATEG_HEX['cited'], width=0.95, height=0.95, fill_alpha=0.3, source=source_c)
    r_aa = p.rect(y='jid', x='loc_abstract', color=CATEG_HEX['abstract'], width=0.95, height=0.95, fill_alpha=0.3, source=source_a, view=view_aa)
    r_at = p.rect(y='jid', x='loc_title', color=CATEG_HEX['title'], width=0.95, height=0.95, fill_alpha=0.3, source=source_a, view=view_at)
    # OVERLAYED TEXT GLYPHS
    p.text(y='jid', x='loc_cited', text='cited', source=source_j, **text_props)
    p.text(y='jid', x='loc_abstract', text='abstract', source=source_j, **text_props)
    p.text(y='jid', x='loc_title', text='title', source=source_j, **text_props)

    # LEFT HAND SIDE: IMPACT
    impact_max_initial = jfs['impact_max'].iloc[0]
    # impact_max = jfs[MT.metric_list].max().max()
    p_l = bkp.figure(tools=TOOLS,
                     x_range=(impact_max_initial, 0), y_range=p.y_range,
                     plot_width=width_l, plot_height=plot_height,
                     x_axis_label=_DEFAULT_IMPACT, x_axis_location="above")
    r_ibg = p_l.hbar(y='jid', height=1, left=0, right='impact_max',
                     source=source_j, color='ax_impact_bg')
    view_known = bkm.CDSView(source=source_j, filters=[filter_dict['known_metric']])
    r_i = p_l.hbar(y='jid', height=0.4, left=0, right='ax_impact',
                   source=source_j, view=view_known)
    taptool_impact = p_l.select(type=bkm.TapTool)
    taptool_impact.callback = bkm.OpenURL(url=_URL_NLM_BK)

    # WIDGETS
    # IMPACT SELECT
    metric_options = {i: i for i in MT.metric_list}
    default_metric_label = _DEFAULT_IMPACT
    select_kws = dict(width=100, width_policy='fixed', margin=(5, 5, 5, 15))
    select1 = bkm.widgets.Select(title="Impact metric:",
                                 value=default_metric_label,
                                 options=list(metric_options),
                                 **select_kws)
    impact_js = """const option = select.value;
            const option_dict = %s;
            const impact_vals = source.data[option_dict[option]];
            var max_impact = 0;
            for (var i = 0; i < impact_vals.length; i++) {
                if (impact_vals[i] > max_impact){
                    max_impact = impact_vals[i];
                }
            }
            let na_vals = [];
            let max_vals = [];
            for (var i = 0; i < impact_vals.length; i++) {
                if (impact_vals[i] < 0){
                    na_vals.push('whitesmoke');
                }
                else {
                    na_vals.push('white');
                }
                max_vals.push(max_impact);
            }
            const new_data = Object.assign({}, source.data);
            new_data.ax_impact = impact_vals;
            new_data.ax_impact_bg = na_vals;
            new_data.impact_max = max_vals;
            ax[0].axis_label = option;
            xrange.start = max_impact;
            source.data = new_data;
            """ % metric_options
    select1.js_on_change('value', bkm.callbacks.CustomJS(
        args=dict(select=select1, xrange=p_l.x_range, ax=p_l.xaxis, source=source_j),
        code=impact_js))
    # SORT SELECT
    select2 = bkm.widgets.Select(title='Sort by:', value='CAT', width=120,
                                 width_policy='fixed',
                                 options=list(factor_dict))
    select2.js_on_change('value', bkm.callbacks.CustomJS(
        args=dict(select=select2, y_range=p.y_range, source=source_a),
        code=f"""const range_dict = %s; 
             y_range.factors = range_dict[select.value];
             source.change.emit();"""  # y_range.change.emit();
             % factor_dict))

    # RIGHT HAND SIDE: SCATTER AND WHISKER
    p_r = bkp.figure(
        tools=TOOLS, x_range=(0, 105),  y_range=p.y_range,
        y_axis_location='right', plot_width=width_r, plot_height=plot_height,
        x_axis_label='Similarity', x_axis_location="above")
    p_r.add_layout(bkm.Whisker(source=source_j, base='jid', dimension='width',
                               upper="sim_max", lower="sim_min",
                               line_alpha=1, line_color='gray', line_width=0.5))
    factor_cm = bkt.factor_cmap('categ', palette=a_colors, factors=stack_factors)
    r_as = p_r.circle(y='jid',  # y=bkt.jitter('jid', width=0.5, range=p_r.y_range),
                      x='sim_max', source=source_a,  # x_range_name='ax_sim',
                      size=10, alpha=0.5, color=factor_cm,)
    taptool = p_r.select(type=bkm.TapTool)
    taptool.callback = bkm.OpenURL(url=_URL_PUBMED)

    # HOVERTOOLS
    cite_cols_dict = {'use_year': 'year',
                      'use_article_title': 'title',
                      'use_authors': 'authors'}
    tooltips_c = [(cite_cols_dict[i], f"@{i}") for i in cite_cols_dict]
    hover_c = bkm.HoverTool(renderers=[r_ac], tooltips=tooltips_c)

    impact_dict = OrderedDict({'journal_name': 'Journal'})
    impact_dict.update({f"{i}_str": i for i in MT.metric_list})
    impact_dict['tags'] = 'tags'
    impact_tooltips = [(impact_dict[i], f"@{i}") for i in impact_dict]
    hover_i = bkm.HoverTool(renderers=[r_ibg], tooltips=impact_tooltips)

    a_cols_dict = {'year': 'year',
                   'title': 'title',
                   'authors_short': 'authors',
                   # 'url': 'url',
                   'sim_max': 'sim'}
    article_tooltips = [(a_cols_dict[i], f"@{i}") for i in a_cols_dict]  # [("(x,y)", "($x, $y)")]
    hover_a = bkm.HoverTool(renderers=[r_aa, r_at], tooltips=article_tooltips)
    hover_s = bkm.HoverTool(renderers=[r_as], tooltips=article_tooltips)

    p_l.add_tools(hover_i)
    p.add_tools(hover_a, hover_c)
    p_r.add_tools(hover_s)

    # MINIMAL STYLING FOR AXES/TICKS/GRIDLINES
    for fig in [p, p_r, p_l]:
        fig.outline_line_color = None
        fig.grid.grid_line_color = None
        fig.axis.axis_line_color = None
        fig.axis.major_tick_line_color = None
        fig.axis.major_label_standoff = 0
    p_r.yaxis.group_text_color = '#ffffff'  # hides y_range j_id group
    # p_l.axis[1].axis_line_color = '#000000'
    p_l.yaxis.visible = False
    p.yaxis.visible = False
    select_row = bkl.row(select1, select2)
    grid = bkl.gridplot([[select_row], [p_l, p, p_r]], toolbar_location=None)

    if show_plot:
        bkp.show(grid)
    # js, div = bke.components(grid)
    return grid


def _mark_dominant_journals(df, metric):
    """Add 'dominant_<metric>' column to journals table."""

    prob = f"p_{metric}"

    def row_is_dominated(temp):
        if pd.isnull(temp[metric]):
            return True
        for ind, r in df.iterrows():
            "Does this row dominate current value."
            if pd.isnull(r[metric]):
                continue
            if r[metric] > temp[metric] and r[prob] > temp[prob]:
                return True
            """Only keep one duplicate m, p."""
        return False

    is_dominated = df.apply(row_is_dominated, axis=1)
    is_dominant = ~is_dominated
    df[f'dominant_{metric}'] = is_dominant
    df[f'label_{metric}'] = df['abbr'].where(is_dominant, '')


def _get_journal_factor_dict(jf):
    """Create dictionary of sorting name -> factor range."""
    metric_col_dict = {i: i for i in MT.metric_list}
    sort_dict = OrderedDict({'CAT': ['CAT', 'sim_sum']})
    for metric_name in metric_col_dict:
        sort_dict[metric_name] = [metric_col_dict[metric_name], 'sim_sum']
    sort_dict.update([
                      ('Max similarity', ['sim_max', 'sim_sum']),
                      ('Cited', ['cited', 'sim_sum']),
                      ('Abstract', ['abstract', 'sim_sum']),
                      ('Title', ['title', 'sim_sum']),
                      ])
    index_dict = {}
    for sort_name in sort_dict:
        index_dict[sort_name] = \
            [list(i) for i in jf.sort_values(sort_dict[sort_name], ascending=False)[
                ['jid', 'abbr']].values][::-1]
    return index_dict
