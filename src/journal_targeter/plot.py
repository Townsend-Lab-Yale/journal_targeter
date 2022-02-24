from importlib import resources
from collections import OrderedDict
from typing import Union

import numpy as np
import pandas as pd
import bokeh as bk
from bokeh import io as bkio
from bokeh import embed as bke
from bokeh import models as bkm
from bokeh import layouts as bkl
from bokeh import plotting as bkp
from bokeh import transform as bkt
from bokeh.model import DataModel
from bokeh.core.properties import String, Float, Bool
from bokeh.models import CustomJS, Toggle

from .colors import CATEG_HEX
from .reference import MT

_URL_NLM_BK = "https://www.ncbi.nlm.nih.gov/nlmcatalog/?term=@uid[nlmid]"
# _URL_NLM_USCORE = "https://www.ncbi.nlm.nih.gov/nlmcatalog/?term=<%= uid %>[nlmid]"
_URL_PUBMED = "https://pubmed.ncbi.nlm.nih.gov/@PMID/"
_URL_ROMEO = "https://v2.sherpa.ac.uk/id/publication/@sr_id"
_DEFAULT_IMPACT = "CiteScore"
_DEFAULT_WEIGHT = 1
_DEFAULT_MATCH = 'sim_max'
_DEFAULT_SHOW_ALL_METRICS = False


class Params(DataModel):
    """Stores key user-preference variables to link widgets and callbacks."""
    metric = String(default=_DEFAULT_IMPACT, help="Preferred impact metric")
    weight = Float(default=_DEFAULT_WEIGHT, help="Impact weight for Prospect")
    show_all_metrics = Bool(default=_DEFAULT_SHOW_ALL_METRICS, help="Include all metrics in Table.")


class ModelTracker:
    """Holds bokeh object handles used in shared callbacks."""
    def __init__(self):
        self.metric_axes = []
        self.weight_slider = None
        self.xrange_icats = None
        self.fig_prospect = None
        self.table_cols = None
        self.all_metrics_toggle = None
        self.metric_col_inds = None


def get_bokeh_components(jf, af, refs_df, pref_metric=None,
                         pref_weight=_DEFAULT_WEIGHT, store_prefs=False,
                         plots_only=False):
    """Gather interactive plots.

    Returns embeddable bokeh_js, bokeh_divs in typical in-app usage where
    plots_only=False. Otherwise returns dictionary of plot objects.
    """
    pref_metric = _DEFAULT_IMPACT if pref_metric is None else pref_metric
    source_j, source_a, source_c = build_bokeh_sources(jf, af, refs_df,
                                                       pref_metric=pref_metric,
                                                       pref_weight=pref_weight)
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
    mt = ModelTracker()
    skip_refs = len(source_c.to_df()) == 0
    plots['icats'] = plot_icats(source_j, source_a, source_c,
                                filter_dict=filter_dict, mt_obj=mt,
                                pref_metric=pref_metric, skip_refs=skip_refs)
    plots['table'] = plot_datatable(source_j, mt_obj=mt, pref_metric=pref_metric,
                                    skip_refs=skip_refs)
    plots['fit'] = plot_fit_scatter(source_j, filter_dict=filter_dict,
                                    mt_obj=mt, pref_metric=pref_metric,
                                    plots_only=plots_only)
    plots['prospect'] = plot_prospect_scatter(source_j, filter_dict=filter_dict,
                                              mt_obj=mt, pref_metric=pref_metric)
    if plots_only:
        return plots
    params = Params()
    params.metric, params.weight = pref_metric, pref_weight
    # Create preference widgets, link to params
    metric_select = bkm.widgets.Select(title="Preferred impact metric:",
                                       value=pref_metric,
                                       options=MT.metric_list,
                                       width=200, width_policy='fixed',
                                       margin=(5, 5, 5, 5))
    metric_select.js_link('value', params, 'metric')
    if store_prefs:
        metric_select.js_on_change('value', _get_pref_ajax_cb('metric'))

    slider = bkm.widgets.Slider(start=0.05, end=5, value=pref_weight, step=0.05,
                                title="Weight", width=200)
    slider.js_link('value', params, 'weight')
    if store_prefs:
        slider.js_on_change('value_throttled', _get_pref_ajax_cb('weight'))

    plots['prefs_widgets'] = bkl.column(metric_select, slider)
    mt.all_metrics_toggle.js_link('active', params, 'show_all_metrics')
    # Set up shared callbacks
    _create_callbacks_for_params(params=params, source=source_j, mt=mt)
    bokeh_js, bokeh_divs = bke.components(plots)
    return bokeh_js, bokeh_divs


def _get_pref_ajax_cb(param_name):
    code = """
            const param_name = '%s';
            let data = {};
            data[param_name] = cb_obj.value;
            ($.post('/prefer', data)
              .done(response => console.log(response['msg']))
              .fail(() => console.log(param_name + " preference setting failed."))
            )""" % param_name
    return CustomJS(code=code)


def _create_callbacks_for_params(params=None, source=None, mt=None):
    metric_code = _get_metric_weight_change_js(metric_changed=True)
    params.js_on_change('metric', CustomJS(
        args=dict(source=source,
                  params=params,
                  metric_axes=mt.metric_axes,
                  xrange=mt.xrange_icats,
                  p=mt.fig_prospect,
                  table_cols=mt.table_cols,
                  metric_col_inds=mt.metric_col_inds,
                  ),
        code=metric_code))
    weight_code = _get_metric_weight_change_js(metric_changed=False)
    params.js_on_change('weight', CustomJS(
        args=dict(source=source, params=params), code=weight_code))
    toggle_code = _get_toggle_all_metrics_js()
    params.js_on_change('show_all_metrics', CustomJS(
        args=dict(
            params=params,
            metric_col_inds=mt.metric_col_inds,
            table_cols=mt.table_cols,
        ),
        code=toggle_code))


def build_bokeh_sources(jf, af, refs_df, pref_metric=_DEFAULT_IMPACT,
                        pref_weight=_DEFAULT_WEIGHT):
    """Return source_j, source_a, source_c."""
    jfs = jf.copy()
    # Populate any missing metrics (in case results are older than last metric refresh)
    if any(map(lambda v: v not in jfs.columns, MT.metric_list)):
        metric_table = MT.df[MT.metric_list]
        for metric_name in MT.metric_list:
            jfs[metric_name] = jfs['uid'].map(metric_table[metric_name])
    for metric in MT.metric_list:
        # Replace nans with -1 as workaround for bokeh nan sorting
        jfs[metric] = jfs[metric].fillna(-1)
        # Create column for hovertool values
        jfs[f'{metric}_str'] = jfs[metric].map(lambda v: 'unknown' if v < 0 else f"{v:0.1f}")
        _mark_dominant_journals(jfs, metric, weight=pref_weight)
    jfs['doaj_score'] = jfs['doaj_score'].fillna(-1)
    jfs['apc'] = jfs['apc'].map({'Yes': 1, 'No': 0, np.nan: -1})

    # checkmark columns
    jfs['is_oa_str'] = jfs['is_open'].map({True: '✔', False: '', np.nan: '?'})
    jfs['in_ml_str'] = jfs['in_medline'].map({True: '✔', False: '', np.nan: '?'})
    jfs['in_pmc_str'] = jfs['in_pmc'].map({True: '✔', False: '', np.nan: '?'})
    # fill jane metric columns
    for col in ['sim_sum', 'sim_max']:
        jfs[col].fillna(-1, inplace=True)

    jfs['loc_cited'] = 'Cited'
    jfs['loc_abstract'] = 'Abstract'
    jfs['loc_title'] = 'Title'
    jfs['ax_impact'] = jfs[pref_metric]  # redundant column for metric toggling
    jfs['dominant'] = jfs[f'dominant_{pref_metric}']
    jfs['ax_impact_bg'] = (jfs['ax_impact'] < 0).map({True: 'whitesmoke', False: 'white'})
    jfs['impact_max'] = jfs['ax_impact'].max()
    jfs['ax_match'] = jfs[_DEFAULT_MATCH]  # redundant column for suitability toggling
    jfs['prospect'] = jfs['CAT'] / (jfs['CAT'] + pref_weight * jfs['ax_impact'])
    jfs['prospect'] = jfs['prospect'].where(jfs['ax_impact'] >= 0, -1)
    jfs['expect'] = jfs.eval('prospect * ax_impact').where(jfs['prospect'] >= 0, -1)
    jfs['label_metric'] = jfs[f"label_{pref_metric}"]
    jfs.rename(columns={i: i.title() for i in ['cited', 'title', 'abstract']},
               inplace=True)
    source_j = bkm.ColumnDataSource(jfs)

    # ARTICLES
    afs = af[af.jid.isin(jfs.jid)].copy()
    afs['loc_abstract'] = 'Abstract'
    afs['loc_title'] = 'Title'
    source_a = bkm.ColumnDataSource(afs)

    # CITATIONS. user cited articles that overlap jane journal results
    if refs_df is not None:
        cited = refs_df.copy()
        cited['loc_cited'] = 'Cited'
    else:
        cited = pd.DataFrame(columns=['jid', 'Cited', 'loc_cited'])
    source_c = bkm.ColumnDataSource(cited)

    return source_j, source_a, source_c


def plot_prospect_scatter(source_j, show_plot=False, filter_dict=None,
                          mt_obj: Union[None, ModelTracker] = None,
                          pref_metric=_DEFAULT_IMPACT):
    TOOLS = "pan,wheel_zoom,box_select,reset,tap"
    plot_width, plot_height = 800, 450
    default_metric_label = pref_metric

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

    if mt_obj is not None:
        mt_obj.metric_axes.append(p1.yaxis)
        mt_obj.fig_prospect = p1

    if show_plot:
        bk.io.show(p1)
    return p1


def plot_fit_scatter(source_j, show_plot=False, filter_dict=None, mt_obj=None,
                     pref_metric=_DEFAULT_IMPACT, plots_only=False,
                     plot_width=400, plot_height=400):
    """Scatter plot: CAT vs impact metric."""
    TOOLS = "pan,wheel_zoom,box_select,reset"
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
                   y_axis_label=label_dict[pref_metric])

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

    if mt_obj is not None:
        mt_obj.metric_axes.extend([p1.yaxis, p2.yaxis])

    if plots_only:
        grid = bkl.gridplot([[p1, p2]], toolbar_location=None)
        return grid

    # WIDGETS
    option_dict = {label_dict[i]: i for i in label_dict}
    select_kws = dict(width=150, width_policy='fixed', margin=(5, 5, 5, 45))
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

    select2.js_on_change('value', bkm.callbacks.CustomJS(
        args=dict(select=select2, axis=p2.xaxis, source=source_j),
        code=get_select_js('ax_match', impact_changed=False)))

    grid = bkl.gridplot([[p1, p2], [None, select2]], toolbar_location='left',
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
    tooltips.append(('CAT', '@CAT (@Cited, @Abstract, @Title)'))

    fig.add_tools(bkm.HoverTool(renderers=[r1_open, r1_closed], tooltips=tooltips))  # , callback=cb_hover
    taptool = fig.select(type=bkm.TapTool)
    taptool.callback = bkm.OpenURL(url=_URL_NLM_BK)
    fig.legend.click_policy = 'hide'


def plot_datatable(source_j, show_plot=False, table_kws=None, mt_obj=None,
                   pref_metric=_DEFAULT_IMPACT, skip_refs=False):
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
        'CAT': ('A+T' if skip_refs else 'CAT', w_xs),
        'Cited': ('C', w_xs),
        'Abstract': ('A', w_xs),
        'Title': ('T', w_xs),
        'both': ('A&T', w_xs),
        'doaj_score': ('DOAJ', w_doaj),
    })
    # if skip_refs:
    #     col_param_dict.pop('Cited')
    col_param_dict.update(metric_dict)
    col_param_dict.update({
        'prospect': ('P', w_sm),
        'expect': ('E', w_sm),
        # 'is_oa_str': ('OA', w_sm),
        'in_ml_str': ('ML', w_sm),
        'in_pmc_str': ('PMC', w_sm),
        # 'conf_title': ('cT', w_sm),
        # 'conf_abstract': ('cA', w_sm),
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
        """<%= uid == 'NaN' ? '<i class="fas fa-landmark fa-fw pr-1 icon-empty"></i>' """
        """: '<a href="https://www.ncbi.nlm.nih.gov/nlmcatalog/?term=' + uid + """
        """'[nlmid]" target="_blank"><i class="fas fa-landmark fa-fw pr-1"></i></a>' %>"""
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
        f"""<%= doaj_seal == 'Yes' ? '{_fragment_seal}' : '' %>"""
        f"""<%= author_copyright == 'Yes' ? '{_fragment_auth_y}' : '' %>"""
        f"""<%= author_copyright == 'No' ? '{_fragment_auth_n}' : '' %>"""
        f"""<%= apc == 1 ? {_fragment_apc_1} : '' %>"""
        f"""<%= apc == 0 ? '{_fragment_apc_0}' : '' %>"""
        f"""<%= preservation == 'NaN' ? '' : {_fragment_preserve} %>"""
        f"""<%= isNaN(n_weeks_avg) ? '' : {_fragment_weeks} %>"""
    )

    format_dict = {
        'journal_name': bkm.widgets.HTMLTemplateFormatter(template=title_template),
        'doaj_score': bkm.widgets.HTMLTemplateFormatter(template=doaj_template),
        'is_oa_str': bkm.widgets.StringFormatter(),
        'in_ml_str': bkm.widgets.StringFormatter(),
        'in_pmc_str': bkm.widgets.StringFormatter(),
    }
    format_dict.update({i: _get_formatter_mark_blank_round_dp(dp=0) for i in
                        ['sim_sum', 'sim_max']})
    format_dict.update({i: _get_formatter_mark_blank_round_dp(dp=1) for i in
                        list(metric_dict)})
    format_dict.update({i: _get_formatter_mark_blank_round_dp(dp=2) for i in
                        ['prospect', 'expect']})
    table_cols = OrderedDict({
        col: dict(width=col_param_dict[col][1],
                  formatter=format_dict.get(col,
                                            bkm.widgets.NumberFormatter(format='0')),
                  **col_kws) for col in col_param_dict})
    # jfp = jf[col_param_dict].rename(columns=col_names)

    # LINK_COLS = []
    columns = []  # FOR DataTable
    metric_col_inds = {}
    for ind, col in enumerate(col_param_dict):
        bk_col = bkm.widgets.TableColumn(field=col, title=col_param_dict[col][0],
                                         **table_cols[col])
        if skip_refs and col == 'Cited':
            bk_col.visible = False
        columns.append(bk_col)
        if col in MT.metric_list:
            metric_col_inds[col] = ind
    # Hide metric columns if necessary
    if not _DEFAULT_SHOW_ALL_METRICS:
        for metric, col_ind in metric_col_inds.items():
            columns[col_ind].visible = metric == pref_metric
    n_journals = len(source_j.data['index'])
    row_height = 25  # pixels
    table_height = (n_journals + 1) * row_height  # add 1 for header
    data_table = bkm.widgets.DataTable(source=source_j, columns=columns,
                                       width=table_width, height=table_height,
                                       row_height=row_height,
                                       index_position=None, fit_columns=False,
                                       **table_kws)
    toggle = Toggle(label="Show all impact metrics", button_type='default', width=150,
                    active=False)
    if mt_obj is not None:
        mt_obj.table_cols = columns
        mt_obj.all_metrics_toggle = toggle
        mt_obj.metric_col_inds = metric_col_inds
    grid = bkl.gridplot([[toggle], [data_table]], toolbar_location=None)

    if show_plot:
        bkio.show(grid)
    return grid


def _get_formatter_mark_blank_round_dp(dp=1):
    scalar = 10 ** dp
    return bkm.widgets.HTMLTemplateFormatter(
        template=f"""<span class="col-metric">"""
                 f"""<%= value < 0 ? '' : Math.round(value * {scalar}) / {scalar} %></span>""")


def plot_icats(source_j, source_a, source_c, show_plot=False, filter_dict=None,
               mt_obj=None, pref_metric=_DEFAULT_IMPACT, skip_refs=False):
    """Create interactive ICATS scatter plot.

    Returns:
         (js, div) Bokeh javascript and html div elements if
         as_components=True, else notebook handle.
    """
    width_l, width_m, width_r = (300, 80, 300) if skip_refs else (300, 120, 300)
    factors = ['Abstract', 'Title'] if skip_refs else ['Cited', 'Abstract', 'Title']
    TOOLS = "ypan,ywheel_zoom,reset,tap"

    text_props = {"text_align": "center", "text_baseline": "middle",
                  'text_color': '#000000', 'text_font_size': '10pt'}
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
    factor_dict = _get_journal_factor_dict(jfs, skip_refs=skip_refs)
    _first_factor = next((i for i in factor_dict.keys()))
    factors_default = [tuple(i) for i in factor_dict[_first_factor]]
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
    if not skip_refs:
        r_ac = p.rect(y='jid', x='loc_cited', color=CATEG_HEX['cited'], width=0.95, height=0.95, fill_alpha=0.3, source=source_c)
    r_aa = p.rect(y='jid', x='loc_abstract', color=CATEG_HEX['abstract'], width=0.95, height=0.95, fill_alpha=0.3, source=source_a, view=view_aa)
    r_at = p.rect(y='jid', x='loc_title', color=CATEG_HEX['title'], width=0.95, height=0.95, fill_alpha=0.3, source=source_a, view=view_at)
    # OVERLAYED TEXT GLYPHS
    if not skip_refs:
        p.text(y='jid', x='loc_cited', text='Cited', source=source_j, **text_props)
    p.text(y='jid', x='loc_abstract', text='Abstract', source=source_j, **text_props)
    p.text(y='jid', x='loc_title', text='Title', source=source_j, **text_props)

    # LEFT HAND SIDE: IMPACT
    impact_max_initial = jfs['impact_max'].iloc[0]
    # impact_max = jfs[MT.metric_list].max().max()
    p_l = bkp.figure(tools=TOOLS,
                     x_range=(impact_max_initial, 0), y_range=p.y_range,
                     plot_width=width_l, plot_height=plot_height,
                     x_axis_label=pref_metric, x_axis_location="above")
    if mt_obj is not None:
        mt_obj.metric_axes.append(p_l.xaxis)
        mt_obj.xrange_icats = p_l.x_range

    r_ibg = p_l.hbar(y='jid', height=1, left=0, right='impact_max',
                     source=source_j, color='ax_impact_bg')
    view_known = bkm.CDSView(source=source_j, filters=[filter_dict['known_metric']])
    p_l.hbar(y='jid', height=0.4, left=0, right='ax_impact', source=source_j,
             view=view_known)
    taptool_impact = p_l.select(type=bkm.TapTool)
    taptool_impact.callback = bkm.OpenURL(url=_URL_NLM_BK)

    # SORT SELECT WIDGET
    select2 = bkm.widgets.Select(title='Sort by:', value='CAT', width=140,
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
    r_as = p_r.circle(y=bkt.jitter('jid', width=0.5, range=p_r.y_range),
                      x='sim_max', source=source_a,  # x_range_name='ax_sim',
                      size=10, alpha=0.5, color=factor_cm,)
    taptool = p_r.select(type=bkm.TapTool)
    taptool.callback = bkm.OpenURL(url=_URL_PUBMED)

    # HOVERTOOLS
    tooltips_c = [
        ('article', "@use_authors (@use_year): @use_article_title"),
    ]
    if not skip_refs:
        hover_c = bkm.HoverTool(renderers=[r_ac], tooltips=tooltips_c)
    impact_dict = OrderedDict({'journal_name': 'Journal'})
    impact_dict.update({f"{i}_str": i for i in MT.metric_list})
    impact_dict['tags'] = 'tags'
    impact_tooltips = [(impact_dict[i], f"@{i}") for i in impact_dict]
    hover_i = bkm.HoverTool(renderers=[r_ibg], tooltips=impact_tooltips)
    article_tooltips = [
        ('article', "@authors_short (@year): @title"),
        ('sim', '@sim_max')
    ]
    hover_a = bkm.HoverTool(renderers=[r_aa, r_at], tooltips=article_tooltips)
    hover_s = bkm.HoverTool(renderers=[r_as], tooltips=article_tooltips)

    p_l.add_tools(hover_i)
    p.add_tools(hover_a)
    if not skip_refs:
        p.add_tools(hover_c)
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
    select_row = bkl.row(select2)
    grid = bkl.gridplot([[p_l, p, p_r, select_row]], toolbar_location=None)

    if show_plot:
        bkp.show(grid)
    # js, div = bke.components(grid)
    return grid


def _get_metric_weight_change_js(metric_changed: bool):
    with resources.path('journal_targeter.callbacks', 'cb_metric_weight.js') as path:
        with open(path) as infile:
            base_js = infile.read()
    if metric_changed:
        preamble = "const changed_metric = true;"
    else:
        preamble = "const changed_metric = false;"
    return '\n'.join([preamble, base_js])


def _get_toggle_all_metrics_js():
    with resources.path('journal_targeter.callbacks', 'cb_toggle_all_metrics.js') as path:
        with open(path) as infile:
            base_js = infile.read()
    return base_js


def _mark_dominant_journals(df, metric, weight=_DEFAULT_WEIGHT):
    """Add 'dominant_<metric>' column to journals table."""

    prospects = df['CAT'] / (df['CAT'] + weight * df[metric])  # type: pd.Series
    prospects = prospects.where(df[metric] >= 0, -1)

    def row_is_dominated(temp):
        temp_p = prospects[temp.name]
        if pd.isnull(temp[metric]):
            return True
        for ind, r in df.iterrows():
            "Does this row dominate current value."
            if pd.isnull(r[metric]):
                continue
            if r[metric] > temp[metric] and prospects[ind] > temp_p:
                return True
            """Only keep one duplicate m, p."""
        return False

    is_dominated = df.apply(row_is_dominated, axis=1)
    is_dominant = ~is_dominated
    df[f'dominant_{metric}'] = is_dominant
    df[f'label_{metric}'] = df['abbr'].where(is_dominant, '')


def _get_journal_factor_dict(jf, skip_refs=False):
    """Create dictionary of sorting name -> factor range."""
    metric_col_dict = {i: i for i in MT.metric_list}
    cat_label = 'A+T' if skip_refs else 'C+A+T'
    sort_dict = OrderedDict({cat_label: ['CAT', 'sim_sum']})
    for metric_name in metric_col_dict:
        sort_dict[metric_name] = [metric_col_dict[metric_name], 'sim_sum']
    # sort_dict.update([
    sort_dict['Max similarity'] = ['sim_max', 'sim_sum']
    sort_dict['Sum of similarities'] = ['sim_sum', 'sim_max']
    sort_dict['Prospect'] = ['prospect']
    sort_dict['Expected impact'] = ['expect']
    if not skip_refs:
        sort_dict['Cited'] = ['Cited', 'sim_sum']
    sort_dict['Abstract'] = ['Abstract', 'sim_sum']
    sort_dict['Title'] = ['Title', 'sim_sum']
    index_dict = {}
    for sort_name in sort_dict:
        index_dict[sort_name] = \
            [list(i) for i in jf.sort_values(sort_dict[sort_name], ascending=False)[
                ['jid', 'abbr']].values][::-1]
    return index_dict
