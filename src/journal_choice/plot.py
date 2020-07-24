import math
from itertools import product
from collections import OrderedDict

import numpy as np
import matplotlib as mpl
import bokeh as bk
from bokeh import embed as bke
from bokeh import models as bkm
from bokeh import layouts as bkl
from bokeh import plotting as bkp
from bokeh import transform as bkt


_URL_NLM = "https://www.ncbi.nlm.nih.gov/nlmcatalog/@uid"
_DEFAULT_IMPACT = "citescore"
_DEFAULT_MATCH = 'sim_max'


def get_bokeh_components(jf, af, refs_df):
    """Returns bokeh_js, bokeh_divs."""
    source_j, source_a, source_c = build_bokeh_sources(jf, af, refs_df)
    plots = OrderedDict()
    plots['icats'] = plot_icats(source_j, source_a, source_c)
    plots['table'] = plot_datatable(source_j)
    plots['fit'] = plot_fit_scatter(source_j)
    plots['prospect'] = plot_prospect_scatter(source_j)
    bokeh_js, bokeh_divs = bke.components(plots)
    return bokeh_js, bokeh_divs


def build_bokeh_sources(jf, af, refs_df):
    """Return source_j, source_a, source_c."""
    # JOURNALS
    jfs = jf.copy()
    jfs['cs_table'] = jfs['citescore'].map(
        lambda v: f'{v:0.1f}' if not np.isnan(v) else '')
    jfs['inf_table'] = jfs['influence'].map(
        lambda v: f'{v:0.1f}' if not np.isnan(v) else '')
    jfs['loc_cited'] = 'cited'
    jfs['loc_abstract'] = 'abstract'
    jfs['loc_title'] = 'title'
    jfs['ax_impact'] = jfs['citescore']  # redundant column for metric toggling
    jfs['ax_match'] = jfs['sim_max']  # redundant column for suitability toggling
    jfs['prospect'] = jfs[f"p_{_DEFAULT_IMPACT}"]
    source_j = bkm.ColumnDataSource(jfs)

    # ARTICLES
    afs = af[af.jid.isin(jfs.jid)].copy()
    afs['loc_abstract'] = 'abstract'
    afs['loc_title'] = 'title'
    source_a = bkm.ColumnDataSource(afs)

    # CITATIONS. user cited articles that overlap jane journal results
    cited = refs_df.join(jf.set_index('uid')['jid'], on='uid', how='left') \
        .dropna(axis=0, subset=['jid'])
    cited['loc_cited'] = 'cited'
    source_c = bkm.ColumnDataSource(cited)

    return source_j, source_a, source_c


def plot_prospect_scatter(source_j, show_plot=False, **kwargs):
    TOOLS = "pan,wheel_zoom,tap,box_select,reset"
    plot_width, plot_height = 800, 400
    label_dict = {'citescore': 'CiteScore',
                  'influence': 'Influence',
                  }
    impact_cols = ['citescore', 'influence']

    fig_kws = dict(tools=TOOLS, plot_width=plot_width, plot_height=plot_height,
                   x_axis_label=label_dict[_DEFAULT_IMPACT], y_axis_label='Prospect')

    # IMPACT VS PROSPECT FIGURE (p1)
    p1 = bkp.figure(**fig_kws)
    impact_kws = dict(x=_DEFAULT_IMPACT, y='prospect')
    _add_scatter(fig=p1, source=source_j, **impact_kws)

    # WIDGETS
    select_kws = dict(width=150, width_policy='fixed', margin=(5, 5, 5, 45))
    select1 = bkm.widgets.Select(title="Impact metric:",
                                 value=label_dict[_DEFAULT_IMPACT],
                                 options=[label_dict[i] for i in impact_cols],
                                 **select_kws)
    option_dict = {label_dict[i]: i for i in label_dict}

    def get_prospect_js():
        code = """
        const option = select.value;
        const option_dict = %s;
        const new_data = Object.assign({}, source.data);
        new_data.prospect = source.data['p_'.concat(option_dict[option])];
        ax[0].axis_label = option;
        source.data = new_data;
        slider.value = 1;
        """ % option_dict
        return code

    slider = bkm.widgets.Slider(start=0.05, end=5, value=1, step=0.05, title="Weight")

    select1.js_on_change('value', bkm.callbacks.CustomJS(
        args=dict(select=select1, ax=p1.xaxis, source=source_j, slider=slider),
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
            let cat = cat_vals[ind];
            let p = cat / (weight * impact + cat);
            prospects.push(p);
        }
        new_data.prospect = prospects;
        source.data = new_data;

    """ % option_dict))

    grid = bkl.gridplot([[bkl.row(select1, slider)], [p1]], toolbar_location='left')
    if show_plot:
        bk.io.show(grid)
    return grid


def plot_fit_scatter(source_j, show_plot=False, **kwargs):
    """Scatter plot: CAT vs CiteScore."""
    TOOLS = "pan,wheel_zoom,tap,box_select,reset"
    plot_width, plot_height = 400, 400
    label_dict = {'citescore': 'CiteScore',
                  'influence': 'Influence',
                  'CAT': 'CAT (Citations + Abstract hits + Title hits)',
                  'sim_max': 'Max Similarity',
                  'sim_sum': 'Sum of Similarities',
                  }
    impact_cols = ['citescore', 'influence']
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
    _add_scatter(p1, source=source_j, **impact_kws)

    # SIM VS IMPACT FIGURE (p2)
    p2 = bkp.figure(x_axis_label=label_dict[_DEFAULT_MATCH], **fig_kws)
    match_kws = dict(x='ax_match', y='ax_impact')
    _add_scatter(fig=p2, source=source_j, **match_kws)

    # WIDGETS
    select_kws = dict(width=150, width_policy='fixed', margin=(5, 5, 5, 45))
    select1 = bkm.widgets.Select(title="Impact metric:",
                                 value=label_dict[_DEFAULT_IMPACT],
                                 options=[label_dict[i] for i in impact_cols],
                                 **select_kws)
    select2 = bkm.widgets.Select(title="Similarity metric:",
                                 value=label_dict[_DEFAULT_MATCH],
                                 options=[label_dict[i] for i in match_cols],
                                 **select_kws)
    option_dict = {label_dict[i]: i for i in label_dict}

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
    grid = bkl.gridplot([[select1, select2], [p1, p2]], toolbar_location='left')
    if show_plot:
        bk.io.show(grid)
    return grid


def _add_scatter(fig=None, source=None, **kwargs):
    """Add circle renderers for open and closed journals, inc hover + tap."""
    cmap_oa = bkt.factor_cmap('is_open', palette=['red', 'blue'], factors=['✔', ''])
    view_closed = bkm.CDSView(source=source,
                              filters=[bkm.BooleanFilter(~source.data['is_oa'])])
    view_open = bkm.CDSView(source=source,
                            filters=[bkm.BooleanFilter(source.data['is_oa'])])
    scatter_kws = dict(size=10, color=cmap_oa, source=source)
    closed_kws = dict(legend_label='closed', fill_alpha=0.3, view=view_closed)
    open_kws = dict(legend_label='open', fill_alpha=0.6, view=view_open)

    r1_open = fig.circle(**open_kws, **scatter_kws, **kwargs)
    r1_closed = fig.circle(**closed_kws, **scatter_kws, **kwargs)

    tooltips = [
        ('journal_name', '@journal_name'),
        ('CiteScore', '@cs_table'),
        ('Influence', '@inf_table'),
        ('CAT', '@CAT (@cited, @abstract, @title)'),
    ]
    fig.add_tools(bkm.HoverTool(renderers=[r1_open, r1_closed], tooltips=tooltips))  # , callback=cb_hover
    taptool = fig.select(type=bkm.TapTool)
    taptool.callback = bkm.OpenURL(url=_URL_NLM)
    fig.legend.click_policy = 'hide'


def plot_datatable(source_j, show_plot=False, table_kws=None):
    if not table_kws:
        table_kws = {}
    col_kws = {
        'default_sort': 'descending',
    }
    w_journal = 300
    w_md = 60
    w_sm = 40
    w_xs = 30

    col_param_dict = OrderedDict({
        'journal_name': ('journal', w_journal),
        'cs_table': ('CiteScore', w_md),
        'inf_table': ('Inf', w_sm),
        'is_open': ('OA', w_sm),
        'in_medline': ('ML', w_sm),
        'in_pmc': ('PMC', w_sm),
        # 'conf_title': ('cT', w_sm),
        # 'conf_abstract': ('cA', w_sm),
        'conf_pc': ('conf', w_sm),
        'sim_sum': ('∑sim', w_sm),
        'sim_max': ('⤒sim', w_sm),
        'CAT': ('CAT', w_sm),
        'cited': ('C', w_xs),
        'abstract': ('A', w_xs),
        'title': ('T', w_xs),
        'both': ('T&A', w_xs),

    })
    index_width = 0  # setting index_position to None
    table_width = sum([i[1] for i in col_param_dict.values()]) + index_width

    cell_template = """<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""
    # url_template = """<a href="<%= value %>" target="_blank"><%= value %></a>"""
    format_dict = {
        'journal': bkm.widgets.HTMLTemplateFormatter(template=cell_template),
        'OA': bkm.widgets.StringFormatter(),
        'ML': bkm.widgets.StringFormatter(),
        'PMC': bkm.widgets.StringFormatter(),
        'CiteScore': bkm.widgets.StringFormatter(),
        'Inf': bkm.widgets.StringFormatter(),
    }
    col_names = {col: col_param_dict[col][0] for col in col_param_dict}
    table_cols = OrderedDict({
        col: dict(width=col_param_dict[col][1],
                  formatter=format_dict.get(col_names[col],
                                            bkm.widgets.NumberFormatter()),
                  **col_kws) for col in col_param_dict})
    # jfp = jf[col_param_dict].rename(columns=col_names)

    # LINK_COLS = []
    columns = []  # FOR DataTable
    for col in col_param_dict:
        columns.append(bkm.widgets.TableColumn(
            field=col, title=col_param_dict[col][0],
            **table_cols[col]))

    data_table = bkm.widgets.DataTable(source=source_j, columns=columns,
                                       width=table_width, index_position=None,
                                       fit_columns=False,
                                       **table_kws)
    if show_plot:
        bkio.show(data_table)
    return data_table


def plot_icats(source_j, source_a, source_c, show_plot=False):
    """Create interactive ICATS scatter plot.

    Returns:
         (js, div) Bokeh javascript and html div elements if
         as_components=True, else notebook handle.
    """
    # if n_journals is None:
    #     n_journals = len(jf)
    width_l, width_m, width_r = 300, 120, 300
    box_height = 0.95
    TOOLS = "ypan,ywheel_zoom,reset,tap"

    text_props = {"text_align": "center", "text_baseline": "middle",
                  'text_color': '#000000', 'text_font_size': '10pt'}
    factors = ['cited', 'title', 'abstract']
    stack_factors = ['cited', 'title_only', 'abstract_only', 'both']
    categ_colors = dict(zip(stack_factors, bk.palettes.Colorblind4))
    categ_hex = {i: get_color_hex(categ_colors[i]) for i in categ_colors}
    categ_hex.update({'title': categ_hex['title_only'],
                      'abstract': categ_hex['abstract_only']})
    a_colors = [categ_hex[i] for i in stack_factors]
    # box_colors = [categ_hex[i] for i in factors]

    n_journals = source_j.data['index'].size
    plot_height = 36 * n_journals + 38

    view_aa = bkm.CDSView(source=source_a, filters=[bkm.BooleanFilter(source_a.data['in_abstract'])])
    view_at = bkm.CDSView(source=source_a, filters=[bkm.BooleanFilter(source_a.data['in_title'])])

    # JID / NAME TUPLES FOR Y RANGES
    jfs = source_j.to_df()  # temporary dataframe to simplify calculations
    impact_max = jfs['citescore'].max()
    # n_articles_max = jfs[factors].sum(axis=1).max()
    jname_tuples = [tuple(i) for i in jfs[['jid', 'abbr']].values][::-1]
    jname_factors2 = bkm.FactorRange(factors=jname_tuples, bounds=(-200, 50),
                                     factor_padding=0, group_padding=0)
    jname_factors = jname_factors2  #bkm.FactorRange(factors=[i[0] for i in jname_tuples])

    # MIDDLE SQUARES
    p = bkp.figure(
        tools=TOOLS,
        x_range=factors,
        y_range=jname_factors,
        plot_width=width_m, plot_height=plot_height,
        x_axis_location="above")
    # INDIVIDUAL ARTICLE RECT GLYPHS
    r_ac = p.rect(y='jid', x='loc_cited', color=categ_hex['cited'], width=0.95, height=box_height, fill_alpha=0.3, source=source_c)
    r_aa = p.rect(y='jid', x='loc_abstract', color=categ_hex['abstract'], width=0.95, height=box_height, fill_alpha=0.3, source=source_a, view=view_aa)
    r_at = p.rect(y='jid', x='loc_title', color=categ_hex['title'], width=0.95, height=box_height, fill_alpha=0.3, source=source_a, view=view_at)
    # OVERLAYED TEXT GLYPHS
    p.text(y='jid', x='loc_cited', text='cited', source=source_j, **text_props)
    p.text(y='jid', x='loc_abstract', text='abstract', source=source_j, **text_props)
    p.text(y='jid', x='loc_title', text='title', source=source_j, **text_props)

    # LEFT HAND SIDE: IMPACT
    p_l = bkp.figure(tools=TOOLS,
                     x_range=(impact_max, 0), y_range=p.y_range,
                     plot_width=width_l, plot_height=plot_height,
                     x_axis_label='Impact', x_axis_location="above")
    r_i = p_l.hbar(y='jid', height=0.4, left=0, right='citescore', source=source_j)

    # RIGHT HAND SIDE: SCATTER AND WHISKER
    p_r = bkp.figure(
        tools=TOOLS, x_range=(0, 105),  y_range=p.y_range,
        y_axis_location='right', plot_width=width_r, plot_height=plot_height,
        x_axis_label='Similarity', x_axis_location="above")
    p_r.add_layout(bkm.Whisker(source=source_j, base='jid', dimension='width',
                               upper="sim_max", lower="sim_min"))
    factor_cm = bkt.factor_cmap('categ', palette=a_colors, factors=stack_factors)
    r_as = p_r.circle(y=bkt.jitter('jid', width=0.5, range=p_r.y_range),
                      x='sim_max', source=source_a,  # x_range_name='ax_sim',
                      size=10, alpha=0.5, color=factor_cm,)
    taptool = p_r.select(type=bkm.TapTool)
    taptool.callback = bkm.OpenURL(url='@url')

    # HOVERTOOLS
    cite_cols_dict = {'use_year': 'year',
                      'use_article_title': 'title',
                      'use_authors': 'authors'}
    tooltips_c = [(cite_cols_dict[i], f"@{i}") for i in cite_cols_dict]
    hover_c = bkm.HoverTool(renderers=[r_ac], tooltips=tooltips_c)

    impact_dict = {'journal_name': 'Journal',
                   'citescore': 'CiteScore',
                   'influence': 'Influence',
                   'tags': 'tags'}
    impact_tooltips = [(impact_dict[i], f"@{i}") for i in impact_dict]
    hover_i = bkm.HoverTool(renderers=[r_i], tooltips=impact_tooltips)

    a_cols_dict = {'year': 'year',
                   'title': 'title',
                   'authors_short': 'authors',
                   'url': 'url',
                   'sim_max': 'sim'}
    article_tooltips = [(a_cols_dict[i], f"@{i}") for i in a_cols_dict]  # [("(x,y)", "($x, $y)")]
    hover_a = bkm.HoverTool(renderers=[r_aa, r_at], tooltips=article_tooltips)
    hover_s = bkm.HoverTool(renderers=[r_as], tooltips=article_tooltips)

    p_l.add_tools(hover_i)
    p.add_tools(hover_a, hover_c)
    p_r.add_tools(hover_s)

    # MINIMAL STYLING FOR AXES/TICKS
    for fig in [p, p_r, p_l]:
        fig.outline_line_color = None
        fig.grid.grid_line_color = None
        fig.axis.axis_line_color = None
        fig.axis.major_tick_line_color = None
        fig.axis.major_label_standoff = 0
    p_r.axis[1].group_text_color = '#ffffff'
    # p_l.axis[1].axis_line_color = '#000000'
    p_l.yaxis.visible = False
    p.yaxis.visible = False

    grid = bkl.gridplot([[p_l, p, p_r]], toolbar_location='left')

    if show_plot:
        bkp.show(grid)
    # js, div = bke.components(grid)
    return grid


def plot_vertical_stacked(jf, af, plot_width=500, n_journals=10):
    # categ_colors = {
    #     'cited': 'tab:green',
    #     'title': 'tab:blue',
    #     'abstract': 'tab:orange',
    #     'both': 'tab:purple',
    # }
    factors = ['cited', 'title_only', 'abstract_only', 'both']
    categ_colors = dict(zip(factors, bk.palettes.Colorblind4))
    categ_hex = {i: get_color_hex(categ_colors[i]) for i in categ_colors}

    jfs = jf.head(n_journals)
    afs = af[af.jid.isin(jfs.jid)].copy()

    name_var = 'abbr'
    # jfs['x_i'] = jfs[name_var].apply(lambda v: (v, 'i'))
    jfs['x_a'] = jfs[name_var].apply(lambda v: (v, 'a'))
    jfs['x_s'] = jfs[name_var].apply(lambda v: (v, 's'))
    afs['x_s'] = afs[name_var].apply(lambda v: (v, 's'))

    jnames = jfs[name_var]
    subcategs = ['a', 's']
    x_factors = list(product(jnames, subcategs))
    a_colors = [categ_hex[i] for i in factors]
    # x_a = [(i, 'a') for i in jnames for atype in a_types] # [('Eurosurv', 'c'), ('Eurosurv', 'a'),...

    # jfs['x_c'] = jfs[name_var].apply(lambda v: (v, 'c'))

    source = bkm.ColumnDataSource(jfs)
    source_a = bkm.ColumnDataSource(afs)

    hover_cols = ['journal_name', 'citescore', 'influence', 'cited', 'conf_weight',
                  'n_articles', 'confidence', 'sim_max', 'sims', 'pc_lower', 'is_oa']
    tooltips = [("index", "$index")] + [(i, f"@{i}") for i in hover_cols] # ("(x,y)", "($x, $y)")]

    p = bkp.figure(
            x_range=bkm.FactorRange(*x_factors),
            plot_width=plot_width, plot_height=500, y_range=(0, 9),
            y_axis_label='Article count',
            # tooltips=tooltips, y_axis_label('')
            # title=title, x_axis_label=x_var, y_axis_label=y_var)
        )
    # p.xaxis.major_label_orientation = "vertical"
    # p.xaxis.subgroup_label_orientation = "normal"
    p.xaxis.group_label_orientation = math.pi / 2  #0.8
    # p.xaxis.major_label_orientation = 0  #math.pi/2


    # SIM Y AXIS
    p.extra_y_ranges = {"ax_sim": bkm.Range1d(start=0, end=105)}
    p.add_layout(bkm.LinearAxis(y_range_name="ax_sim", axis_label="Similarity"), 'right',)

    # SIM WHISKER
    p.add_layout(bkm.Whisker(source=source, base="x_s", upper="sim_max",
                                   lower="sim_min", y_range_name='ax_sim'))
    # SIM CIRCLES
    r_a = p.circle(x=bkt.jitter('x_s', width=0.6, range=p.x_range),
                   #jitter('day', width=0.6, range=p.y_range)
                   y='sim_max', y_range_name='ax_sim', source=source_a,
                   size=10, alpha=0.5,
                   color=bkt.factor_cmap('categ', palette=a_colors, factors=factors)
                   )

    # IMPACT
    p.extra_y_ranges.update({"ax_impact": bkm.Range1d(start=0, end=10)})
    # p.add_layout(bkm.LinearAxis(y_range_name="ax_impact"), 'center')
    # p.vbar('x_i', width=0.9, top='citescore', source=source, y_range_name='ax_impact')

    # ARTICLE COUNTS
    p.vbar_stack(factors, x='x_a', width=0.9, source=source,
                 color=a_colors)  # legend_label=factors,
    hover_cols_a = ['year', 'title', 'authors', 'a_id', 'url']
    hover_a = bkm.HoverTool(renderers=[r_a],
                            tooltips=[(i, f"@{i}") for i in hover_cols_a])
    p.add_tools(hover_a)

    legend_items = []
    for factor in factors:
        hex = categ_hex[factor]
        r_temp = p.vbar(x=[0], width=1, bottom=0, top=[0], color=hex, visible=False)
        legend_items.append((factor, [r_temp]))
    legend = bkm.Legend(items=legend_items)

    p.add_layout(legend, 'left')



    t = bkp.show(p, notebook_handle=True)
    return t


def get_color_hex(color_name):
    return mpl.colors.rgb2hex(mpl.colors.to_rgba(color_name)[:3])
