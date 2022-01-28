import os
import logging
import tempfile
from io import BytesIO
from datetime import datetime

from flask import render_template, redirect, url_for, flash, session, \
    current_app, abort, request, send_file

from . import main
from .forms import UploadForm

from ...mapping import run_queries
from ...plot import get_bokeh_components
from ...demo import get_demo_data_with_prefix, update_demo_plot
from ...ref_loading import BadRisException
from ...reference import MT

_logger = logging.getLogger(__name__)


@main.route('/', methods=['GET', 'POST'])
def index():
    last_title = session.get('title', None)
    return render_template('home.html', title='Journal Targeter',
                           last_title=last_title,)


@main.route('/results', methods=['GET', 'POST'])
def results():
    if 'refs_df' not in session:
        flash("Let's start by uploading some data.")
        return redirect(url_for('.search'))
    pref_kwargs = _get_pref_dict_from_session()
    js, divs = get_bokeh_components(session['jf'], session['af'],
                                    session['refs_df'], store_prefs=True,
                                    **pref_kwargs)
    skip_refs = session['refs_df'] is None
    return render_template('index.html', title='Results',
                           query_title=session['title'],
                           query_abstract=session['abstract'],
                           query_ris=session['ris_name'],
                           bokeh_js=js,
                           bokeh_divs=divs,
                           skip_refs=skip_refs,
                           )


@main.route('/demo', methods=['GET', 'POST'])
@main.route('/demo/<demo_prefix>', methods=['GET', 'POST'])
def demo(demo_prefix=None):
    if demo_prefix is None:
        demo_prefix = current_app.config['DEMO_PREFIX']
    # Temporary demo update process
    force_update = (os.environ.get('FORCE_DEMO_UPDATE', 'false').lower()
                    in ['true', '1'])
    if force_update:
        update_demo_plot(demo_prefix)
    data = get_demo_data_with_prefix(demo_prefix)
    if data is None:
        flash("Requested demo not found", "error")
        return redirect(url_for('.index'))
    return render_template('index.html', title='Demo',
                           query_title=data['title'],
                           query_abstract=data['abstract'],
                           query_ris=data['ris_name'],
                           bokeh_js=data['bokeh_js'],
                           bokeh_divs=data['bokeh_divs'],
                           )


@main.route('/download/', methods=['GET'])
@main.route('/download/<category>', methods=['GET'])
def download(category=None):
    # check for populated session and matching data pickle
    category = 'session' if not category else category
    if category not in {'demo', 'session'}:
        abort(404)
    if category == 'session':
        if 'refs_df' not in session:
            abort(404)  # session is not populated
        data = session.copy()
        pref_kwargs = _get_pref_dict_from_session()
        js, divs = get_bokeh_components(session['jf'], session['af'],
                                        session['refs_df'], **pref_kwargs)
        data.update({'bokeh_js': js, 'bokeh_divs': divs})
        time_str = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
        out_name = f"jot_results_{time_str}.html"
        page_title = 'Results (local)'
        skip_refs = session['refs_df'] is None
    else:  # Use demo data
        prefix = request.args.get('prefix', current_app.config['DEMO_PREFIX'])
        data = get_demo_data_with_prefix(prefix)
        if not data:
            abort(404)
        out_name = f'demo_{prefix}.html'
        page_title = f'{prefix} (local)'
        skip_refs = False
    html = render_template('index.html', title=page_title,
                           standalone=True,
                           query_title=data['title'],
                           query_abstract=data['abstract'],
                           query_ris=data['ris_name'],
                           bokeh_js=data['bokeh_js'],
                           bokeh_divs=data['bokeh_divs'],
                           skip_refs=skip_refs,
                           )
    tmp_bytes = BytesIO()
    tmp_bytes.write(html.encode('utf-8'))
    tmp_bytes.seek(0)
    return send_file(tmp_bytes, mimetype='html', as_attachment=True,
                     attachment_filename=out_name, cache_timeout=-1)


@main.route('/search', methods=['GET', 'POST'])
def search():
    form = UploadForm()
    if form.validate_on_submit():
        # flash('Form accepted!')
        # ACT ON FORM
        # ...store query_title, query_abstract in session
        title, abstract = form.title.data, form.abstract.data
        session['title'] = title
        session['abstract'] = abstract
        if form.ref_file.has_file():
            ref_obj = form.ref_file.raw_data[0]  # type: werkzeug.FileStorage
            fs = ref_obj.stream
            fs.seek(0)
            tempf = tempfile.TemporaryFile(mode='w+t', encoding='utf8')
            tempf.writelines([i.decode('utf8') for i in fs.readlines()])
            tempf.seek(0)
            session['ris_name'] = ref_obj.filename
        else:
            tempf = None
            session['ris_name'] = None
        # Send queries to Jane and build jf, af, refs_df
        try:
            jf, af, refs_df = run_queries(
                query_title=title, query_abstract=abstract, ris_path=tempf)
            session.update({'jf': jf, 'af': af, 'refs_df': refs_df})
        except BadRisException as e:
            msg = "Invalid RIS file. Please modify and try again."
            if str(e):
                msg = ' '.join([msg, str(e)])
            flash(msg)
            return render_template('upload.html', form=form)
        finally:
            if form.ref_file.has_file():
                tempf.close()
                fs.close()
        return redirect(url_for('.results'))
    # if 'title' in session:
    #     form.title.data = session['title']
    # if 'abstract' in session:
    #     form.abstract.data = session['abstract']
    return render_template('upload.html', form=form, title='Search')


@main.route('/prefer', methods=['POST'])
def set_preference():
    # allowed args: metric, weight
    if 'metric' in request.form:
        val = request.form['metric']
        if val in MT.metric_list:
            session['metric'] = val
            return {'msg': f'Saved {val} as preferred metric.'}
        else:
            return {'msg': f'Unknown metric: {val}.'}
    if 'weight' in request.form:
        val = request.form['weight']
        val = round(float(val), 2)  # round to 2dp
        session['weight'] = val
        return {'msg': f'Saved {val} as preferred weight.'}


def _get_pref_dict_from_session():
    pref_dict = dict()
    if 'metric' in session:
        pref_dict['pref_metric'] = session['metric']
    if 'weight' in session:
        pref_dict['pref_weight'] = session['weight']
    return pref_dict
