import os
import logging
import tempfile

from flask import render_template, redirect, url_for, flash, session, current_app

from . import main
from .forms import UploadForm

from ...mapping import run_queries
from ...plot import get_bokeh_components
from ...demo import get_demo_data_with_prefix
from ...ref_loading import BadRisException

_logger = logging.getLogger(__name__)


@main.route('/', methods=['GET', 'POST'])
def index():
    # check for populated session
    has_results = 'bokeh_js' in session
    last_title = session.get('title', None)
    return render_template('home.html', title='Journal Targeter',
                           last_title=last_title,
                           )


@main.route('/results', methods=['GET', 'POST'])
def results():
    # check for populated session and matching data pickle
    if 'bokeh_js' not in session:
        flash("Let's start by uploading some data.")
        return redirect(url_for('.search'))
    # Session is populated.

    bokeh_js = session['bokeh_js']
    if 'bokeh_divs' not in session:
        flash("Previous session has expired. Please run a new search.")
        return redirect(url_for('.search'))
    else:
        bokeh_divs = session['bokeh_divs']

    return render_template('index.html', title='Results',
                           query_title=session['title'],
                           query_abstract=session['abstract'],
                           query_ris=session['ris_name'],
                           bokeh_js=bokeh_js,
                           bokeh_divs=bokeh_divs,
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
        from ...demo import update_demo_plot
        update_demo_plot(demo_prefix)
    data = get_demo_data_with_prefix(demo_prefix)
    return render_template('index.html', title='Demo',
                           query_title=data['title'],
                           query_abstract=data['abstract'],
                           query_ris=data['ris_name'],
                           bokeh_js=data['bokeh_js'],
                           bokeh_divs=data['bokeh_divs'],
                           )


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
        ref_obj = form.ref_file.raw_data[0]  # type: werkzeug.FileStorage
        fs = ref_obj.stream
        fs.seek(0)
        tempf = tempfile.TemporaryFile(mode='w+t', encoding='utf8')
        tempf.writelines([i.decode('utf8') for i in fs.readlines()])
        tempf.seek(0)

        session['ris_name'] = ref_obj.filename
        # Send queries to Jane and build jf, af, refs_df
        try:
            jf, af, refs_df = run_queries(
                query_title=title, query_abstract=abstract, ris_path=tempf)
        except BadRisException as e:
            msg = "Invalid RIS file. Please modify and try again."
            if str(e):
                msg = ' '.join([msg, str(e)])
            flash(msg)
            return render_template('upload.html', form=form)
        finally:
            tempf.close()
            fs.close()

        js, divs = get_bokeh_components(jf, af, refs_df)
        session['bokeh_js'] = js
        session['bokeh_divs'] = divs
        return redirect(url_for('.results'))
    # if 'title' in session:
    #     form.title.data = session['title']
    # if 'abstract' in session:
    #     form.abstract.data = session['abstract']
    return render_template('upload.html', form=form, title='Search')
