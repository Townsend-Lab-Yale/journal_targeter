# Jot: Journal Targeter

Jot is a web app that identifies potential target journals for a manuscript,
based on the manuscript's title, abstract, and (optionally) references. Jot
gathers a wealth of data on journal quality, impact, fit, and open access
options that can be explored through linked, interactive visualizations.

To try it out, you have two options:
1. Visit the website: Jot is available at https://jot.publichealth.yale.edu
2. Run your own Jot server. Instructions below.


## Contents
<!-- MarkdownTOC autolink="true" -->

- [About Jot](#about-jot)
- [How to run your own server](#how-to-run-your-own-server)
   - [Installation](#installation)
   - [Command-line interface \(CLI\)](#command-line-interface-cli)
      - [Quick start](#quick-start)
      - [Available commands](#available-commands)
- [Modifying the code](#modifying-the-code)

<!-- /MarkdownTOC -->

## About Jot

Jot builds upon the API of Jane (Journal/Author Name Estimator,
https://jane.biosemantics.org/) to identify PubMed articles that are similar
in content to a manuscript's title and abstract. Jot gathers these articles
and their similarity scores together with manuscript citations and a journal
metadata assembled from the National Library of Medicine (NLM) Catalog, the
Directory of Open Access Journals (DOAJ), Sherpa Romeo, and impact metric
databases. The result is a personalised, multi-dimensional data set that can
be navigated through a series of linked, interactive plots and tables,
allowing an author to sort and study journals according to the attributes most
important to them.


## How to run your own server

### Installation

To run a Jot server, you first need to install the python package `journal_targeter`
on your machine. You have a few options:

1. (Easiest) Install from **PyPI**.
   1. To install directly into your current Python (virtual) environment, run:
       ```shell
       pip install journal_targeter
       ```
   2. For the convenience of an app-specific environment, use 
    [pipx](https://github.com/pypa/pipx):
       ```shell
       pipx install journal_targeter
       ```
2. Install from source code.
   1. In your terminal, clone the `journal_targeter` repository to a convenient 
      for long-term storage, and `cd` into the new directory.
   2. (Optional/Recommended) Create and activate a new virtual environment using 
       [venv](https://docs.python.org/3/library/venv.html) or 
       [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
       - With conda/miniconda installed, you can easily create an environment 
         with the required dependencies using the provided `environment.yaml` file:
         ```shell
         conda env create -n jot -f environment.abstract.yaml
         ```
         Activate this environment (necessary each time you want to run Jot) with:
         ```shell
         conda activate jot
         ```
   3. To install dependencies (if you didn't use the conda step above), run:
      ```shell
      pip install -r requirements.txt
      ```
   4. Finally, install the package in development mode using:
      ```shell
      python setup.py develop
      ```
   
### Command-line interface (CLI)

#### Quick start

With the Python package installed as above, an executable called `journal_targeter`
should now be available on your path. Without any further configuration, you can 
try out the server using:
```shell
journal_targeter flask run
```
This will set up the application (copying/building key data in an application 
support folder) then start a Flask development server. The app will be available 
in your browser at `http://127.0.0.1:5000/`.

#### Available commands

Run `journal_targeter` without arguments to see a list of commands. Add the 
'--help' flag after a command name to get more information on the command.

```
Usage: journal_targeter [OPTIONS] COMMAND [ARGS]...

Options:
  --verbose / --quiet
  --help               Show this message and exit.

Commands:
  build-demo      (Re)build demo data.
  flask           Serve using Flask cli.
  gunicorn        Serve using gunicorn.
  lookup-journal  Find journal metadata using title and optional ISSNs.
  match           Run search and save results as html file.
  setup           Set up environment variables for running server.
  update-sources  Update data sources, inc NLM, DOAJ, Sherpa Romeo, etc.
```

To configure the application, the `setup prompt` command will walk you through the 
creation of a configuration `.env` file.
```shell
journal_targeter setup prompt
```

To serve the app, you can use the Flask development server 
(not recommended for production settings) or `gunicorn` (Mac/Unix/Linux):
```shell
# Flask, running on port 5005
journal_targeter flask run -p 5005 -h 0.0.0.0

# ...or gunicorn, running on port 5005 with 1 gevent worker
journal_targeter gunicorn -b 127.0.0.1:5005 -w 1 -k gevent
```

You can update data sources without waiting for a new `journal_targeter` release.
Examples:
```shell
# Update NLM catalog data, adding --clear-metadata to start with the latest 
# metadata for all journals. (~13min) 
journal_targeter update-sources --update-nlm --clear-metadata

# Update DOAJ data from a downloaded CSV (https://doaj.org/csv), with 5 cores for matching (~4min)
journal_targeter update-sources --ncpus 5 -d journalcsv__doaj_20211028_1036_utf8.csv

# Update Sherpa Romeo data, downloaded via API (requires API KEY), with 5 cores, 
# skipping the optional NLM update
journal_targeter update-sources --ncpus 5 --skip-nlm --romeo

# Update the Scopus metrics from a downloaded 'source titles and metrics' file
# via https://www.scopus.com/sources 
journal_targeter update-sources --ncpus 5 --scopus-path "CiteScore 2011-2020 new methodology - May 2021.xlsb"
```

## Modifying the code

This code comes with a [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html), so feel free to tinker and share under 
the license terms.

To enable the interactive debugger, set the FLASK_ENV variable to 'development':
```shell
FLASK_ENV=development journal_targeter flask run
```