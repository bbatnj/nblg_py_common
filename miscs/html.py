import base64
from io import BytesIO

import numpy as np
import pandas as pd

from common.miscs.basics import format_df_nums


def df_to_html(df, title, dp=2):
    df = df.copy()
    for c in df:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].apply(lambda x: format_col(x, dp=dp))

    txt = f'<h2>{title}</h2>'
    txt += df.to_html().replace('<table border="1" class="dataframe">',
                                '<table class="blueTable" style="height: 164px;" width="369">')
    return txt

def title_to_html(title: str):
    msg = ""
    msg += "<p>"
    msg += f"<h2>{title}</h2>"
    msg += "</p>"

    return msg


def pph(df, title, level=3, show_result=True): #pretty print html
    # pretty print html
    if not show_result:
        return
    from IPython.display import display, HTML
    title_html = f'<h{level}>{title}</h{level}>'
    display(HTML(title_html))

    if df is not None:
        display(format_df_nums(df))

def fig_to_html(fig, title, width: int = 1000):
    saved_path = BytesIO()
    fig.savefig(saved_path)
    encoded = base64.b64encode(saved_path.getvalue()).decode()
    html = title_to_html(title) + f'<img src="data:image/image/png;base64,{encoded}" width="{width}">' + "\n"
    '''
    encoded = mpld3.fig_to_html(fig)
    html = title_to_html(title) + f"<div class='figure-container'> {encoded} </div>" + "\n"
    '''
    return html

def format_col(x, dp=2):
    if np.abs(x) < 1e3 and x == round(x):
        return f'{round(x)}'
    if np.abs(x) <= 1e-3:
        return f'{x * 1e4 :.2f} bps'
    elif np.abs(x) >= 1e6:
        return f'{x / 1e6 :.{dp}f} M'
    elif np.abs(x) >= 1e3:
        return f'{x / 1e3 :.{dp}f} K'
    else:
        return f'{x :.{dp}f}'

