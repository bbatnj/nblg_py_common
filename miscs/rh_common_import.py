from IPython.core.display import display, HTML
%matplotlib notebook
display(HTML("<style>.container { width:100% !important; } </style>"))
import pandas as pd
pd.set_option('display.max_rows', 2000)
import logging
import warnings

# Suppress debug messages
logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Suppress anything below ERROR level
# Suppress all warnings
warnings.filterwarnings("ignore")

from common.quants.pta import run_pta, run_pta_group, run_pta_group_old
from common.miscs.plot import plot_multiple_lines