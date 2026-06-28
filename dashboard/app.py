from pathlib import Path
import sys
# Project Root

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )
    

from dash import Dash
import plotly.io as pio

pio.templates.default = "plotly"

from layout import create_layout
from callbacks import register_callbacks


app = Dash(
    __name__,
    suppress_callback_exceptions=True
)

app.title = (
    "Interactive Visual Analytics Dashboard"
)

app.layout = create_layout()

register_callbacks(app)


if __name__ == "__main__":
    app.run(
        debug=True
    )

