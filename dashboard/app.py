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