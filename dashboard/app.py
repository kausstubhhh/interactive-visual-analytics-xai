from dash import Dash
import plotly.io as pio

from layout import create_layout
from callbacks import register_callbacks

# Configure Plotly
pio.templates.default = "plotly"


app = Dash(__name__, suppress_callback_exceptions=True)

app.title = "Interactive Visual Analytics Dashboard"

app.layout = create_layout()

register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True)
