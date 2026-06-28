import plotly.graph_objects as go
import plotly.express as px


def create_empty_figure(message):
    """
    Create an empty Plotly figure with a centered message.
    """

    fig = go.Figure()

    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 18},
            }
        ],
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
    )

    return fig


def create_confusion_figure(df):
    """
    Create confusion matrix heatmap.
    """

    # Rearrange matrix:
    # [[TP, FN],
    #  [FP, TN]]

    matrix = [[df.iloc[1, 1], df.iloc[1, 0]], [df.iloc[0, 1], df.iloc[0, 0]]]

    labels = [["TP", "FN"], ["FP", "TN"]]

    annotations = [
        [f"{labels[i][j]}<br>{matrix[i][j]}" for j in range(2)] for i in range(2)
    ]

    fig = px.imshow(
        matrix,
        x=["Positive", "Negative"],
        y=["Positive", "Negative"],
        color_continuous_scale="Blues",
        text_auto=False,
        aspect="equal",
    )

    fig.update_traces(text=annotations, texttemplate="%{text}")

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=550,
    )

    return fig
