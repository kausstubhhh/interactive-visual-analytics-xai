from dash import html


# ==========================================
# Dashboard Cards
# ==========================================

def create_card(
    title,
    *children,
    class_name=""
):

    classes = "dashboard-card"

    if class_name:
        classes += f" {class_name}"

    return html.Div(

        [

            html.H3(
                title,
                className="card-title"
            ),

            *children

        ],

        className=classes

    )


# ==========================================
# Workflow Components
# ==========================================

def workflow_step(
    label,
    state="pending"
):

    state_icons = {
        "completed": "✓",
        "active": "●",
        "pending": "○"
    }

    return html.Div(

        [

            html.Div(
                state_icons[state],
                className=f"workflow-icon {state}"
            ),

            html.Div(
                label,
                className="workflow-text"
            )

        ],

        className="workflow-step"

    )


def create_workflow_banner():

    return html.Div(

        [

            workflow_step(
                "Upload",
                "active"
            ),

            html.Div(className="workflow-line"),

            workflow_step(
                "Analyse"
            ),

            html.Div(className="workflow-line"),

            workflow_step(
                "Compare"
            ),

            html.Div(className="workflow-line"),

            workflow_step(
                "Explain"
            )

        ],

        className="workflow-banner"

    )