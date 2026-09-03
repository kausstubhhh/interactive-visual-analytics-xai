# Interactive Visual Analytics for Explaining and Comparing Classification Models

**MSc Project -- Advanced Computer Science (Data Analytics)**

This repository contains the implementation of an interactive visual
analytics system for exploring, explaining, and comparing binary
classification models. The project was developed as an MSc research and
development project and focuses on selected Explainable Artificial
Intelligence (XAI) tasks within the model diagnosis, model improvement,
and model selection stages of the MAVIS framework.

The system combines machine-learning evaluation, SHAP-based
explanations, error analysis, and interactive visualisation in a single
Python application.

------------------------------------------------------------------------

## 1. Project Overview

Classification models can provide strong predictive performance while
offering limited insight into how predictions are produced. Standard
metrics such as accuracy, precision, recall, F1-score, and ROC-AUC
quantify performance, but they do not by themselves explain feature
influence, individual decisions, or differences in model behaviour.

This project addresses that problem through an interactive visual
analytics workflow that allows the user to:

-   evaluate and compare classification models;
-   inspect global feature importance;
-   investigate misclassified instances;
-   examine individual prediction explanations;
-   compare model behaviour across datasets;
-   move between different analytical views without manually reproducing
    the underlying analysis.

The implemented system evaluates two classification models:

-   **Logistic Regression**
-   **Random Forest**

and supports two tabular datasets:

-   **HELOC**
-   **Bank Marketing**

The implementation uses **Python**, **scikit-learn**, **SHAP**,
**Dash**, and **Plotly**.

The project specification defines the system around selected XAI tasks
including feature importance, class separation, misclassification
analysis, decision behaviour, and model comparison. Stakeholder
communication (Stage 4 of MAVIS) is outside the project scope.

------------------------------------------------------------------------

## 2. Main Features

### 2.1 Dataset management

The application provides a dedicated Dataset Management view. The
analysis workflow is designed around a dataset, its target column,
schema detection, validation, and preprocessing before model training.

The supplied datasets are:

  Dataset          File                                   Target column
  ---------------- -------------------------------------- -------------------
  HELOC            `data/raw/heloc_dataset_v1.csv`        `RiskPerformance`
  Bank Marketing   `data/raw/bank-additional-full.xlsx`   `y`

The data layer contains separate components for loading, schema
detection, validation, and preprocessing.

### 2.2 Performance comparison

The Performance Comparison view supports comparison of the two models
using:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   Confusion-matrix values

These measures provide the quantitative baseline for subsequent XAI
analysis.

### 2.3 Feature importance

The Feature Importance view uses SHAP-derived global feature importance
to identify which features contribute most strongly to model
predictions.

The implementation generates feature-importance tables for both models
and both datasets. SHAP analysis uses a representative sample rather
than necessarily explaining every test instance.

### 2.4 Misclassification analysis

The Misclassification view focuses on prediction errors rather than
aggregate performance alone.

It supports analysis of:

-   total incorrect predictions;
-   false positives;
-   false negatives;
-   differences in error patterns between models.

This is useful when two models have similar accuracy but make different
types of mistakes.

### 2.5 Decision Behaviour

The Decision Behaviour view provides local explanation of a selected
prediction using SHAP values.

This complements global feature importance:

-   **Global SHAP:** which features are influential across the analysed
    data;
-   **Local SHAP:** which features contribute to a particular
    prediction.

------------------------------------------------------------------------

## 3. System Architecture

The repository follows a modular, layered structure.

``` text
Raw datasets
     |
     v
Data loading and validation
     |
     v
Schema detection and preprocessing
     |
     v
Train / test split
     |
     v
Logistic Regression + Random Forest
     |
     +--------------------+
     |                    |
     v                    v
Model evaluation       SHAP analysis
     |                    |
     |              +-----+------+
     |              |            |
     v              v            v
Performance      Global SHAP   Local SHAP
metrics          importance    explanation
     |              |            |
     +--------------+------------+
                    |
                    v
          Interactive Dash dashboard
                    |
       +------------+------------+
       |            |            |
       v            v            v
 Performance   Importance   Misclassification
                    |
                    v
             Decision Behaviour
```

The architecture separates the dashboard interface from analytical
services and core processing components. This makes individual parts of
the workflow easier to test, maintain, and extend.

------------------------------------------------------------------------

## 4. Repository Structure

``` text
interactive-visual-analytics-xai/
│
├── dashboard/
│   ├── app.py
│   ├── bootstrap.py
│   ├── callbacks.py
│   ├── layout.py
│   ├── narrative.py
│   ├── theme.py
│   ├── ui_components.py
│   │
│   ├── callback_modules/
│   │   ├── dataset_callbacks.py
│   │   ├── performance_callbacks.py
│   │   ├── feature_importance_callbacks.py
│   │   ├── misclassification_callbacks.py
│   │   └── decision_behaviour_callbacks.py
│   │
│   ├── components/
│   │   ├── dataset_management_tab.py
│   │   ├── performance_tab.py
│   │   ├── feature_importance_tab.py
│   │   ├── misclassification_tab.py
│   │   └── decision_behaviour_tab.py
│   │
│   ├── helpers/
│   └── visualisations/
│       ├── comparison_plots.py
│       ├── decision_behaviour.py
│       ├── feature_importance.py
│       ├── misclassification.py
│       └── performance_plots.py
│
├── src/
│   ├── config/
│   │   └── settings.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocess.py
│   │   ├── schema_detector.py
│   │   └── validator.py
│   │
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   └── trainer.py
│   │
│   ├── evaluation/
│   │   ├── confusion.py
│   │   ├── evaluator.py
│   │   ├── exporter.py
│   │   ├── metrics.py
│   │   └── misclassification.py
│   │
│   ├── explainability/
│   │   ├── exporter.py
│   │   ├── importance.py
│   │   ├── local_explanations.py
│   │   ├── shap_explainer.py
│   │   └── shap_generator.py
│   │
│   └── services/
│       ├── analysis_pipeline.py
│       ├── dataset_service.py
│       ├── evaluation_service.py
│       ├── model_service.py
│       ├── verification_service.py
│       └── xai_service.py
│
├── scripts/
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── generate_feature_importance.py
│   ├── generate_misclassification_analysis.py
│   ├── generate_local_explanations.py
│   └── test_shap.py
│
├── data/
│   ├── raw/
│   │   ├── heloc_dataset_v1.csv
│   │   └── bank-additional-full.xlsx
│   └── exports/
│       ├── confusion/
│       ├── errors/
│       ├── local_explanations/
│       ├── shap/
│       └── evaluation_summary.csv
│
├── notebooks/
├── tests/
├── docs/
│   └── architecture/
│       └── system_architecture.md
│
├── files/
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## 5. Requirements

### Software

The project requires:

-   Python 3.11 or later is recommended for the supplied dependency set;
-   Git;
-   a modern web browser.

The complete Python dependency set is specified in `requirements.txt`.

Important libraries include:

-   Dash
-   Plotly
-   pandas
-   NumPy
-   scikit-learn
-   SHAP
-   SciPy
-   openpyxl
-   pytest

The repository pins package versions in `requirements.txt` to improve
reproducibility.

------------------------------------------------------------------------

# 6. Installation

## 6.1 Clone the repository

``` bash
git clone https://github.com/kausstubhhh/interactive-visual-analytics-xai.git
cd interactive-visual-analytics-xai
```

## 6.2 Create a virtual environment

### Windows

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, the terminal should indicate that the virtual
environment is active.

## 6.3 Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Installation may take several minutes because the project includes
scientific-computing and SHAP dependencies.

## 6.4 Verify the installation

Run:

``` bash
python --version
pip --version
```

Then run the automated tests:

``` bash
pytest -q
```

A successful test run confirms that the core components can be imported
and exercised in the configured environment.

------------------------------------------------------------------------

# 7. Running the Interactive Dashboard

The main entry point is:

``` text
dashboard/app.py
```

From the repository root, run:

``` bash
python dashboard/app.py
```

The Dash application starts a local development server. Open the local
address displayed in the terminal, normally:

``` text
http://127.0.0.1:8050/
```

The application currently runs with Dash's development/debug mode
enabled.

To stop the application, return to the terminal and press:

``` text
Ctrl + C
```

------------------------------------------------------------------------

# 8. How to Use the Dashboard

The dashboard is organised into five main views:

1.  Dataset Management
2.  Performance Comparison
3.  Feature Importance
4.  Misclassification
5.  Decision Behaviour

The recommended workflow is to start with Dataset Management and then
move from quantitative performance to increasingly detailed
explanations.

------------------------------------------------------------------------

## 8.1 Step 1 -- Dataset Management

Open **Dataset Management** first.

The supplied project datasets are already available under:

``` text
data/raw/
```

The default datasets are:

``` text
data/raw/heloc_dataset_v1.csv
data/raw/bank-additional-full.xlsx
```

For the supplied datasets, the target columns are:

``` text
HELOC           -> RiskPerformance
Bank Marketing  -> y
```

The data-processing pipeline performs schema detection and preprocessing
before model training.

The preprocessing stage is important because the two datasets have
different structures. HELOC is primarily numerical, while Bank Marketing
contains mixed feature types.

### Expected outcome

After a dataset is successfully prepared, the application can perform
model analysis and populate the other dashboard views.

------------------------------------------------------------------------

## 8.2 Step 2 -- Performance Comparison

Select **Performance Comparison**.

This view provides the quantitative baseline for comparing Logistic
Regression and Random Forest.

The main metrics are:

### Accuracy

The proportion of all predictions that are correct.

### Precision

The proportion of predicted positive cases that are actually positive.

### Recall

The proportion of actual positive cases that are correctly identified.

### F1-score

The harmonic mean of precision and recall.

### ROC-AUC

The area under the receiver operating characteristic curve, measuring
discrimination across classification thresholds.

The view should be interpreted comparatively rather than by looking at
accuracy alone. Two models can have similar accuracy while producing
different precision, recall, and error profiles.

------------------------------------------------------------------------

## 8.3 Step 3 -- Feature Importance

Select **Feature Importance**.

This view uses SHAP-based global explanations.

The analysis pipeline:

1.  prepares the dataset;
2.  trains the selected classification model;
3.  creates a SHAP explainer;
4.  generates SHAP values;
5.  aggregates the absolute contribution of features;
6.  ranks the features;
7.  presents the resulting importance information.

The standalone feature-importance script uses a representative sample of
up to 500 test rows for SHAP analysis.

### How to interpret the result

A feature with a larger mean absolute SHAP contribution has a larger
influence on model output within the analysed sample.

This does **not** mean that the feature is causally responsible for the
target outcome. It describes its contribution to the model's
predictions.

Use this view to answer questions such as:

-   Which features are most influential?
-   Do the two models prioritise the same features?
-   Where do the feature rankings diverge?
-   Does similar predictive performance correspond to similar feature
    usage?

------------------------------------------------------------------------

## 8.4 Step 4 -- Misclassification

Select **Misclassification**.

This view focuses on incorrect predictions.

The analysis separates errors into:

-   True Positives (TP)
-   True Negatives (TN)
-   False Positives (FP)
-   False Negatives (FN)

The view is particularly useful when the models have similar aggregate
performance.

For example, two models can have almost identical accuracy but one may
produce more false positives while the other produces more false
negatives. This difference would not be visible from accuracy alone.

Use this view to investigate:

-   which model produces fewer total errors;
-   which model produces fewer false positives;
-   which model produces fewer false negatives;
-   whether the error distribution changes between datasets.

------------------------------------------------------------------------

## 8.5 Step 5 -- Decision Behaviour

Select **Decision Behaviour**.

This view provides a local explanation for an individual prediction.

The system compares the feature contributions associated with the
selected prediction for the two models.

A local SHAP explanation answers a different question from global
feature importance:

> Global importance asks which features matter generally; local
> explanation asks which features contributed to this particular
> prediction.

The explanation can therefore be used to investigate why two models
produce different decisions for the same input instance.

Positive and negative SHAP contributions should be interpreted relative
to the model's output and the SHAP baseline. They should not be
interpreted as causal effects.

------------------------------------------------------------------------

# 9. Running the Analytical Scripts

The repository also contains standalone scripts under `scripts/`. These
are useful when the dashboard is not required and the user wants to
reproduce particular analytical outputs.

Run the scripts from the **repository root**.

------------------------------------------------------------------------

## 9.1 Train the models

``` bash
python scripts/train_models.py
```

This script loads the project datasets, prepares the data, constructs
the Logistic Regression and Random Forest models, trains them, and
reports the resulting workflow.

The model definitions themselves are located in:

``` text
src/models/logistic_regression.py
src/models/random_forest.py
```

The common training logic is implemented in:

``` text
src/models/trainer.py
```

------------------------------------------------------------------------

## 9.2 Evaluate the models

``` bash
python scripts/evaluate_models.py
```

This script evaluates both models on both supplied datasets.

It reports:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   TN
-   FP
-   FN
-   TP

It also writes the combined evaluation results to:

``` text
data/exports/evaluation_summary.csv
```

------------------------------------------------------------------------

## 9.3 Generate global feature importance

``` bash
python scripts/generate_feature_importance.py
```

This script:

1.  loads each dataset;
2.  detects its schema;
3.  preprocesses the data;
4.  trains Logistic Regression and Random Forest;
5.  constructs SHAP explainers;
6.  calculates SHAP values for a representative test sample;
7.  calculates global feature importance;
8.  exports the results.

The SHAP sample size in the script is currently:

``` text
500 test instances
```

The output files are written to:

``` text
data/exports/shap/
```

with filenames following the pattern:

``` text
<dataset>_<model>_importance.csv
```

------------------------------------------------------------------------

## 9.4 Generate misclassification analysis

``` bash
python scripts/generate_misclassification_analysis.py
```

This script trains each model, compares predictions with the true
labels, calculates an error summary, and exports the results.

Output location:

``` text
data/exports/errors/
```

Files follow the pattern:

``` text
<dataset>_<model>_errors.csv
```

------------------------------------------------------------------------

## 9.5 Generate local explanations

``` bash
python scripts/generate_local_explanations.py
```

This script generates a local SHAP explanation for a test instance for
both models.

The current standalone script uses:

``` text
INSTANCE_INDEX = 0
```

Output location:

``` text
data/exports/local_explanations/
```

Files follow the pattern:

``` text
<dataset>_<model>_instance_0.csv
```

The dashboard's service layer can also generate local explanations as
part of the interactive XAI workflow.

------------------------------------------------------------------------

## 9.6 Run the SHAP test script

``` bash
python scripts/test_shap.py
```

This script is provided as an additional check of the SHAP
functionality.

------------------------------------------------------------------------

# 10. Running the Test Suite

The project contains automated tests under:

``` text
tests/
```

Run the complete suite with:

``` bash
pytest -q
```

The tests cover multiple parts of the system, including:

-   analysis pipeline;
-   confusion-matrix calculations;
-   dataset services;
-   evaluation;
-   exporting;
-   feature importance;
-   local explanations;
-   metrics;
-   misclassification;
-   model definitions;
-   SHAP exporting;
-   model training;
-   verification services.

Individual test files can also be run, for example:

``` bash
pytest tests/test_evaluator.py -q
```

or:

``` bash
pytest tests/test_local_explanations.py -q
```

------------------------------------------------------------------------

# 11. Data and Preprocessing

The data pipeline is implemented in:

``` text
src/data/
```

### `loader.py`

Responsible for reading supported dataset files.

### `schema_detector.py`

Detects relevant schema information, including feature and target
structure.

### `validator.py`

Provides dataset validation functionality.

### `preprocess.py`

Creates the preprocessing workflow used before model training.

The preprocessing stage is applied separately to training and test data
to avoid fitting preprocessing transformations on the test set.

For the Bank Marketing dataset, categorical variables require encoding
before they can be supplied to the machine-learning models. Numerical
data can also undergo the required preprocessing transformations.

The project deliberately keeps preprocessing separate from the dashboard
so that the same analytical workflow can be reused by scripts and
services.

------------------------------------------------------------------------

# 12. Model Implementation

Two models are implemented.

## Logistic Regression

Located at:

``` text
src/models/logistic_regression.py
```

Logistic Regression provides a linear classification baseline and is
useful for comparison against a non-linear ensemble model.

## Random Forest

Located at:

``` text
src/models/random_forest.py
```

Random Forest provides an ensemble-based non-linear classifier capable
of modelling more complex relationships.

Both models are created through dedicated builder functions and trained
through the shared trainer component:

``` text
src/models/trainer.py
```

This common interface allows the dashboard and analytical scripts to
treat the models consistently.

------------------------------------------------------------------------

# 13. Explainability Implementation

The XAI implementation is located in:

``` text
src/explainability/
```

### `shap_explainer.py`

Creates the SHAP explainer for a trained model.

### `shap_generator.py`

Generates SHAP values for the data being explained.

### `importance.py`

Converts SHAP values into global feature-importance information.

### `local_explanations.py`

Extracts feature contributions for an individual prediction.

### `exporter.py`

Writes explanation results to CSV files.

The service-level orchestration is provided by:

``` text
src/services/xai_service.py
```

The service supports both global feature-importance analysis and local
explanations.

------------------------------------------------------------------------

# 14. Evaluation Implementation

The evaluation components are located in:

``` text
src/evaluation/
```

The evaluator calculates the main classification metrics, while separate
components handle confusion matrices, misclassification analysis, and
result exporting.

This separation allows the same evaluation functions to be used by both
the standalone scripts and the interactive dashboard.

------------------------------------------------------------------------

# 15. Dashboard Implementation

The Dash application is located in:

``` text
dashboard/
```

The main entry point is:

``` text
dashboard/app.py
```

The application creates the Dash instance, sets the page title, creates
the layout, and registers the callbacks.

The main layout is defined in:

``` text
dashboard/layout.py
```

The current interface contains five tabs:

``` text
Dataset Management
Performance Comparison
Feature Importance
Misclassification
Decision Behaviour
```

The interaction logic is separated into callback modules:

``` text
dashboard/callback_modules/
```

The visualisation functions are separated into:

``` text
dashboard/visualisations/
```

This modular structure prevents the dashboard entry point from
containing the complete analytical implementation.

------------------------------------------------------------------------

# 16. Exported Results

Generated analytical outputs are stored under:

``` text
data/exports/
```

The main output groups are:

``` text
data/exports/
├── confusion/
├── errors/
├── local_explanations/
├── shap/
└── evaluation_summary.csv
```

These files are useful for:

-   inspecting analytical results outside the dashboard;
-   reproducing tables used during evaluation;
-   checking intermediate outputs;
-   supporting dissertation analysis.

If the scripts are rerun, existing generated files may be replaced or
regenerated depending on the script.

------------------------------------------------------------------------

# 17. Reproducing the Main Project Workflow

For a complete reproduction of the analytical workflow, use the
following sequence from the repository root:

``` bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated tests
pytest -q

# 3. Train the models
python scripts/train_models.py

# 4. Evaluate model performance
python scripts/evaluate_models.py

# 5. Generate global SHAP feature importance
python scripts/generate_feature_importance.py

# 6. Generate misclassification summaries
python scripts/generate_misclassification_analysis.py

# 7. Generate local explanations
python scripts/generate_local_explanations.py

# 8. Start the interactive dashboard
python dashboard/app.py
```

For normal use of the final application, running the dashboard is the
primary step. The standalone scripts are mainly provided for
reproducibility, analysis, and inspection of generated outputs.

------------------------------------------------------------------------

# 18. Adding or Replacing a Dataset

The current scripts are configured for the two project datasets.

The dataset paths and target columns are defined explicitly in the
scripts, for example:

``` python
DATASETS = {
    "HELOC": {
        "path": DATA_DIR / "heloc_dataset_v1.csv",
        "target": "RiskPerformance",
    },
    "BANK": {
        "path": DATA_DIR / "bank-additional-full.xlsx",
        "target": "y",
    },
}
```

To experiment with another dataset, the corresponding configuration must
be changed and the dataset must be compatible with the data-loading and
preprocessing pipeline.

At minimum, the dataset should have:

1.  a clearly identifiable target column;
2.  valid feature columns;
3.  supported file format;
4.  data types that can be processed by the preprocessing pipeline;
5.  a binary target if the existing classification workflow is retained
    unchanged.

A new dataset should be validated before model training.

------------------------------------------------------------------------

# 19. Modifying the Models

The current project uses two models:

``` text
Logistic Regression
Random Forest
```

To modify their definitions, edit:

``` text
src/models/logistic_regression.py
src/models/random_forest.py
```

The trainer expects the models to follow the common scikit-learn-style
interface used by the project.

If a new model is introduced, it should also be incorporated into the
relevant model/service and dashboard logic so that it can be evaluated
and explained consistently.

------------------------------------------------------------------------

# 20. Modifying the Visualisations

Dashboard visualisation functions are located in:

``` text
dashboard/visualisations/
```

The relevant files are:

``` text
comparison_plots.py
performance_plots.py
feature_importance.py
misclassification.py
decision_behaviour.py
```

The corresponding user-interface components are in:

``` text
dashboard/components/
```

and interaction logic is in:

``` text
dashboard/callback_modules/
```

A typical modification therefore involves three layers:

``` text
Analytical data
      ↓
Visualisation function
      ↓
Dashboard component / callback
```

Keeping these responsibilities separate makes visualisation changes
easier to test without changing the underlying analytical calculations.

------------------------------------------------------------------------

# 21. Troubleshooting

## `ModuleNotFoundError: No module named 'src'`

Make sure commands are being run from the repository root:

``` text
interactive-visual-analytics-xai/
```

For example:

``` bash
python scripts/evaluate_models.py
```

rather than changing into the `scripts/` directory first.

For the dashboard, use:

``` bash
python dashboard/app.py
```

from the repository root.

------------------------------------------------------------------------

## Missing dataset error

Check that the supplied files exist at:

``` text
data/raw/heloc_dataset_v1.csv
data/raw/bank-additional-full.xlsx
```

The filenames and locations used by the scripts must match the
configured paths.

------------------------------------------------------------------------

## Missing Python package

Activate the project's virtual environment and reinstall the
dependencies:

``` bash
pip install -r requirements.txt
```

If the environment has become inconsistent, creating a new virtual
environment is usually preferable to manually changing individual
package versions.

------------------------------------------------------------------------

## Dashboard does not open automatically

Check the terminal for the local Dash server address and open it
manually in a browser.

The normal development address is:

``` text
http://127.0.0.1:8050/
```

------------------------------------------------------------------------

## SHAP analysis is slow

SHAP calculations can be substantially more expensive than ordinary
metric calculations. The standalone global feature-importance script
therefore limits its explanation sample to up to 500 test rows.

The service-level XAI workflow uses a smaller explanation sample for
interactive analysis.

------------------------------------------------------------------------

# 22. Scope and Limitations

The implementation should be interpreted within the scope of the MSc
project.

The current evaluation is based on:

-   two datasets;
-   two binary classification models;
-   selected XAI tasks;
-   technical and analytical evaluation rather than formal user studies.

The system is therefore not intended to establish that the
visualisations are universally optimal or that the conclusions
generalise to every classification dataset or model family.

The project also does not implement the stakeholder-communication stage
of MAVIS.

The outputs are intended for academic analysis and should not be treated
as recommendations for real-world financial or other high-impact
decisions.

------------------------------------------------------------------------

# 23. Relationship to the MSc Project

The project specification defines the aim as designing, implementing,
and evaluating an interactive visual analytics system for exploring and
comparing classification models through selected XAI tasks.

The implementation directly supports the specified objectives:

  Project objective                  Main implementation
  ---------------------------------- --------------------------------------------
  Data selection and preprocessing   `src/data/`
  Logistic Regression                `src/models/logistic_regression.py`
  Random Forest                      `src/models/random_forest.py`
  Model evaluation                   `src/evaluation/`
  Feature importance                 `src/explainability/importance.py`
  Local explanations                 `src/explainability/local_explanations.py`
  Misclassification analysis         `src/evaluation/misclassification.py`
  Interactive visual analytics       `dashboard/`
  Analytical exports                 `data/exports/`
  Automated validation               `tests/`

The architecture is consistent with the project's
research-and-development approach and separates data preparation,
modelling, evaluation, explainability, services, and visual analytics.

------------------------------------------------------------------------

# 24. Academic Context

The project is aligned with the MAVIS research direction, particularly
the use of visual analytics to support model diagnosis, model
improvement, and model selection.

The supplied project materials identify XAI tasks such as:

-   accuracy;
-   bias;
-   class separation;
-   decision boundary;
-   example model predictions;
-   feature dependency;
-   feature importance;
-   feature interaction;
-   justification for decisions;
-   misclassification;
-   robustness;
-   uncertainty.

This implementation focuses on a selected subset that can be supported
within the project's scope.

The project specification also explicitly limits the work to Stages 1--3
of the MAVIS framework and excludes stakeholder communication (Stage 4).

------------------------------------------------------------------------

# 25. Development and Version Control

The repository is maintained using Git and GitHub.

The source code is organised so that changes to individual analytical
components can be made independently. The separation between services,
models, evaluation, explainability, and dashboard components also
supports incremental development and testing.

The `tests/` directory provides automated checks for core functionality.

------------------------------------------------------------------------

# 26. Responsible and Ethical Use

The project uses publicly available datasets for academic analysis and
does not require user participation.

The system should nevertheless be interpreted carefully because one of
the datasets concerns financial-risk-related classification. Model
explanations describe learned model behaviour; they do not establish
causation, fairness, or suitability for real-world decision-making.

The project was developed for academic purposes and should not be used
as a production decision-making system without additional validation,
governance, fairness assessment, security review, and domain-specific
evaluation.

------------------------------------------------------------------------

# 27. Licence

The repository contains an MIT licence.

See:

``` text
LICENSE
```

for the applicable licence terms.

------------------------------------------------------------------------

# 28. References and Further Reading

The project is informed by research and guidance on:

-   Explainable Artificial Intelligence;
-   visual analytics for machine learning;
-   SHAP-based model explanations;
-   interactive classifier comparison;
-   the MAVIS research project.

Key references used in the project include work by Ribeiro et al. on
LIME, Lundberg and Lee on SHAP, and the MAVIS project materials.

The dissertation and project specification provide the detailed academic
context and evaluation methodology for the implementation.

------------------------------------------------------------------------

# 29. Author

**Kaustubh Kaushal**

MSc Advanced Computer Science (Data Analytics)\
School of Computer Science\
University of Leeds

Project title:

**Interactive Visual Analytics for Explaining and Comparing
Classification Models**

Supervisor:

**Prof. Roy A. Ruddle**
