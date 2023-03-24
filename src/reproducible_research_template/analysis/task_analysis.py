"""Tasks running the core analyses."""

import pandas as pd
import pytask

from reproducible_research_template.analysis.model import fit_logit_model, load_model
from reproducible_research_template.analysis.predict import predict_prob_by_age
from reproducible_research_template.config import BLD, GROUPS, SRC
from reproducible_research_template.utilities import read_yaml


@pytask.mark.depends_on(
    {
        "scripts": ["model.py", "predict.py"],
        "data": BLD / "python" / "data" / "data_clean.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "models" / "model.pickle")
def task_fit_model_python(depends_on, produces):
    """Fit a logistic regression model (Python version)."""
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    model = fit_logit_model(data, data_info, model_type="linear")
    model.save(produces)


for group in GROUPS:

    kwargs = {
        "group": group,
        "produces": BLD / "python" / "predictions" / f"{group}.csv",
    }

    @pytask.mark.depends_on(
        {
            "data": BLD / "python" / "data" / "data_clean.csv",
            "model": BLD / "python" / "models" / "model.pickle",
        },
    )
    @pytask.mark.task(id=group, kwargs=kwargs)
    def task_predict_python(depends_on, group, produces):
        """Predict based on the model estimates (Python version)."""
        model = load_model(depends_on["model"])
        data = pd.read_csv(depends_on["data"])
        predicted_prob = predict_prob_by_age(data, model, group)
        predicted_prob.to_csv(produces, index=False)


@pytask.mark.r(script=SRC / "analysis" / "model.r", serializer="yaml")
@pytask.mark.depends_on(
    {
        "scripts": ["predict.r"],
        "data": BLD / "r" / "data" / "data_clean.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "r" / "models" / "model.rds")
def task_fit_model_r():
    """Fit a logistic regression model (R version)."""


for group in GROUPS:

    kwargs = {
        "group": group,
        "produces": BLD / "r" / "predictions" / f"{group}.csv",
    }

    @pytask.mark.depends_on(
        {
            "data": BLD / "r" / "data" / "data_clean.csv",
            "model": BLD / "r" / "models" / "model.rds",
        },
    )
    @pytask.mark.task(id=group, kwargs=kwargs)
    @pytask.mark.r(script=SRC / "analysis" / "predict.r", serializer="yaml")
    def task_predict_r():
        """Predict based on the model estimates (R version)."""
