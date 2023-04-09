"""Tasks running the results formatting (tables, figures)."""

import estimagic as em
import pandas as pd
import pytask
import statsmodels.formula.api as sm

from estimagic_tables_examples.config import BLD, IN_DATA

PARAMETRIZATION = {}
for return_type, file_ending in [("latex", "tex"), ("html", "html")]:
    depends_on = IN_DATA / "diabetes.csv"
    produces = BLD / "tables" / f"statsmodels_simple_two_step.{file_ending}"
    PARAMETRIZATION[return_type] = {
        "depends_on": depends_on,
        "produces": produces,
        "return_type": return_type,
    }


for task_id, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=task_id)
    def task_two_step_statsmodels_table_latex(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        return_type=kwargs["return_type"],
    ):
        """Simple Statsmodels table using two step procedure."""
        df = pd.read_csv(depends_on, index_col=0)
        mod1 = sm.ols("target ~ Age + Sex", data=df).fit()
        mod2 = sm.ols("target ~ Age + Sex + BMI + ABP", data=df).fit()
        models = [mod1, mod2]
        render_inputs = em.estimation_table(models, return_type="render_inputs")

        # Remove rows from footer.
        render_inputs["footer"] = render_inputs["footer"].loc[["R$^2$", "Observations"]]

        # Remove rows from body.
        render_inputs["body"] = pd.concat(
            [render_inputs["body"].iloc[:6], render_inputs["body"].iloc[-2:]],
        )

        # Add a row to the footer.
        render_inputs["footer"].loc[("Control for BMI",)] = ["Yes"] + ["No"]

        if return_type == "html":
            out = em.render_html(render_inputs["body"], render_inputs["footer"])
        elif return_type == "latex":
            out = em.render_latex(
                render_inputs["body"],
                render_inputs["footer"],
                siunitx_warning=False,
            )

        with open(produces, "w") as f:
            f.writelines(out)
