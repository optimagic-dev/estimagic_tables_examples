"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask
import estimagic as em

from estimagic_tables_examples.config import BLD, IN_DATA
import statsmodels.formula.api as sm

PARAMETRIZATION = {}
for return_type, file_ending in [("latex", "tex"), ("html", "html")]:
    depends_on = IN_DATA / "diabetes.csv"
    produces = BLD / "tables" / f"statsmodels_basic.{file_ending}"
    PARAMETRIZATION[return_type] = {
        "depends_on": depends_on,
        "produces": produces,
        "return_type": return_type,
    }


for task_id, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=task_id)
    def task_simple_statsmodels_table_latex(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        return_type=kwargs["return_type"],
    ):
        """Simple statsmodels table. The example is taken from the documentation."""
        df = pd.read_csv(depends_on, index_col=0)
        mod1 = sm.ols("target ~ Age + Sex", data=df).fit()
        mod2 = sm.ols("target ~ Age + Sex + BMI + ABP", data=df).fit()
        models = [mod1, mod2]
        table = em.estimation_table(models, return_type=return_type)
        with open(produces, "w") as f:
            f.writelines(table)
