"""Tasks running the results formatting (tables, figures)."""

import estimagic as em
import pandas as pd
import pytask
import statsmodels.formula.api as sm

from estimagic_tables_examples.config import BLD, IN_DATA

PARAMETRIZATION = {}
for return_type, file_ending in [("latex", "tex"), ("html", "html")]:
    depends_on = IN_DATA / "diabetes.csv"
    produces = BLD / "tables" / f"statsmodels_advanced_two_step.{file_ending}"
    PARAMETRIZATION[return_type] = {
        "depends_on": depends_on,
        "produces": produces,
        "return_type": return_type,
    }


for task_id, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=task_id)
    def task_two_step_table(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        return_type=kwargs["return_type"],
    ):
        """Results table using two step procedure and the following advanced options:

        - Combine statsmodels and other results
        - Group columns (resulting in a multiindex)
        - Specify column_format: two significant digits
        - Add midrule to table after rendering to latex

        """
        df = pd.read_csv(depends_on, index_col=0)
        mod1 = sm.ols("target ~ Age + Sex", data=df).fit()
        mod2 = sm.ols("target ~ Age + Sex + BMI + ABP", data=df).fit()
        models = [mod1, mod2]

        params = pd.DataFrame(
            {
                "value": [142.123, 51.456, -33.789],
                "standard_error": [3.1415, 2.71828, 1.6180],
                "p_value": [1e-8] * 3,
            },
            index=["Intercept", "Age", "Sex"],
        )
        mod3 = {"params": params, "name": "target", "info": {"n_obs": 4425}}
        models.append(mod3)

        render_inputs = em.estimation_table(
            models,
            return_type="render_inputs",
            custom_col_groups=["Statsmodels", "Statsmodels", "Other"],
            number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        )

        # Remove rows from body.
        render_inputs["body"] = pd.concat(
            [render_inputs["body"].iloc[:6], render_inputs["body"].iloc[-2:]],
        )

        # Add a row to the footer.
        render_inputs["footer"].loc[("Control for BMI",)] = ["Yes"] + ["No"] * 2

        if return_type == "html":
            out = em.render_html(render_inputs["body"], render_inputs["footer"])
        elif return_type == "latex":
            out = em.render_latex(
                render_inputs["body"],
                render_inputs["footer"],
                siunitx_warning=False,
                custom_notes=[
                    "Two significant digits. ",
                    "Note that this is not applied to integer cells.",
                    "Midrule before ABP added after rendering to latex.",
                ],
            )
            out = (add_midrules_to_latex(out, [14]),)

        with open(produces, "w") as f:
            f.writelines(out)


def add_midrules_to_latex(out, rows, midrule_text=r"\midrule"):
    # Add midrules
    latex_list = out.splitlines()
    for row in rows:
        latex_list.insert(row, midrule_text)

    # join split lines to get the modified latex output string
    out = "\n".join(latex_list)
    return out
