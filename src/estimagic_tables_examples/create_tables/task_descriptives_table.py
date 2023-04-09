"""Tasks running the results formatting (tables, figures)."""

import estimagic as em
import numpy as np
import pandas as pd
import pytask
from pandas.api.types import is_numeric_dtype

from estimagic_tables_examples.config import BLD, IN_DATA

PARAMETRIZATION = {}
for return_type, file_ending in [("latex", "tex"), ("html", "html")]:
    depends_on = IN_DATA / "diabetes.csv"
    produces = BLD / "tables" / f"descriptive_stats.{file_ending}"
    PARAMETRIZATION[return_type] = {
        "depends_on": depends_on,
        "produces": produces,
        "return_type": return_type,
    }


for task_id, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=task_id)
    def task_descriptives_table(
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

        # making summary stats
        descriptive_stats = (
            df[["Age", "Sex", "BMI", "ABP"]]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .loc[["count", "mean", "std", "25%", "50%", "75%"]]
        ).T
        for v in ["Sex"]:
            descriptive_stats.loc[v, ["std", "25%", "50%", "75%"]] = np.nan

        descriptive_stats = descriptive_stats.rename(
            columns={
                "count": "N subj.",
                "mean": "Mean",
                "std": "Std. dev.",
                "5%": "$q_{0.05}$",
                "10%": "$q_{0.1}$",
                "25%": "$q_{0.25}$",
                "90%": "$q_{0.9}$",
                "50%": "$q_{0.5}$",
                "75%": "$q_{0.75}$",
                "95%": "$q_{0.95}$",
            },
        )

        # formatting
        # ToDo: Provide (part of) this function in estimagic?
        descriptive_stats = apply_custom_number_format(
            descriptive_stats,
            int_cols=["N subj."],
            number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        )
        if return_type == "html":
            out = em.render_html(
                descriptive_stats,
                {},
                append_notes=False,
                render_options={},
                show_footer=False,
                siunitx_warning=False,
                escape_special_characters=False,
            )
        elif return_type == "latex":
            out = em.render_latex(
                descriptive_stats,
                {},
                append_notes=False,
                render_options={},
                show_footer=False,
                siunitx_warning=False,
                escape_special_characters=False,
            )
            out = out.replace("Std. dev.", r"\makecell{Std. \\ Dev.}")
            out = out.replace("N subj.", r"\makecell{N\\ Subj.}")

        with open(produces, "w") as f:
            f.writelines(out)


def apply_number_format_to_series(series, number_format):
    """Apply string format to a pandas Series."""
    formatted = series.copy(deep=True).astype("float")
    for formatter in number_format[:-1]:
        formatted = formatted.apply(formatter.format).astype("float")
    formatted = formatted.astype("float").apply(number_format[-1].format)
    return formatted


def _add_multicolumn_left_format_to_column(column):
    """Align observation numbers at the center of model column."""
    out = column.replace(
        {i: f"\\multicolumn{{1}}{{r}}{{{i}}}" for i in column.unique()},
    )
    return out


def apply_custom_number_format(data, int_cols, number_format):
    """Apply custom number format to a pandas DataFrame.

    Take specific care of integer columns.

    """
    out = data.copy()
    for c in int_cols:
        out[c] = out[c].apply(lambda x: f"{x:.0f}")
        out[c] = _add_multicolumn_left_format_to_column(out[c])

    for c in out:
        if c not in int_cols and is_numeric_dtype(data[c]):
            out[c] = apply_number_format_to_series(out[c], number_format)

    out = out.replace({"nan": ""})
    return out
