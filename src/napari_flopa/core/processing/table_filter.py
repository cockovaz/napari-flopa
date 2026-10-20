import re

import numpy as np
import pandas as pd

OPS = (
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "contains",
    "not contains",
    "is empty",
    "not empty",
)

#: Operators that take no value.
UNARY_OPS = ("is empty", "not empty")

_PLAIN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_column(col: str) -> str:
    """Column reference usable in a pandas query expression."""
    return col if _PLAIN_NAME.match(col) else f"`{col}`"


def build_clause(
    col: str, op: str, value: str = "", numeric: bool = False
) -> str:
    """One query clause, e.g. ``tau_phi_ns > 2`` or ``grp == "A"``.

    *value* is the raw text the user typed. On numeric columns it must parse
    as a number for comparisons; text values are quoted. ``contains`` is a
    plain substring match (not a regex) that treats missing values as no match.
    """
    ref = quote_column(col)
    if op == "is empty":
        return f"{ref}.isna()" if numeric else f'({ref}.isna() or {ref} == "")'
    if op == "not empty":
        return (
            f"{ref}.notna()" if numeric else f'({ref}.notna() and {ref} != "")'
        )
    if op in ("contains", "not contains"):
        expr = f"{ref}.astype('str').str.contains({value!r}, regex=False, na=False)"
        return f"not {expr}" if op == "not contains" else expr
    if op not in OPS:
        raise ValueError(f"Unknown operator: {op}")
    if numeric:
        try:
            num = float(value)
        except ValueError as e:
            raise ValueError(
                f"'{value}' is not a number for column {col}"
            ) from e
        lit = repr(int(num)) if num.is_integer() else repr(num)
    else:
        lit = repr(value)
    return f"{ref} {op} {lit}"


def evaluate(df: pd.DataFrame, expr: str) -> np.ndarray:
    """Boolean row mask for a query *expr*; raises ValueError if it's invalid."""
    try:
        result = df.eval(expr, engine="python")
    except Exception as e:  # pandas raises many types for a bad expression
        raise ValueError(str(e) or type(e).__name__) from e
    mask = np.asarray(result)
    if mask.shape != (len(df),) or mask.dtype != bool:
        raise ValueError("Expression must give True/False per row")
    return mask
