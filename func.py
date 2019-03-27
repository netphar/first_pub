import pandas as pd


def locing(df, a, b, c):
    """

    :param a: drug_row_id
    :param b: drug_col_id
    :param c: cell_line_id
    :param df: input dataframe
    :return: output dataframe
    """
    var: pd.DataFrame = df.loc[(df.drug_row_id == a) & (df.drug_col_id == b) & (df.cell_line_id == c)]
    assert isinstance(var, pd.DataFrame)
    return var
