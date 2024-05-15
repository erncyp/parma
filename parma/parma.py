import pandas as pd
from itertools import combinations


def _create_column_combinations(columns, limit=2):
    """
    Returns all possible column combinations up to limit.
    """
    combintaion_list = []
    for i in range(1, limit + 1):
        combintaion_list += list(combinations(columns,i))
    return combintaion_list


def rule_miner(data, feature_cols, y_col, limit=2):
    """
    uses pandas groupby for association rule mining
    """
    # TODO: check that y_col exists and its binary
    df = data.copy()

    for col in feature_cols:
        df[col] = "(" + str(col) + " = " + df[col].astype(str) + ")"

    num_rows = df.shape[0]
    support_y = df[y_col].mean()
    results = []

    for combination in _create_column_combinations(feature_cols, limit):
        _df = df.groupby(list(combination))[y_col].agg(["count", "mean"]).reset_index()
        _df["itemset"] = _df.iloc[:, :-2].sum(axis=1)
        _df = _df.iloc[:, -3:]
        results.append(_df)

    df_total = pd.concat(results, axis=0).reset_index(drop=True)

    df_total = df_total.rename(columns={"mean": "confidence(X,Y)", "count": "count(X)"})
    df_total["support(X)"] = df_total["count(X)"] / num_rows
    df_total["lift(X,Y)"] = df_total["confidence(X,Y)"] / support_y
    df_total = df_total[['itemset', 'count(X)', 'support(X)', 'confidence(X,Y)', 'lift(X,Y)']]
    return df_total
