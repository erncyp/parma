import seaborn as sns
import pandas as pd
from parma import parma

def example():
    df = sns.load_dataset("titanic")
    df = df.drop(columns=["alive"])  # is duplicate of survived

    # bin age and fare to 10 equal sized bins
    for x in ["age", "fare"]:
        df[x] = pd.qcut(df[x], 10).astype(str)

    # define our y column (this must be binary). This is the feature we are hoping to gain insight about
    y_col = "survived"
    feature_cols = [x for x in df.columns if x != y_col]
    rules_df = parma.rule_miner(df, feature_cols, y_col, limit=3)
    print(rules_df)


if __name__ == "__main__":
    example()