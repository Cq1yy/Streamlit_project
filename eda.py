import pandas as pd
from pickle import dump, load
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
def split_data(df: pd.DataFrame):
    y = df['TARGET']
    X = df[["AGREEMENT_RK", "AGE", "GEN_TITLE", "SOCSTATUS_WORK_FL",
            "SOCSTATUS_PENS_FL", "GENDER", "CHILD_TOTAL", "DEPENDANTS",
            "PERSONAL_INCOME","LOAN_NUM_TOTAL","LOAN_NUM_CLOSED"]]

    return X, y


def open_data(path="data/final_df.csv"):
    df = pd.read_csv(path)

    return df


def make_correlation(df):
    cols = df.columns[df.dtypes != 'object']
    corr = df[cols[1::]].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.heatmap(corr, cmap="Blues", annot=True)
    return fig


def draw_distribution_graphs(df, x, xmin, xmax, bins='auto'):
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.histplot(data=df, x=x, ax=ax, bins=bins,kde=True)
    ax.set_xlim(xmin, xmax)
    return fig

def plot_graph(df, x, xmin, xmax, bins='auto'):
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.lineplot(data=df, x=x, y="TARGET")
    ax.set_xlim(xmin, xmax)
    return fig

def target_quantity_column(df, income_column, new_colomn, group_width=5000):
    df[new_colomn] = (df[income_column] // group_width) * group_width
    df1 = df.groupby(new_colomn)
    count_df = pd.DataFrame(df1["TARGET"].sum())
    return count_df


def target_addiction(df,x,y, bins='auto'):

    df1 = df.groupby(x)
    count_df = df1[y].sum()
    df["count_of_target"] = count_df

    fig, ax = plt.subplots(figsize=(10, 6))
    sb.regplot(data=df, x=x,y="count_of_target", ax=ax)
    return fig

def get_numeric_features(df):
    numeric_df = df.select_dtypes(include=['number'])

    # Вычисление числовых характеристик
    numeric_summary = numeric_df.describe()

    # Добавление имени столбца в качестве индекса
    numeric_summary.index.names = ['feature']

    return pd.DataFrame(numeric_summary)


def get_categorical_features(df: pd.DataFrame, col) -> pd.DataFrame:

    result = df[col].value_counts(normalize=True).to_frame(name='Частота')

    return result

#if __name__ == "__main__":
