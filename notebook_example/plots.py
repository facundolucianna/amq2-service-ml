import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif


def plot_correlation_with_target(X, y, target_col="num", save_path=None):
    """
    Plots the correlation of each variable in the dataframe with the target column.

    Args:
    - X (pd.DataFrame): DataFrame containing the data, not including the target column.
    - y (pd.DataFrame): DataFrame containing the target column. Must have the same amount of observation of X.
    - target_col (str, optional): Name of the target column. If not specified, the name "num" will be used.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - fig: The figure to plot in the notebook
    """

    # Chequeamos que los dataframes estén alineados
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y are not aligned")
    
    df = pd.concat([X, y], axis=1)

    # Computa la correlación con respecto a la columna objetivo
    correlations = df.corr()[target_col].drop(target_col).sort_values()

    # Generamos la paleta de colores
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    # Configuramos a seaborn
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )

    # Creamos la gráfica de barra
    fig = plt.figure(figsize=(12, 8))
    bars = plt.barh(correlations.index, correlations.values, color=color_mapped)

    # Configuramos el título y nombre de ejes
    plt.title(f"Correlation with {target_col.title()}", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # Grabamos el gráfico si save_path es especificado
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # Prevemos que Matplotlib de mostrar el gráfico cada vez que llamamos a la función
    plt.close(fig)

    return fig


def plot_information_gain_with_target(X, y, target_col="num", save_path=None):
    """
    Plots the information gain of each variable in the dataframe with the target column.

    Args:
    - X (pd.DataFrame): DataFrame containing the data, not including the target column.
    - y (pd.DataFrame): DataFrame containing the target column. Must have the same amount of observation of X.
    - target_col (str, optional): Name of the target column. If not specified, the name "num" will be used.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - fig: The figure to plot in the notebook
    """

    # Chequeamos que los dataframes estén alineados
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y are not aligned")

    # Computa la ganancia de información con respecto a la columna objetivo
    importances = pd.Series(mutual_info_classif(X, y.to_numpy().ravel()), X.columns).sort_values()

    # Generamos la paleta de colores
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = importances.map(colors)

    # Configuramos a seaborn
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )

    # Creamos la gráfica de barra
    fig = plt.figure(figsize=(12, 8))
    bars = plt.barh(importances.index, importances, color=color_mapped)

    # Configuramos el título y nombre de ejes
    plt.title(f"Importance Gain with {target_col.title()}", fontsize=18)
    plt.xlabel("Importance Gain", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # Grabamos el gráfico si save_path es especificado
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # Prevemos que Matplotlib de mostrar el gráfico cada vez que llamamos a la función
    plt.close(fig)

    return fig
