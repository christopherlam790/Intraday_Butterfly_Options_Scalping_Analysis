import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import download_heuristic_data

"""
Visualize session heuristics as rankings- Locates "best" session for butterflies

@param df: pd.DataFrame - Heuristic df
@param plot_name:str - Name of plot
@param ascending:bool - Order of ranking; High-low by default
@param save_fig:bool - To save figure or not

@returns: None
"""
def visualize_ranked_bar_chart(df: pd.DataFrame, plot_name:str, ascending:bool = False, save_fig: bool = False) -> None:

    plot_df = df.copy()
    plot_df = plot_df.sort_values("mean_heuristic_score", ascending=ascending)

    # classify session type
    def session_type(s):
        return "intraday" if "_to_" in s else "macro"

    types = plot_df.index.map(session_type)

    # color map
    colors = np.where(types == "macro", "tab:red", "tab:blue")

    plt.figure(figsize=(12, 5))
    plt.bar(plot_df.index, plot_df["mean_heuristic_score"], color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Heuristic Score")
    plt.title(plot_name)

    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f"assets/charts/{plot_name}.png")
            
    plt.show()




"""
Visualize session heuristics as heatmap- Identifies mean heuristic score on more granular level

@param df: pd.DataFrame - Heuristic df
@param plot_name:str - Name of plot
@param indicator_cols:list - List of heuristic indicators to use in heatmap
@param save_fig:bool - To save figure or not

@returns: None
@raise: Exception - Error if given indicator not in df
"""
def visualize_heat_map(df:pd.DataFrame, plot_name:str, indicator_cols:list = [
    "rsi", "roc", "vwap", "cmf", "atr", "bb_width", "adx", "ema"
], save_fig:bool = False) -> None:
    
    
    
    for ta in indicator_cols:
        if ta not in df.columns:
            raise Exception(f"Indicator {ta} not found in df")
    
    plot_df = df.copy()


    heat_df = plot_df[indicator_cols].to_numpy()

    plt.figure(figsize=(10, 6))
    im = plt.imshow(heat_df, aspect="auto")

    plt.colorbar(im, fraction=0.03, pad=0.02)

    plt.yticks(
        ticks=np.arange(len(plot_df.index)),
        labels=plot_df.index
    )
    plt.xticks(
        ticks=np.arange(len(indicator_cols)),
        labels=indicator_cols,
        rotation=45,
        ha="right"
    )

    plt.title(plot_name)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f"assets/charts/{plot_name}.png")
    
    plt.show()



"""
Visualize session heuristics as time stability - Shows any noise in intraday timing

@param df: pd.DataFrame - Heuristic df
@param plot_name:str - Name of plot
@param save_fig:bool - To save figure or not

@returns: None
"""
def visualize_time_stability_plot(df:pd.DataFrame, plot_name:str, save_fig:bool = False):
    intraday_df = df[df.index.str.contains("_to_")].copy()

    # preserve chronological order
    intraday_df["start_min"] = intraday_df.index.str.slice(0, 5)
    intraday_df = intraday_df.sort_values("start_min")

    plt.figure(figsize=(10, 4))
    plt.plot(
        intraday_df.index,
        intraday_df["mean_heuristic_score"],
        marker="o"
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Heuristic Score")
    plt.title(plot_name)

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"assets/charts/{plot_name}.png")
    
    plt.show()



if __name__ == "__main__":
    
    df = download_heuristic_data.get_raw_df_from_sql("spy_2025_5_minute_ta_heuristics")
    
    visualize_time_stability_plot(df=df, plot_name="spy_2025_5_minute_ta_heuristic_time_stability", save_fig=True)
    
    pass