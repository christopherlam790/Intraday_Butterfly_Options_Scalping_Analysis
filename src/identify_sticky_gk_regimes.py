
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import os
from dotenv import load_dotenv

"""
Get sticky GK regime data from SQL table
@param table_name: str - name of the SQL table
@returns: pd.DataFrame
"""
def get_sticky_regime_data(table_name: str) -> pd.DataFrame:
    
    # Connect to DB
    try:
        load_dotenv()

        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            dbname=os.getenv("PG_DB"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            port=os.getenv("PG_PORT")
        )
    except:
        raise Exception("No connection; Check creds")

    # Run query and identify sticky GK regimes
    try:
        
        query = f"""
            SELECT
                time_till_eod,
                SUM(CASE WHEN is_sticky_vol_regime THEN 1 ELSE 0 END)::float
                    / COUNT(*) AS sticky_prob,
                COUNT(*) AS n
            FROM {table_name}
            GROUP BY time_till_eod
            ORDER BY time_till_eod;
            """

        df = pd.read_sql(query, conn)        
        
        conn.close()
        

        return df

    except:
        raise Exception("Failed to pull data; Check df integrity & schema")


"""
Visualize sticky volatility probability vs time to EOD
@param df: pd.DataFrame - DataFrame with 'time_till_eod' and 'sticky_prob' columns
@param plot_name: str - name for the plot
@param show_hour_bands: bool - whether to show hour bands
@param show_intraday_regimes: bool - whether to show intraday regimes
@param show_lift_thresholds: bool - whether to show lift thresholds
@param save_fig: bool - whether to save the figure
@returns: None
"""
def visualize_sticky_volatility_probability_vs_time_to_eod(df: pd.DataFrame, plot_name: str, 
                                                           show_hour_bands: bool = True, show_intraday_regimes: bool = True, 
                                                           show_lift_thresholds:bool = True, 
                                                           save_fig: bool = False) -> None:
        
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df["time_till_eod"], df["sticky_prob"])
        ax.set_xlabel("Minutes to EOD")
        ax.set_ylabel("P(Sticky Vol Regime)")
        ax.set_title(f"{plot_name} Sticky Volatility Probability vs Time to EOD")

        # Hour bands
        if show_hour_bands:
            ax.axvline(360, color='black', linestyle=':', alpha=0.25) #10:00 am
            ax.axvline(300, color='black', linestyle=':', alpha=0.25) # 11:00 am
            ax.axvline(240, color='black', linestyle=':', alpha=0.25) # 12:00 pm
            ax.axvline(180, color='black', linestyle=':', alpha=0.25) # 1:00 pm
            ax.axvline(120, color='black', linestyle=':', alpha=0.25) # 2:00 pm
            ax.axvline(60, color='black', linestyle=':', alpha=0.25) # 3:00 pm
        
        # Intraday regimes
        if show_intraday_regimes:
            ax.axvspan(390, 330, color='b', alpha=0.3, label='Market Open Power Hour (9:30–10:30pm)')
            ax.axvspan(330, 270, color='g', alpha=0.3, label='Early Morning (10:30–11:30pm)')
            ax.axvspan(270, 150, color='y', alpha=0.3, label='Midday (11:30–1:30pm)')
            ax.axvspan(150, 60,  color='r', alpha=0.3, label='Afternoon (1:30–3:00pm)')
            ax.axvspan(60,  0,   color='c', alpha=0.3, label='Close Power Hour (3:00–4:00pm)')
        
        # Lift thresholds
        
        if show_lift_thresholds:
            ax.axhline(df['sticky_prob'].mean(), color='red', linestyle='--', label='Average Sticky Prob', alpha=0.3)
            ax.axhline(df['sticky_prob'].mean() + (0.5 * df['sticky_prob'].std()), color='blue', linestyle='--', label='Average Sticky Prob + 0.5 Std Dev', alpha=0.3)
            ax.axhline(df['sticky_prob'].mean() + df['sticky_prob'].std(), color='green', linestyle='--', label='Average Sticky Prob + Std Dev', alpha=0.3)
            
 
        

        # Legend outside plot
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0
        )

        plt.tight_layout()

        
        if save_fig:
            plt.savefig(f"assets/charts/{plot_name}_sticky_volatility_probability_vs_time_to_eod.png")
            

        plt.show()

        
        return None





def identify_sticky_gk_regimes(table_name: str, save_fig: bool = False) -> None:
    df = get_sticky_regime_data(table_name=table_name)
    
    visualize_sticky_volatility_probability_vs_time_to_eod(df, plot_name=table_name, save_fig=save_fig)

    pass

if __name__ == "__main__":


    df = identify_sticky_gk_regimes(table_name="spy_2024_5_minute_annual", save_fig=True)

    print(df)