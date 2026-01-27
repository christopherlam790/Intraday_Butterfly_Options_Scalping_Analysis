
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import os
from dotenv import load_dotenv

"""
Identify sticky GK regimes from SQL table
@param table_name: str - name of the SQL table
@param visualize: bool - whether to visualize the results
@returns: pd.DataFrame
"""
def identify_sticky_gk_regimes(table_name: str, visualize_fig = False, save_fig = False) -> pd.DataFrame:
    
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
        
        def visualize_sticky_volatility_probability_vs_time_to_eod(df: pd.DataFrame, save_fig: bool = False) -> None:
        
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(df["time_till_eod"], df["sticky_prob"])
            ax.set_xlabel("Minutes to EOD")
            ax.set_ylabel("P(Sticky Vol Regime)")
            ax.set_title(f"{table_name} Sticky Volatility Probability vs Time to EOD")

            # Hour bands
            ax.axvline(360, color='black', linestyle='--', alpha=0.4) #10:00 am
            ax.axvline(300, color='black', linestyle='--', alpha=0.4) # 11:00 am
            ax.axvline(240, color='black', linestyle='--', alpha=0.4) # 12:00 pm
            ax.axvline(180, color='black', linestyle='--', alpha=0.4) # 1:00 pm
            ax.axvline(120, color='black', linestyle='--', alpha=0.4) # 2:00 pm
            ax.axvline(60, color='black', linestyle='--', alpha=0.4) # 3:00 pm
            
            # Intraday regimes
            ax.axvspan(390, 330, color='b', alpha=0.3, label='Market Open Power Hour (9:30–10:30pm)')
            ax.axvspan(330, 270, color='g', alpha=0.3, label='Early Morning (10:30–11:30pm)')
            ax.axvspan(270, 150, color='y', alpha=0.3, label='Midday (11:30–1:30pm)')
            ax.axvspan(150, 60,  color='r', alpha=0.3, label='Afternoon (1:30–3:00pm)')
            ax.axvspan(60,  0,   color='c', alpha=0.3, label='Close Power Hour (3:00–4:00pm)')
            
            # Stickiness thresholds
            ax.axhline(0.05, color='black', linestyle='--', label='5%', alpha=0.3)
            ax.axhline(0.10, color='black', linestyle='--', label='10%', alpha=0.3)
            ax.axhline(0.15, color='black', linestyle='--', label='15%', alpha=0.3)

            # Legend outside plot
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0
            )

            plt.tight_layout()

            
            if save_fig:
                plt.savefig(f"assets/charts/{table_name}_sticky_volatility_probability_vs_time_to_eod.png")
            plt.show()

            
            return None
            
        if visualize_fig:
            visualize_sticky_volatility_probability_vs_time_to_eod(df, save_fig=save_fig)
            
        return df

    except:
        raise Exception("Failed to pull data; Check df integrity & schema")



if __name__ == "__main__":


    df = identify_sticky_gk_regimes(table_name="spy_2025_5_minute_annual", visualize_fig=True, save_fig=True)

    print(df)