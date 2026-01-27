
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import os
from dotenv import load_dotenv

import download_raw_data



def analyze_ta_indicator_combos(df: pd.DataFrame, indicator_columns: list[str], time_zone: int = 0) -> None:
    
    
    
    return



if __name__ == "__main__":
    
    df = download_raw_data.get_raw_df_from_sql(
        table_name="spy_2025_5_minute_annual",
        fields=["rsi_3_isolated"],)
    
    print(df)
