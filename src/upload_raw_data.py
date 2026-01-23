"""
TAKE PREPROCESSED RAW DATA -> INGESTION INTO POSGRESQL
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import os

load_dotenv()


import json_preprocessing

"""
Prep df for SQL upload: reset index, lowercase columns
@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame
"""
def prep_df_for_sql(df: pd.DataFrame) -> pd.DataFrame:

    df.index.name = "date"
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    return df

"""
Upload DataFrame into PostgreSQL
@param df: pd.DataFrame - DataFrame to upload
@param table_name: str - name of the SQL table
@returns: None
"""
def upload_data_as_postgressql(df: pd.DataFrame, table_name: str) -> None:

    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        port=os.getenv("PG_PORT")
    )

    cur = conn.cursor()

    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        date TIMESTAMPTZ NOT NULL PRIMARY KEY,
        volume BIGINT,
        open DOUBLE PRECISION,
        close DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        number_of_trades BIGINT,

        rsi_3_isolated DOUBLE PRECISION,
        rsi_6_isolated DOUBLE PRECISION,
        rsi_12_isolated DOUBLE PRECISION,

        roc_3_isolated DOUBLE PRECISION,
        roc_6_isolated DOUBLE PRECISION,
        roc_12_isolated DOUBLE PRECISION,

        cmf_3_isolated DOUBLE PRECISION,
        cmf_6_isolated DOUBLE PRECISION,
        cmf_12_isolated DOUBLE PRECISION,
        cmf_6_rolling DOUBLE PRECISION,
        cmf_12_rolling DOUBLE PRECISION,

        vwap_3_isolated DOUBLE PRECISION,
        vwap_6_isolated DOUBLE PRECISION,
        vwap_12_isolated DOUBLE PRECISION,

        atr_6_rolling DOUBLE PRECISION,
        atr_12_rolling DOUBLE PRECISION,

        bb_width_3_isolated DOUBLE PRECISION,
        bb_width_6_isolated DOUBLE PRECISION,
        bb_width_12_isolated DOUBLE PRECISION,

        adx_6_rolling DOUBLE PRECISION,
        adx_12_rolling DOUBLE PRECISION,

        ema_3_isolated DOUBLE PRECISION,
        ema_6_isolated DOUBLE PRECISION,
        ema_6_rolling DOUBLE PRECISION,
        ema_12_rolling DOUBLE PRECISION,

        time_till_eod INTEGER
    );
    """)

    columns = list(df.columns)
    col_names = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    insert_sql = f"""
    INSERT INTO {table_name} ({col_names})
    VALUES ({placeholders})
    ON CONFLICT (date) DO NOTHING;
    """

    records = df.itertuples(index=False, name=None)

    execute_batch(cur, insert_sql, records, page_size=1000)

    conn.commit()
    cur.close()
    conn.close()

    return None


"""
Upload raw data into PostgreSQL
@param path: str - path to raw JSON data
@param cols_to_drop: list - list of columns to drop during preprocessing
@param rth: bool - whether to filter for regular trading hours
@returns: pd.DataFrame - preprocessed DataFrame ready for SQL upload
"""
def upload_raw_data(path: str, cols_to_drop: list=[], rth: bool=True) -> pd.DataFrame:
    
    df = json_preprocessing.get_json_results(path)
    
    df = json_preprocessing.preprocess_data(df, cols_to_drop=cols_to_drop, rth=rth)
        
    df_sql = prep_df_for_sql(df=df)
    

    return df_sql



"""
==========================================
Testing Section
"""
if __name__ == "__main__":


    df_sql = upload_raw_data(path="assets/raw/2024_5_minute/annual_2024_5_minute_SPY.json",
                    cols_to_drop=["volume_weighted_average_price", "UTC_timestamp", "datetime"],
                    rth=True)

    upload_data_as_postgressql(df_sql, table_name="spy_2024_5_minute_annual")
    
    print("TESTING COMPLETE")
    
    


    
    

    