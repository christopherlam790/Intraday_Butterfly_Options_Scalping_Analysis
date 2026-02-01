"""
TAKE ta HEURISTICS -> INGESTION INTO POSGRESQL
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import os

load_dotenv()


import identify_sticky_ta_classification

"""
Prep df for SQL upload: reset index, lowercase columns
@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame
"""
def prep_df_for_sql(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df.index.name = "session"
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
        session TEXT PRIMARY KEY,

        rsi DOUBLE PRECISION,
        roc DOUBLE PRECISION,
        vwap DOUBLE PRECISION,
        cmf DOUBLE PRECISION,
        atr DOUBLE PRECISION,
        bb_width DOUBLE PRECISION,
        adx DOUBLE PRECISION,
        ema DOUBLE PRECISION,

        mean_heuristic_score DOUBLE PRECISION

    );
    """)

    columns = list(df.columns)
    col_names = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    insert_sql = f"""
    INSERT INTO {table_name} ({col_names})
    VALUES ({placeholders})
    ON CONFLICT (session) DO NOTHING;
    """

    records = df.itertuples(index=False, name=None)

    execute_batch(cur, insert_sql, records, page_size=1000)

    conn.commit()
    cur.close()
    conn.close()

    return None


"""
Get heuristic data for PostgreSQL
@param table_name: str - name of table from SQL to derrive heristics from
@param indicators: list - list of ta indicators to derrive heuristics from
@param sessions: dict - sessions to apply ta indicators on
@returns: pd.DataFrame - heuristic df, prepped for PGSql
"""
def get_heuristics_data(table_name:str, indicators: list = ["rsi", "roc", "vwap", "cmf", "atr", "bb_width", "adx", "ema"],
                               sessions: dict = {"overall": (390,0),
                                                 "open": (390, 330),
                                                 "post_open": (330,270),
                                                 "lunch": (270,150),
                                                 "afternoon": (150,60),
                                                 "close": (60,0)}) -> pd.DataFrame:
    
    df = identify_sticky_ta_classification.get_all_ta_info_by_session(table_name=table_name, indicators=indicators, sessions=sessions)
    
        
    df_sql = prep_df_for_sql(df=df)
    

    return df_sql



"""
==========================================
Testing Section
"""
if __name__ == "__main__":


    df_sql = get_heuristics_data(table_name="spy_2025_5_minute_annual_data", sessions={"overall": (390,0),
                                                 "open": (390, 330),
                                                 "post_open": (330,270),
                                                 "lunch": (270,150),
                                                 "afternoon": (150,60),
                                                 "close": (60,0),
                                                 
                                                 "09_30_to_10_00": (390, 360),
                                                 
                                                 "10_00_to_10_30": (360, 330), 
                                                 "10_30_to_11_00": (330, 300), 
                                                 
                                                 "11_00_to_11_30": (300, 270),
                                                 "11_30_to_12_00": (270, 240),
                                                 
                                                 "12_00_to_12_30": (240, 210),
                                                 "12_30_to_13_00": (210, 180),
                                                 
                                                 "13_00_to_13_30": (180, 150),
                                                 "13_30_to_14_00": (150, 120),
                                                 
                                                 "14_00_to_14_30": (120, 90),
                                                 "14_30_to_15_00": (90, 60), 
                                                
                                                 "15_00_to_15_30": (60, 30),
                                                 "15_30_to_16_00": (30, 0),
    
                                                                                  
                                                 })

    upload_data_as_postgressql(df_sql, table_name="spy_2025_5_minute_ta_heuristics")
    
    print("TESTING COMPLETE")
    
    


    
    

    