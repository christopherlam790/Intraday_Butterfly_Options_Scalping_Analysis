"""
PGSQL RAW DATA -> PYTHON DF
"""



import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv


"""
Pull raw data from table name, if it exists, from PGSQL db with fields specified
@param table_name: str - name of the SQL table
@param fields: list - list of fields to pull from table; if empty, pulls all fields
@returns: pd.DataFrame
"""
def get_raw_df_from_sql(table_name: str, fields: list = []) -> pd.DataFrame:
    
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

    try:
        
        
        query = f"""
        SELECT date, {', '.join(fields) if fields else '*'}
        FROM {table_name}
        ORDER BY date;
        """

        df = pd.read_sql(query, conn)
        conn.close()

        df = df.set_index("date")

        return df
    except:
        raise Exception("Failed to pull data; Check df integrity & schema")


"""
=======
TEST AREA
"""
if __name__ == "__main__":
    df = get_raw_df_from_sql("spy_2024_5_minute_annual")
    
    print(df)