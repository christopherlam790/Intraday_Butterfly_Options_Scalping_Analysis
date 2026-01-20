from dotenv import load_dotenv
import os
import json
import pandas as pd

load_dotenv()

API_KEY = os.getenv("MASSIVE_API_KEY")


def load_json(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        

        file.close()
        
        return data["results"]

    except FileNotFoundError:
        print("Error: The file was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    
    
if __name__ == "__main__":
    data = load_json(json_path="assets/2025_5_minute/12_2025_5_minute_SPY.json")
    
    df = pd.DataFrame(data)
    
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["datetime_est"] = df["datetime"].dt.tz_convert("America/New_York")
    
    df = df.set_index("datetime_est")
    df = df.between_time("09:30", "16:00")

    
    print(df)