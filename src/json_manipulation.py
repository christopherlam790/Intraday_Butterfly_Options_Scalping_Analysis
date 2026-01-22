import os
import json
import pandas as pd
from typing import List, Dict, Any


"""
Gets the results from a single JSON file and returns as a DataFrame.

@param: json_path: str - path to the JSON file
@retruns: pd.DataFrame
@raises: FileNotFoundError, json.JSONDecodeError
"""
def get_json_results(json_path) -> pd.DataFrame:
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        file.close()
        
        return pd.DataFrame(data["results"])

    except FileNotFoundError:
        print("Error: The file was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    

"""
Aggregate multiple JSON results into a single DataFrame; Calls on get_json_results.

@param: json_paths: List[str] - list of paths to JSON files
@returns: pd.DataFrame
@raises: ValueError - if no valid DataFrames are found
"""
def aggregate_json_results(json_paths: List[str]) -> pd.DataFrame:
    
        
    """
    Gets the results from a single JSON file and returns as a DataFrame.

    @param: json_path: str - path to the JSON file
    @retruns: pd.DataFrame
    @raises: FileNotFoundError, json.JSONDecodeError
    """
    def get_json_results(json_path) -> pd.DataFrame:
        try:
            with open(json_path, 'r') as file:
                data = json.load(file)
            
            file.close()
            
            return pd.DataFrame(data["results"])

        except FileNotFoundError:
            print("Error: The file was not found.")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    
    
    return pd.concat([get_json_results(path) for path in json_paths], ignore_index=True)
    
    
"""
Save df as a JSON file in specific directory. Makes directory if it does not exist.

@param: directory: str - directory to save the file
@param: filename: str - name of the file
@param: df: pd.DataFrame - DataFrame to save
@returns: str - path to the saved file
"""
def save_json_file(directory, filename, df):
    
    filepath = os.path.join(directory, filename)

    # Ensure the directory exists (create it if not)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the DataFrame to a JSON file
    df.to_json(filepath, orient='records', indent=4) 
    
    return filepath 
    
    
"""
TESTING AREA
"""
if __name__ == "__main__":
    df_2024 = aggregate_json_results(["assets/raw/2024_5_minute/02_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/03_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/04_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/05_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/06_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/07_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/08_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/09_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/10_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/11_2024_5_minute_SPY.json",
                            "assets/raw/2024_5_minute/12_2024_5_minute_SPY.json",
                          ])
    
    
    print(df_2024)

    df_2025 = aggregate_json_results(["assets/raw/2025_5_minute/01_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/02_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/03_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/04_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/05_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/06_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/07_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/08_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/09_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/10_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/11_2025_5_minute_SPY.json",
                            "assets/raw/2025_5_minute/12_2025_5_minute_SPY.json",
                          ])

    
    print(df_2025)
    