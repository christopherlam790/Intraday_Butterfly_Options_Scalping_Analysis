from massive import RESTClient


from dotenv import load_dotenv
import os
import json

load_dotenv()

API_KEY = os.getenv("MASSIVE_API_KEY")



def get_options_data(API_KEY, ticker, start_time, end_time):
    
    try:
    
        client = RESTClient(api_key=API_KEY)

        # List Aggregates (Bars)
        aggs = []
        for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_=start_time, to=end_time, limit=50000):
            aggs.append(a)

        print(aggs)
        
        return

    except Exception as ex:
        return ex


"""
===============
TESTING AREA
"""
if __name__ == "__main__":
    
    response = get_options_data(API_KEY=API_KEY, ticker="AAPL", start_time="2020-01-01", end_time="2021-01-01")
    
    print(response)
    
