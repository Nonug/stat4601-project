import json
import requests
import pandas as pd


def get_data(filename, station="STY"):
    datatype = "CLMTEMP"
    fields = ["year", "month", "day", "temperature", "completeness"]

    url = f"https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType={datatype}&station={station}&lang=en&rformat=json"
    temp_request = requests.get(url, timeout=10)

    if temp_request.status_code == 200:
        res = temp_request.json()
    else:
        return None

    df = pd.DataFrame(res["data"], columns=fields)
    if filename:
        df.to_csv(f"./data/{filename}", index=False)
    return df
