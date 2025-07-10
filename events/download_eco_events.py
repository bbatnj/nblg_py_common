from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
import pandas as pd

TIMEZONE = 'Asia/Singapore'
KEYS = ["Time", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous", "Alert"]
INVESTING_URL = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "Priority": "u=1, i",
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "X-Requested-With": "XMLHttpRequest"
}

# Modify the request body to accept date ranges
def get_request_body(date_from, date_to):
    return {
        #"country": ["25", "32", "6", "37", "72", "22", "17", "39", "14", "10", "35", "43", "56", "36", "110", "11", "26", "12", "4", "5"],
        "country[]":["72","17","35","4","5"],
        "importance[]":["3"],
        "category":[
            "_employment",
            "_economicActivity",
            "_inflation",
            "_credit",
            "_centralBanks",
            "_confidenceIndex",
            "_balance"
        ],
        "dateFrom": date_from.strftime('%Y-%m-%d'),
        "dateTo": date_to.strftime('%Y-%m-%d'),
        #"timeZone": "28", # UTC
        "timeZone": "113", # Singapore
        "timeFilter": "timeOnly",
        "currentTab": "custom",
        "limit_from": "0",
    }
def get_and_parse_one_day_webpage(events, date):
    req_data = get_request_body(date, date)
    print(f'Requesting data {date}...')
    response = requests.post(INVESTING_URL, headers=HEADERS, data=req_data)
    if response.status_code == 200:
        response_data = response.json()
        if 'data' in response_data:  # data is a dict, the real data is html str.
            html_content = response_data['data']
            soup = BeautifulSoup(html_content, "html.parser")
            rows = soup.find_all("tr")
            table_data = []
            date_info = None
            for row in rows:
                columns = row.find_all("td")
                if columns and len(columns) == 1:
                    date_info = datetime.strptime(columns[0].text.strip(), '%A, %B %d, %Y').date()
                elif columns:
                    data = [column.text.strip() for column in columns]
                    if events is None or data[3].strip() in events:
                        try:
                            data[0] = pytz.timezone(TIMEZONE).localize(
                                datetime.combine(
                                    date_info, datetime.strptime(
                                        data[0], '%H:%M').time())).astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S %Z%z')
                        except:
                            print(f"Failed to parse UTC time of event: {data}, keep str format")
                        table_data.append(data)
            return table_data
    else:
        print(f"Failed to retrieve the URL. Status code: {response.status_code}")
        return None

# Parse the investing.com website contents
def get_and_parse_webpage(events, date_from, date_to):
    table_data = []
    date_range = [date_from + timedelta(days=x) for x in range((date_to - date_from).days + 1)]
    for date in date_range:
        try:
            table_data += get_and_parse_one_day_webpage(events, date)
        except Exception as e:
            print(f"Failed to get data for {date}: {e}")
    return table_data

def get_and_parse_webpage_to_df(events=None, sdate=None, edate=None, dst='prod', save_dir = '/nas-1/ShareFolder/bb/data/econ_events'):
    table_data = get_and_parse_webpage(events, sdate, edate)
    event_data = pd.DataFrame(table_data, columns=KEYS)

    event_data["Event"] = event_data["Event"].str.upper()
    event_data = event_data[~event_data["Event"].str.contains("HOLIDAY")]
    event_data = event_data[~event_data["Event"].str.contains("CRUDE OIL")]

    event_data["Event"] = event_data["Event"].str.replace(r"\(|\)", "_", regex=True).str.replace(r"\s+", " ", regex=True).str.strip() # remove multiple spaces and ()
    event_data["Event"] = event_data["Event"].str.replace(r"_(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_", "",
                                                          regex=True).str.strip()

    event_data["Event"] = event_data["Event"].str.replace(r"_(Q1|Q2|Q3|Q4)_", "", regex=True).str.strip()


    df = event_data.query("Impact != 'Holiday'").copy()
    df['Time'] = df['Time'].apply(lambda x : pd.Timestamp(x).tz_convert('Asia/Singapore'))
    #df.set_index('Time', inplace=True)

    # fn = '/nas-1/ShareFolder/bb/data/econ_events/prod/20241214_20250112'
    #
    sdate = sdate.strftime('%Y%m%d')
    edate = edate.strftime('%Y%m%d')
    df.to_csv(f'{save_dir}/{dst}/events_{sdate}_{edate}' + '.csv')
    #df.to_parquet(fn + '.parquet')
    return df

# Example usage
if __name__ == '__main__':
    # Define date range
    sdate = datetime.now()
    edate = datetime.now() + timedelta(days=30)

    # Get all data within the range
    df = get_and_parse_webpage_to_df(sdate=sdate, edate=edate)

    # fn = '/nas-1/ShareFolder/bb/data/econ_events/prod/20241214_20250112'
    #
    # df.to_csv(fn + '.csv')
    # df.to_parquet(fn + '.parquet')
    pass


