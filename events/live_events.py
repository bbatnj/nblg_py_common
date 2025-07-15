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

def get_and_parse_webpage_to_df(events=None, date_from=None, date_to=None):
    table_data = get_and_parse_webpage(events, date_from, date_to)
    df_result = pd.DataFrame(table_data, columns=KEYS)
    return df_result

# Example usage
if __name__ == '__main__':
    # Define date range
    date_from = datetime.now()
    date_to = datetime.now() + timedelta(days=8)

    # Get all data within the range
    df0 = get_and_parse_webpage_to_df(date_from=date_from, date_to=date_to)
    df0['Time'] = df0['Time'].apply(lambda x : pd.Timestamp(x).tz_convert('Asia/Singapore'))
    df0.set_index('Time', inplace=True)

    print('================ Data from last 7 days =================')
    print(df0)

    pass

    # Filter specific events
    # exp_events = ['Saudi Arabia - Eid al-Adha', 'PPI (YoY)  (May)', 'Gasoline Inventories']
    # df = get_and_parse_webpage_to_df(events=exp_events, date_from=date_from, date_to=date_to)
    # print('================ Selected events =================')
    # print(df)

    # Save the results
    # df0.to_csv('investing_data.csv', index=False)
