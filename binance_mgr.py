import requests
import json
import sys
import time
import hmac
import hashlib


def get_hmac_sha256_signature(query_string, secret):
    return hmac.new(secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()


# hftops4
api_key = 'QRf0CnHabbSwNI9vhAy5vzgOAd757EWz4ZxqDgtESVURqAE4XFflA7sgm2GoIFIt'
secret_key = 'eyXDoRaJjZrzqAIx2WBEYKAJhVKG74ezMjfIrYV5riJVFjfytosAqLvNGCxuhGjZ'

# binance5
#api_key = '8tRLNwz4MwH4QM9NVFAnhGKTdrbGPMBMq5RBpmO9X3KZK1CA1EnH5UzV5hx4b4Ts'
#secret_key = 'JVorP4Ofh2BigX2yBwbmLgo5h2Vl5JOvGVZQznwiooaNZxgcUWedrCBJSc3GP9ID'


def handmade_query_balance():
    url = 'https://fapi.binance.com/fapi/v3/balance'

    d = f'timestamp={ int(time.time() * 1000) }'
    qry = f'{d}&signature={get_hmac_sha256_signature(d, secret_key)}'
    print(qry)
    resp = requests.get(url, data=qry, headers={
        'X-MBX-APIKEY': api_key
    })
    if resp.status_code != 200:
        print(f"Error: {resp.status_code}, {resp.text}")
        sys.exit(1)

    result = resp.json()
    for asset in result:
        print(f"{asset['asset']}: {asset['balance']} {asset['asset']}")

    print(json.dumps(result, indent=4))


#from binance.um_futures import UMFutures
handmade_query_balance()


def manage_binance_account(api_key, secret_key):
    # Initialize Binance Futures client
    client = UMFutures(api_key, secret_key)

    # 1. Cancel all open orders
    try:
        for symbol in ('BTCUSDT', 'BTCUSDC', 'ETHUSDT', 'SOLUSDT'):
            print("Cancelling all open orders...")
            response = client.cancel_open_orders(symbol)
            print("All open orders cancelled:", response)
    except Exception as e:
        print("Error cancelling orders:", e)

    # 2. Close all positions with market orders
    try:
        print("Fetching all open positions...")
        account_info = client.account()
        positions = account_info.get('positions', [])

        for position in positions:
            symbol = position['symbol']
            position_amt = float(position['positionAmt'])
            if position_amt != 0:  # If there is an open position
                side = 'SELL' if position_amt > 0 else 'BUY'
                print(f"Closing position for {symbol} with market order, Side: {side}")
                response = client.new_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=abs(position_amt)
                )
                print(f"Closed {symbol}: {response}")
    except Exception as e:
        print("Error closing positions:", e)

    # 3. Check current balance
    try:
        print("Fetching account balance...")
        balance_info = client.balance()
        print("Current balance:", balance_info)
    except Exception as e:
        print("Error fetching balance:", e)


# Example usage:
#manage_binance_account(api_key, secret_key)

