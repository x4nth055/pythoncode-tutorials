import yahoo_fin.stock_info as si
from datetime import datetime, timedelta

def convert_currency_yahoofin(src, dst, amount):
    # construct the currency pair symbol
    symbol = f"{src}{dst}=X"
    # extract minute data of the recent 2 days
    latest_data = si.get_data(symbol, interval="1m", start_date=datetime.now() - timedelta(days=2))
    # get the latest datetime
    last_updated_datetime = latest_data.index[-1].to_pydatetime()
    # get the latest price
    latest_price = latest_data.iloc[-1].close
    # return the latest datetime with the converted amount
    return last_updated_datetime, latest_price * amount


if __name__ == "__main__":
    import sys
    source_currency = sys.argv[1]
    destination_currency = sys.argv[2]
    amount = float(sys.argv[3])
    last_updated_datetime, exchange_rate = convert_currency_yahoofin(source_currency, destination_currency, amount)
    print("Last updated datetime:", last_updated_datetime)
    print(f"{amount} {source_currency} = {exchange_rate} {destination_currency}")