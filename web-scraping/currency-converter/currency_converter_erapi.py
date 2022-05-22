import requests
from dateutil.parser import parse 

def get_all_exchange_rates_erapi(src):
    url = f"https://open.er-api.com/v6/latest/{src}"
    # request the open ExchangeRate API and convert to Python dict using .json()
    data = requests.get(url).json()
    if data["result"] == "success":
        # request successful
        # get the last updated datetime
        last_updated_datetime = parse(data["time_last_update_utc"])
        # get the exchange rates
        exchange_rates = data["rates"]
    return last_updated_datetime, exchange_rates
        
        

def convert_currency_erapi(src, dst, amount):
    # get all the exchange rates
    last_updated_datetime, exchange_rates = get_all_exchange_rates_erapi(src)
    # convert by simply getting the target currency exchange rate and multiply by the amount
    return last_updated_datetime, exchange_rates[dst] * amount


if __name__ == "__main__":
    import sys
    source_currency = sys.argv[1]
    destination_currency = sys.argv[2]
    amount = float(sys.argv[3])
    last_updated_datetime, exchange_rate = convert_currency_erapi(source_currency, destination_currency, amount)
    print("Last updated datetime:", last_updated_datetime)
    print(f"{amount} {source_currency} = {exchange_rate} {destination_currency}")
    