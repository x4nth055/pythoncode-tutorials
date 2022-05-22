import requests
from datetime import date, datetime

API_KEY = "8c3dce10dc5fdb6ec1f555a1504b1373"
# API_KEY = "<YOUR_API_KEY_HERE>"


def convert_currency_fixerapi_free(src, dst, amount):
    """converts `amount` from the `src` currency to `dst` using the free account"""
    url = f"http://data.fixer.io/api/latest?access_key={API_KEY}&symbols={src},{dst}&format=1"
    data = requests.get(url).json()
    if data["success"]:
        # request successful
        rates = data["rates"]
        # since we have the rate for our currency to src and dst, we can get exchange rate between both
        # using below calculation
        exchange_rate = 1 / rates[src] * rates[dst]
        last_updated_datetime = datetime.fromtimestamp(data["timestamp"])
        return last_updated_datetime, exchange_rate * amount
    
    
def convert_currency_fixerapi(src, dst, amount):
    """converts `amount` from the `src` currency to `dst`, requires upgraded account"""
    url = f"https://data.fixer.io/api/convert?access_key={API_KEY}&from={src}&to={dst}&amount={amount}"
    data = requests.get(url).json()
    if data["success"]:
        # request successful
        # get the latest datetime
        last_updated_datetime = datetime.fromtimestamp(data["info"]["timestamp"])
        # get the result based on the latest price
        result = data["result"]
        return last_updated_datetime, result
        

    
if __name__ == "__main__":
    import sys
    source_currency = sys.argv[1]
    destination_currency = sys.argv[2]
    amount = float(sys.argv[3])
    # free account
    last_updated_datetime, exchange_rate = convert_currency_fixerapi_free(source_currency, destination_currency, amount)
    # upgraded account, uncomment if you have one
    # last_updated_datetime, exchange_rate = convert_currency_fixerapi(source_currency, destination_currency, amount)
    print("Last updated datetime:", last_updated_datetime)
    print(f"{amount} {source_currency} = {exchange_rate} {destination_currency}")
    