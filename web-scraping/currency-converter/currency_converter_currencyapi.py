import requests
import urllib.parse as p

API_KEY = "<YOUR_API_KEY>"
base_url = "https://api.currencyapi.com/v3/"

# utility function that both functions will use
def get_currencyapi_data(endpoint, date=None, base_currency="USD", print_all=True):
    """Get the list of currency codes from the API"""
    # construct the url
    url = p.urljoin(base_url, 
                    f"{endpoint}?apikey={API_KEY}{'' if endpoint == 'latest' else f'&date={date}'}&base_currency={base_currency}")
    # make the request
    res = requests.get(url)
    # get the json data
    data = res.json()
    # print all the currency codes and their values
    c = 0
    if print_all:
        for currency_code, currency_name in data.get("data").items():
            print(f"{currency_code}: {currency_name.get('value')}")
            c += 1
    
    print(f"Total: {c} currencies")
    if endpoint == "latest":
        # get the last updated date
        last_updated = data.get("meta").get("last_updated_at")
        print(f"Last updated: {last_updated}")
    return data

def get_latest_rates(base_currency="USD", print_all=True):
    """Get the latest rates from the API"""
    return get_currencyapi_data(endpoint="latest", base_currency=base_currency, print_all=print_all)
    
def get_historical_rates(base_currency="USD", print_all=True, date="2023-01-01"):
    """Get the historical rates from the Currency API
    `date` must be in the format of YYYY-MM-DD"""
    return get_currencyapi_data(endpoint="historical", base_currency=base_currency, date=date, print_all=print_all)
    
    
if __name__ == "__main__":
    latest_rates = get_latest_rates()
    print(f"\n{'-'*50}\n")
    # get the historical rates for the date 2021-01-01
    historical_rates = get_historical_rates(date="2021-01-01", print_all=False)
    # get EUR rate, for example
    eur_rate = historical_rates.get("data").get("EUR").get("value")
    print(f"EUR rate on 2021-01-01: {eur_rate}")