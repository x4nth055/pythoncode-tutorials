from geopy.geocoders import Nominatim
import time
from pprint import pprint

# instantiate a new Nominatim client
app = Nominatim(user_agent="tutorial")
# get location raw data
location = app.geocode("Nairobi, Kenya").raw
# print raw data
pprint(location)


def get_location_by_address(address):
    """This function returns a location as raw from an address
    will repeat until success"""
    time.sleep(1)
    try:
        return app.geocode(address).raw
    except:
        return get_location_by_address(address)

address = "Makai Road, Masaki, Dar es Salaam, Tanzania"
location = get_location_by_address(address)
latitude = location["lat"]
longitude = location["lon"]
print(f"{latitude}, {longitude}")
# print all returned data
pprint(location)