from geopy.geocoders import Nominatim
from pprint import pprint
import time

app = Nominatim(user_agent="tutorial")

def get_address_by_location(latitude, longitude, language="en"):
    """This function returns an address as raw from a location
    will repeat until success"""
    # build coordinates string to pass to reverse() function
    coordinates = f"{latitude}, {longitude}"
    # sleep for a second to respect Usage Policy
    time.sleep(1)
    try:
        return app.reverse(coordinates, language=language).raw
    except:
        return get_address_by_location(latitude, longitude)

# define your coordinates
latitude = 36.723
longitude = 3.188
# get the address info
address = get_address_by_location(latitude, longitude)
# print all returned data
pprint(address)