# [How to Extract Weather Data from Google in Python](https://www.thepythoncode.com/article/extract-weather-data-python)
To run this:
- `pip3 install -r requirements.txt`
- To get the Google weather data of your current region you live in:
    ```
    python weather.py
    ```
- To get the weather of a specific region in the world:
    ```
    python weather.py "New York"
    ```
    This will grab the weather information of "New York" state in the US:
    ```
    Weather for: New York, NY, USA
    Now: wednesday 2:00 PM
    Temperature now: 20°C
    Description: Mostly Cloudy
    Precipitation: 0%
    Humidity: 52%
    Wind: 13 km/h
    Next days:
    ======================================== wednesday ========================================
    Description: Mostly Cloudy
    Max temperature: 21°C
    Min temperature: 12°C
    ======================================== thursday ========================================
    Description: Sunny
    Max temperature: 22°C
    Min temperature: 14°C
    ======================================== friday ========================================
    Description: Partly Sunny
    Max temperature: 28°C
    Min temperature: 18°C
    ======================================== saturday ========================================
    Description: Sunny
    Max temperature: 30°C
    Min temperature: 19°C
    ======================================== sunday ========================================
    Description: Partly Sunny
    Max temperature: 29°C
    Min temperature: 21°C
    ======================================== monday ========================================
    Description: Partly Cloudy
    Max temperature: 30°C
    Min temperature: 19°C
    ======================================== tuesday ========================================
    Description: Mostly Sunny
    Max temperature: 26°C
    Min temperature: 16°C
    ======================================== wednesday ========================================
    Description: Mostly Sunny
    Max temperature: 25°C
    Min temperature: 19°C
```