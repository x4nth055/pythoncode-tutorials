# Default Library
import requests
import json
import os

# Download with pip install pycountry
import pycountry

for country in list(pycountry.countries):

    # All Points from all Groups
    # used to analyze
    allPoints = []

    # Countries that dont consist of one body 
    # will have multiple groups of coordinates
    pointGroups = []

    # Country Code with 3 letters
    countryCode = country.alpha_3
    countryName = country.name

    # Check if the SVG file already Exists and skip if it does
    if os.path.exists(f'output/{countryName}.svg'):
        print(f'{countryName}.svg Already exists ... Skipping to next Country\n')
        continue

    print('Generating Map for: ', countryName)

    # Get the Data
    re = requests.get(f'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{countryCode}_0.json')

    # If the string cant be parsed an invalid country was requested
    try:
        data = json.loads(re.text)
    except json.decoder.JSONDecodeError:
        print('Could not decode ... Skipping to next Country\n')
        continue

    # Organise the Data 
    # Get the groups and all coordinates
    for i in data['features'][0]['geometry']['coordinates']:
        for group in i:
            pointGroups.append(group)
            for coord in group:
                allPoints.append(coord)

    print(f'\n{len(allPoints)} Points')

    # Analyse Data
    # Use these Information to calculate 
    # offset, height and width of the Country
    lowestX = 9999999999
    highestX = -9999999999

    lowestY = 9999999999
    highestY = -9999999999

    for x, y in allPoints:
        lowestX = x if x < lowestX else lowestX
        highestX = x if x > highestX else highestX

        lowestY = y if y < lowestY else lowestY
        highestY = y if y > highestY else highestY

    print('lowestX', lowestX)
    print('highestX', highestX)

    print('lowestY', lowestY)
    print('highestY', highestY)

    svgWidth = (highestX - lowestX)
    svgHeight = (highestY - lowestY)

    # Transfrom Points to Polygon Strings
    polygonString = ''
    for group in pointGroups:
        coordinateString = ''
        for x, y in group:
            x  = (x - lowestX)
            y  = (y - lowestY)
            
            coordinateString = coordinateString + f'{x},{y} '

        polygonString += f'<polygon points="{coordinateString}"></polygon>'

    svgContent = f"""
    <svg width="{svgWidth}" height="{svgHeight}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="transform: scale(1, -1)">
        {polygonString}
    </svg>
    """

    # make the output folder
    if not os.path.isdir("output"):
        os.mkdir("output")
    # write the svg file
    with open(f'output/{countryName}.svg', 'w') as f:
        f.write(svgContent)
    # new line
    print('\n')
