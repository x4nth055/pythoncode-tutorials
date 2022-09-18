import random

def add_donations(rows):
    total_donations = len(rows)
    donations = []

    # create list of random donation values 
    donations = list_of_donations(total_donations)

    # add donations onto main records
    count = 0
    while count < total_donations:
        rows[count].append(donations[count])
        count += 1


def create_row_base():
    first_name_options = ['Rosemaria', 'Jodi', 'Alvy', 'Blake', 'Ellis', '']
    last_name_options = ['Roderick', 'Hesbrook', 'Summerton', 'Rappport', 'Alben', '']
    city_options = ['Hialeah', 'Arlington', 'Springfield', 'Carrollton', 'Cambridge', '']
    state_options = ['CT', 'NY', 'VA', 'WA', 'AZ', '']

    first_name = random.choice(first_name_options)
    last_name = random.choice(last_name_options)
    street =  street_address()
    city = random.choice(city_options)
    state = random.choice(state_options)
    
    return [
        first_name, 
        last_name, 
        street, 
        city, 
        state
    ]


def list_of_donations(size):
    donations = []

    donation_amt = random_dollar_amt()
    for i in range(size):
        # randomly change donation value
        if random.choice([1, 2, 3, 4, 5]) > 1:
            donation_amt = random_dollar_amt()
        donations.append(donation_amt)

    return donations


def random_dollar_amt():
    dollars = random.randint(-50, 200)
    cents = random.randint(0, 99)
    return '${}.{}'.format(dollars, cents)


def scramble_capitalization(str):
    final_str = ''
    for letter in str:
        final_str += random.choice([letter.upper(), letter.lower()])
    return final_str


def street_address():
    num = random.randint(40,1001)
    road_name = random.choice(['Western Plank', 'Forest Run', 'Kings', 'Oaktree'])
    road_type = random.choice(['Street', 'St', 'Road', 'Rd', ''])

    address = '{} {} {}'.format(num, road_name, road_type)   
    return address

