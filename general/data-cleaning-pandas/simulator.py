import csv
import random
import helpers

def generate_dataset():
    rows = []
    count = 0

    # generate list of base records: names data + address data
    while count < 20:
        row_to_add = helpers.create_row_base()
        rows.append(row_to_add)

        # randomly add duplicate records 
        if random.choice([1, 2, 3, 4, 5]) > 2:
            rows.append(row_to_add.copy())
            # scramble formatting of street address
            rows[-1][2] = helpers.scramble_capitalization(rows[-1][2])
        count += 1

    # add donation amounts to each record
    helpers.add_donations(rows)

    return rows


with open('simulated_data.csv', 'w') as f:
    f_csv = csv.writer(f)
    
    # write headers first
    f_csv.writerow(['first_name','last_name','street_address',
        'city','state', 'donation'])
    f_csv.writerows(generate_dataset())