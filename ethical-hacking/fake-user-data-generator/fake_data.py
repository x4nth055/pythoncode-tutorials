# Import necessary libraries and modules.
from faker import Faker
from faker.providers import internet
import csv


# Function to generate user data with the specified number of users.
def generate_user_data(num_of_users):
    # Create a Faker instance.
    fake = Faker()
    # Add the Internet provider to generate email addresses and IP addresses.
    fake.add_provider(internet)

    # Initialize an empty list to store user data.
    user_data = []
    # Loop to generate data for the specified number of users.
    for _ in range(num_of_users):
        # Create a dictionary representing a user with various attributes.
        user = {
            'Name': fake.name(),
            'Email': fake.free_email(),
            'Phone Number': fake.phone_number(),
            'Birthdate': fake.date_of_birth(),
            'Address': fake.address(),
            'City': fake.city(),
            'Country': fake.country(),
            'ZIP Code': fake.zipcode(),
            'Job Title': fake.job(),
            'Company': fake.company(),
            'IP Address': fake.ipv4_private(),
            'Credit Card Number': fake.credit_card_number(),
            'Username': fake.user_name(),
            'Website': fake.url(),
            'SSN': fake.ssn()
        }
        # Append the user data dictionary to the user_data list.
        user_data.append(user)

    # Return the list of generated user data.
    return user_data


# Function to save user data to a CSV file.
def save_to_csv(data, filename):
    # Get the keys (column names) from the first dictionary in the data list.
    keys = data[0].keys()
    # Open the CSV file for writing.
    with open(filename, 'w', newline='') as output_file:
        # Create a CSV writer with the specified column names.
        writer = csv.DictWriter(output_file, fieldnames=keys)
        # Write the header row to the CSV file.
        writer.writeheader()
        # Iterate through each user dictionary and write a row to the CSV file.
        for user in data:
            writer.writerow(user)
    # Print a success message indicating that the data has been saved to the file.
    print(f'[+] Data saved to {filename} successfully.')


# Function to save user data to a text file.
def save_to_text(data, filename):
    # Open the text file for writing.
    with open(filename, 'w') as output_file:
        # Iterate through each user dictionary.
        for user in data:
            # Iterate through key-value pairs in the user dictionary and write to the text file.
            for key, value in user.items():
                output_file.write(f"{key}: {value}\n")
            # Add a newline between users in the text file.
            output_file.write('\n')
    # Print a success message indicating that the data has been saved to the file.
    print(f'[+] Data saved to {filename} successfully.')


# Function to print user data vertically.
def print_data_vertically(data):
    # Iterate through each user dictionary in the data list.
    for user in data:
        # Iterate through key-value pairs in the user dictionary and print vertically.
        for key, value in user.items():
            print(f"{key}: {value}")
        # Add a newline between users.
        print()


# Get the number of users from user input.
number_of_users = int(input("[!] Enter the number of users to generate: "))
# Generate user data using the specified number of users.
user_data = generate_user_data(number_of_users)

# Ask the user if they want to save the data to a file.
save_option = input("[?] Do you want to save the data to a file? (yes/no): ").lower()

# If the user chooses to save the data.
if save_option == 'yes':
    # Ask the user for the file type (CSV, TXT, or both).
    file_type = input("[!] Enter file type (csv/txt/both): ").lower()

    # Save to CSV if the user chose CSV or both.
    if file_type == 'csv' or file_type == 'both':
        # Ask the user for the CSV filename.
        custom_filename_csv = input("[!] Enter the CSV filename (without extension): ")
        # Concatenate the filename with the .csv extension.
        filename_csv = f"{custom_filename_csv}.csv"
        # Call the save_to_csv function to save the data to the CSV file.
        save_to_csv(user_data, filename_csv)

    # Save to TXT if the user chose TXT or both.
    if file_type == 'txt' or file_type == 'both':
        # Ask the user for the TXT filename.
        custom_filename_txt = input("[!] Enter the TXT filename (without extension): ")
        # Concatenate the filename with the .txt extension.
        filename_txt = f"{custom_filename_txt}.txt"
        # Call the save_to_text function to save the data to the text file.
        save_to_text(user_data, filename_txt)

    # If the user entered an invalid file type.
    if file_type not in ['csv', 'txt', 'both']:
        # Print an error message indicating that the file type is invalid.
        print("[-] Invalid file type. Data not saved.")
# If the user chose not to save the data, print it vertically.
else:
    # Call the print_data_vertically function to print the data vertically.
    print_data_vertically(user_data)
