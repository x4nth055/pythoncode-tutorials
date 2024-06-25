import argparse  # Import argparse for command-line argument parsing
import re  # Import re for regular expression matching

# Validate credit card number using Luhn Algorithm
def luhn_algorithm(card_number):
    def digits_of(n):
        return [int(d) for d in str(n)]  # Convert each character in the number to an integer
    
    digits = digits_of(card_number)  # Get all digits of the card number
    odd_digits = digits[-1::-2]  # Get digits from the right, skipping one digit each time (odd positions)
    even_digits = digits[-2::-2]  # Get every second digit from the right (even positions)
    
    checksum = sum(odd_digits)  # Sum all odd position digits
    for d in even_digits:
        checksum += sum(digits_of(d*2))  # Double each even position digit and sum the resulting digits
    
    return checksum % 10 == 0  # Return True if checksum modulo 10 is 0


# Function to check credit card number using Luhn's alogorithm
def check_credit_card_number(card_number):
    card_number = card_number.replace(' ', '')  # Remove spaces from the card number
    if not card_number.isdigit():  # Check if the card number contains only digits
        return False
    return luhn_algorithm(card_number)  # Validate using the Luhn algorithm

# Function to get the card type based on card number using RegEx
def get_card_type(card_number):
    card_number = card_number.replace(' ', '')  # Remove spaces from the card number
    card_types = {
        "Visa": r"^4[0-9]{12}(?:[0-9]{3})?$",  # Visa: Starts with 4, length 13 or 16
        "MasterCard": r"^5[1-5][0-9]{14}$",  # MasterCard: Starts with 51-55, length 16
        "American Express": r"^3[47][0-9]{13}$",  # AmEx: Starts with 34 or 37, length 15
        "Discover": r"^6(?:011|5[0-9]{2})[0-9]{12}$",  # Discover: Starts with 6011 or 65, length 16
        "JCB": r"^(?:2131|1800|35\d{3})\d{11}$",  # JCB: Starts with 2131, 1800, or 35, length 15 or 16
        "Diners Club": r"^3(?:0[0-5]|[68][0-9])[0-9]{11}$",  # Diners Club: Starts with 300-305, 36, or 38, length 14
        "Maestro": r"^(5018|5020|5038|56|57|58|6304|6759|676[1-3])\d{8,15}$",  # Maestro: Various starting patterns, length 12-19
        "Verve": r"^(506[01]|507[89]|6500)\d{12,15}$"  # Verve: Starts with 5060, 5061, 5078, 5079, or 6500, length 16-19
    }
    
    for card_type, pattern in card_types.items():
        if re.match(pattern, card_number):  # Check if card number matches the pattern
            return card_type
    return "Unknown"  # Return Unknown if no pattern matches


# Processing a file containing card numbers.
def process_file(file_path):
   
    try:
        with open(file_path, 'r') as file:  # Open the file for reading
            card_numbers = file.readlines()  # Read all lines from the file
        results = {}
        for card_number in card_numbers:
            card_number = card_number.strip()  # Remove any leading/trailing whitespace
            is_valid = check_credit_card_number(card_number)  # Validate card number
            card_type = get_card_type(card_number)  # Detect card type
            results[card_number] = (is_valid, card_type)  # Store result
        return results
    except Exception as e:
        print(f"Error reading file: {e}")  # Print error message if file cannot be read
        return None


def main():
    parser = argparse.ArgumentParser(description="Check if a credit card number is legitimate and identify its type using the Luhn algorithm.")
    parser.add_argument('-n', '--number', type=str, help="A single credit card number to validate.")  # Argument for single card number
    parser.add_argument('-f', '--file', type=str, help="A file containing multiple credit card numbers to validate.")  # Argument for file input
    
    args = parser.parse_args()  # Parse command-line arguments
    
    if args.number:
        is_valid = check_credit_card_number(args.number)  # Validate single card number
        card_type = get_card_type(args.number)  # Detect card type
        print(f"[!] Credit card number {args.number} is {'valid' if is_valid else 'invalid'} and is of type {card_type}.")  # Print result
    
    if args.file:
        results = process_file(args.file)  # Process file with card numbers
        if results:
            for card_number, (is_valid, card_type) in results.items():
                print(f"[!] Credit card number {card_number} is {'valid' if is_valid else 'invalid'} and is of type {card_type}.")  # Print results for each card number

# Execute tha main function
if __name__ == '__main__':
    main()  
