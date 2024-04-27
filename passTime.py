from __future__ import division
from getpass import getpass
import re

# Load external datasets (common passwords, dictionaries, first names, last names, etc.)
def load_dataset(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file]

common_passwords = load_dataset("common_passwords.csv")  # List of common passwords
entropies = {'Uppercase characters': 26,
             'Lowercase characters': 26,
             'Special characters': 33,
             'Numbers': 10}

crack_speed = 20000000000
# Function to perform substitution attacks
def perform_substitution(password):
    substitutions = {
        'a': ['@', '4'],
        'b': ['8'],
        'e': ['3'],
        'i': ['1', '!'],
        'l': ['1'],
        'o': ['0'],
        's': ['$', '5'],
        't': ['7'],
    }
    modified_password = password.lower()
    for char, replacements in substitutions.items():
        for replacement in replacements:
            modified_password = modified_password.replace(char, replacement)
    return modified_password

# Function to check for sequential characters
def check_sequential_characters(password):
    sequential_patterns = ['123', '234', '345', '456', '567', '678', '789', '890']
    for pattern in sequential_patterns:
        if pattern in password:
            return True
    return False

# Function to check for proximity of characters on the keyboard
def check_keyboard_proximity(password):
    keyboard_patterns = ['qwerty', 'asdf', 'zxcv', 'uiop', 'jkl', 'mnb', '123qwe', '123456']
    for pattern in keyboard_patterns:
        if pattern in password:
            return True
    return False

def evaluate_password_strength(password):
    # Initialize strength score and entropy
    entropy = 0
    mixed_case = False
    
    # Check if password is a common password
    if password in common_passwords:
        entropy -= 50  # Subtract a penalty score for using a common password
    
    # Perform substitution attacks
    modified_password = perform_substitution(password)
    if modified_password != password:
        entropy -= 30  # Subtract a penalty score for using a modified common word or name
    
    # Check for sequential characters
    if check_sequential_characters(password):
        entropy -= 20  # Subtract a penalty score for using sequential characters
    
    # Check for proximity of characters on the keyboard
    if check_keyboard_proximity(password):
        entropy -= 20  # Subtract a penalty score for using characters with keyboard proximity
    
    # Calculate password length and character types
    pass_len = len(password)
    policies = {'Uppercase characters': 0,
                'Lowercase characters': 0,
                'Special characters': 0,
                'Numbers': 0}

    for char in password:
        if re.match("[0-9]", char):
            policies["Numbers"] += 1
        elif re.match("[a-z]", char):
            policies["Lowercase characters"] += 1
        elif re.match("[A-Z]", char):
            policies["Uppercase characters"] += 1
            mixed_case = True
        else:
            policies["Special characters"] += 1
    
    # Calculate entropy based on password characteristics
    entropy = calculate_entropy(pass_len, policies, mixed_case)
    
    return entropy


# Function to calculate entropy based on password characteristics
def calculate_entropy(pass_len, policies, mixed_case):
    entropy = 0

    # Calculate entropy based on the length of the password
    entropy += pass_len * 4

    # Increase entropy for the presence of different character types
    for policy, count in policies.items():
        if count > 0:
            entropy += count * entropies[policy]

    # Increase entropy if mixed case is present
    if mixed_case:
        entropy *= 1.5

    return entropy

# Function to format time
def format_time(time_value):
    if time_value < 1:
        return round(time_value, 3), 'seconds'

    units = ['seconds', 'minutes', 'hours', 'days', 'years', 'decades', 'centuries']
    magnitudes = [60, 60, 24, 365, 10, 10, 10]

    for i in range(len(units)):
        if time_value < magnitudes[i]:
            return round(time_value, 3), units[i]
        time_value /= magnitudes[i]

    # After centuries, adjust the units dynamically
    large_units = ['thousand', 'million', 'billion', 'trillion']
    large_magnitudes = [1000, 1000, 1000, 1000]

    for i in range(len(large_units)):
        if time_value < large_magnitudes[i]:
            return round(time_value/100, 3), f"{large_units[i]} {units[-1]}"
        time_value /= large_magnitudes[i]

    if time_value >= 100:
        return round(time_value/100, 3), 'Incredibly long time'

    # Handle values beyond trillion
    remaining_units = ['thousand', 'million', 'billion', 'trillion']
    remaining_magnitudes = [1000, 1000, 1000, 1000]
    remaining = len(remaining_units)

    while time_value >= remaining_magnitudes[-1]:
        remaining_magnitudes.append(remaining_magnitudes[-1] * 1000)
        remaining_units.append(remaining_units[-1] + ' trillion')

    for i in range(len(remaining_units)):
        if time_value < remaining_magnitudes[i]:
            rounded_value = round(time_value, 3)
            return rounded_value, f"{remaining_units[i - 1]} years"

    return round(time_value, 3), 'Incredibly long time'



import math
def custom_scaling(password):
    length = len(password)
    if length < 4:
        return length
    special_chars = sum(10 for char in password if char in "!@#$%^&*()_+[]{}|;':,.<>?")
    numbers = sum(5 for char in password if char.isdigit())
    uppercase = sum(2 for char in password if char.isupper())
    lowercase = sum(1 for char in password if char.islower())

    strength_factors = (special_chars * 2) + (numbers * 2) + (uppercase * 2) + lowercase

    return max(3 + math.log(strength_factors - 2), 5)

def main():
    password = input("Enter Password: ")
    
    # Evaluate password strength
    entropy = evaluate_password_strength(password)
    
    # Calculate the time to crack the password
    length_factor = custom_scaling(password)
    cracked_seconds = ((entropy ** length_factor) / crack_speed)  # Time in seconds
    time_value, time_unit = format_time(cracked_seconds)
    
    # Print the time to crack the password
    if time_unit == 'Incredibly long time':
        print("\n[+] Time to crack password:", time_unit)
    else:
        print("\n[+] Time to crack password:", time_value, time_unit)

if __name__ == "__main__":
    main()
