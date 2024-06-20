from zxcvbn import zxcvbn
import pprint, getpass, sys


def test_single_password():
    password = getpass.getpass("[?] Enter your password: ")
    result = zxcvbn(password)
    print(f"Value: {result['password']}")
    print(f"Password Score: {result['score']}/4")
    print(f"Crack Time: {result['crack_times_display']['offline_slow_hashing_1e4_per_second']}")
    print(f"Feedback: {result['feedback']['suggestions']}")
    #pprint.pp(result)


def test_multiple_passwords(password_file):
    try:
        with open(password_file, 'r') as passwords:
            for password in passwords:
                result = zxcvbn(password.strip('\n'))
                print('\n[+] ######################')# for readability
                print(f"Value: {result['password']}")
                print(f"Password Score: {result['score']}/4")
                print(f"Crack Time: {result['crack_times_display']['offline_slow_hashing_1e4_per_second']}")
                print(f"Feedback: {result['feedback']['suggestions']}")
                #pprint.pp(result)
            
    except Exception:
        print('[!] Please make sure to specify an accessible file containing passwords.')


if len(sys.argv) == 2:
    test_multiple_passwords(sys.argv[1])
elif len(sys.argv) == 1:
    test_single_password()
else:
    print('Usage: python test_password_strength.py <file> (for a file containing passwords) or \
        \npython test_password_strength.py (for a single password.)')