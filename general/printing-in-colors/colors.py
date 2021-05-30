from colorama import init, Fore, Back, Style

# essential for Windows environment
init()
# all available foreground colors
FORES = [ Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE ]
# all available background colors
BACKS = [ Back.BLACK, Back.RED, Back.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN, Back.WHITE ]
# brightness values
BRIGHTNESS = [ Style.DIM, Style.NORMAL, Style.BRIGHT ]


def print_with_color(s, color=Fore.WHITE, brightness=Style.NORMAL, **kwargs):
    """Utility function wrapping the regular `print()` function 
    but with colors and brightness"""
    print(f"{brightness}{color}{s}{Style.RESET_ALL}", **kwargs)

# printing all available foreground colors with different brightness
for fore in FORES:
    for brightness in BRIGHTNESS:
        print_with_color("Hello world!", color=fore, brightness=brightness)

# printing all available foreground and background colors with different brightness
for fore in FORES:
    for back in BACKS:
        for brightness in BRIGHTNESS:
            print_with_color("A", color=back+fore, brightness=brightness, end=' ')
    print()