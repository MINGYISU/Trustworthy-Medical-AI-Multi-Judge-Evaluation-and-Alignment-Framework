
def prRed(s): print("\033[1;31m {}\033[0m".format(s))
def prGreen(s): print("\033[92m {}\033[00m".format(s))
def prYellow(s): print("\033[93m {}\033[00m".format(s))
def prBlue(s): print("\033[94m {}\033[00m".format(s))
def prOrange(s): print("\033[38;5;214m {}\033[00m".format(s))
def prPurple(s): print("\033[95m {}\033[00m".format(s))
def prCyan(s): print("\033[96m {}\033[00m".format(s))
def prLightGray(s): print("\033[97m {}\033[00m".format(s))
def prBlack(s): print("\033[90m {}\033[00m".format(s))

_color_funcs = {
    "red": prRed,
    "green": prGreen,
    "yellow": prYellow,
    "blue": prBlue,
    "orange": prOrange,
    "purple": prPurple,
    "cyan": prCyan,
    "light_gray": prLightGray,
    "black": prBlack,
}

def pr(s, color="green"):
    func = _color_funcs.get(color.lower(), prGreen)
    func(s)