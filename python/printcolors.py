#!/usr/bin/env python3


# Credit: https://www.geeksforgeeks.org/print-colors-python-terminal/
class colors:
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'
    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgray = '\033[37m'
        darkgray = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'
    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgray = '\033[47m'


if __name__ == '__main__':
    C = colors
    text = 'hello world'

    print(f'{text}')
    print(f'{C.fg.red}{text}{C.reset}')
    print(f'{C.fg.green}{text}{C.reset}')
    print(f'{C.fg.orange}{text}{C.reset}')
    print(f'{C.fg.blue}{text}{C.reset}')
    print(f'{C.fg.purple}{text}{C.reset}')
    print(f'{C.fg.cyan}{text}{C.reset}')
    print(f'{C.fg.lightgray}{text}{C.reset}')
    print(f'{C.fg.darkgray}{text}{C.reset}')
    print(f'{C.fg.lightred}{text}{C.reset}')
    print(f'{C.fg.lightgreen}{text}{C.reset}')
    print(f'{C.fg.yellow}{text}{C.reset}')
    print(f'{C.fg.lightblue}{text}{C.reset}')
    print(f'{C.fg.pink}{text}{C.reset}')
    print(f'{C.fg.lightcyan}{text}{C.reset}')

    print(f'{C.bold}{text}{C.reset}')
    print(f'{C.bold}{C.fg.red}{text}{C.reset}')
    print(f'{C.bold}{C.fg.green}{text}{C.reset}')
    print(f'{C.bold}{C.fg.orange}{text}{C.reset}')
    print(f'{C.bold}{C.fg.blue}{text}{C.reset}')
    print(f'{C.bold}{C.fg.purple}{text}{C.reset}')
    print(f'{C.bold}{C.fg.cyan}{text}{C.reset}')
    print(f'{C.bold}{C.fg.lightgray}{text}{C.reset}')
    print(f'{C.bold}{C.fg.darkgray}{text}{C.reset}')
    print(f'{C.bold}{C.fg.lightred}{text}{C.reset}')
    print(f'{C.bold}{C.fg.lightgreen}{text}{C.reset}')
    print(f'{C.bold}{C.fg.yellow}{text}{C.reset}')
    print(f'{C.bold}{C.fg.lightblue}{text}{C.reset}')
    print(f'{C.bold}{C.fg.pink}{text}{C.reset}')
    print(f'{C.bold}{C.fg.lightcyan}{text}{C.reset}')

    print(f'{C.reverse}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.red}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.green}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.orange}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.blue}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.purple}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.cyan}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.lightgray}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.darkgray}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.lightred}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.lightgreen}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.yellow}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.lightblue}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.pink}{text}{C.reset}')
    print(f'{C.reverse}{C.fg.lightcyan}{text}{C.reset}')
