import os
import labs.cnn.cnn as cnn
import labs.image_classification.image_classification as ic
import labs.sam.sam as sam


def menu():
    print("#######################################################")
    print("# you're in main menu")
    print("# what do you want to do?")
    print("#\t- press 0 to clear terminal")
    print("#\t- press 1 to open CNN lab")
    print("#\t- press 2 to open Image classification lab")
    print("#\t- press 3 to open Segment Anything lab")
    print("#\t- press x to terminate the program")
    print("#######################################################")


def clear_terminal():
    # for windows
    if os.name == 'nt':
        os.system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        os.system('clear')


def main():
    while True:
        menu()
        action = input()
        if action == "0":
            clear_terminal()
        elif action == "1":
            cnn.cnn_main()
        elif action == "2":
            ic.ic_main()
        elif action == "3":
            sam.sam_main()
        elif action == "x":
            print("see ya!")
            break


main()
