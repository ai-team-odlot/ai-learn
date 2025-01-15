# https://www.w3schools.com/python/default.asp

def menu():
    print("#######################################################")
    print("what do you want to do?")
    print("\t- press 1 to read the file")
    print("\t- press 2 to create new file")
    print("\t- press 3 to append a text to existing file")
    print("\t- press x to terminate the program")
    print("#######################################################")

def read_file():

    print("please provide name of file to be opened")
    file_name = input()

    file = open(file_name, "rt")
    content = file.read()

    print(f"the content of the file {file_name} is")
    print(content)

    file.close()

def save_to_file():

    print("please provide the name of new file")
    file_name = input()

    print("please provide the content of new file")
    content = input()

    file = open(file_name, "x")
    file.write(content)

    file.close()

def append_to_file():

    print("please provide name of file to be opened")
    file_name = input()

    print("please provide text which will be appended to the file")
    content_to_append = input()

    file = open(file_name, "at")
    file.write(content_to_append)

    file.close()

def main():

    while True:
        menu()
        action = input()
        if action == "1":
            read_file()
        elif action == "2":
            save_to_file()
        elif action == "3":
            append_to_file()
        elif action == "x":
            print("goodbye!")
            break

main()
