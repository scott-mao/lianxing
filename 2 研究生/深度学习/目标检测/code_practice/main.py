# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def aaa(a):
    b=change(a)
    return b

def change(a):
    a = torch.tensor([10.1],requires_grad=True)
    return a


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    a = torch.tensor([2.1],requires_grad=True)
    aaa(a)
    a=2

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
