import numpy as np
import matplotlib.pyplot as plt
import string as strn
# 1.1
def ex1_1():
    val1 = int(input('Enter value 1: '))
    val2 = int(input('Enter value 2: '))

    prod = val1 * val2

    if prod >= 1000:
        print('\nThe sum is: ')
        print(val1 + val2)
    return

# 1.2
def ex1_2():
    for i in range(0, 7):
        print(i + i - 1)
    return

# 1.3
def ex1_3():
    lst = [1, 2, 4, 5, 2, 6, 1]

    if lst[0] == lst[-1]:
        statuss = True
    else:
        statuss = False
    return statuss

# 1.4
def ex1_4():
    lst = [1, 100, 500, 3, 8, 78, 985, 5, 10, 11]

    for el in lst:
        modulo = el % 5
        if modulo == 0:
            print(el)
    return

# ex1.5
def ex1_5(string):
    cntr = 0
    word= ''
    for el in string:
        if el != ' ':
            word += el
        elif el == ' ':
            if word == 'Emma':
                cntr += 1
            word = ''
    return cntr

# ex1.6
def ex1_6(list1, list2):
    list3 = list(list1)
    for i in range(0, len(list3)):
        if i % 2 == 0:
            list3[i] = list2[i]
    return list3

# ex1.7
def ex1_7():
    s1 = 'abcdefgh'
    s2 = '12345678'
    str = s1[0:4] + s2 + s1[4:8]
    print(str)
    return

# ex1.8
def ex1_8():
    s1 = 'abcdefghi'
    s2 = '123456789'
    str = s1[0] + s1[int(np.floor(len(s1)/2))]+ s1[len(s1)-1] + s2[0] + s2[int(np.floor(len(s1)/2))] + s2[len(s1)-1]
    print(str)
    return

# ex 1.9
def ex1_9():
    str = input('insert string here: ')
    cntL = 0
    cntU = 0
    cntD = 0
    cntS = 0
    for i in str:
        if i.isalpha() | i.isdigit():
            if i.islower():
                cntL += 1
            elif i.isupper():
                cntU += 1
            elif i.isdigit():
                cntD += 1
            continue
        else:
            if i != ' ':
                cntS += 1
    print('Number of lower case letters: ', cntL)
    print('Number of upper case letters: ', cntU)
    print('Number of digits: ', cntD)
    print('Number of special symbols: ', cntS)
    return

# ex 1.10
def ex1_10(string):
    string = string.lower()
    cntr = 0
    word = ''
    for el in string:
        if el != ' ':
            word += el
        elif el == ' ':
            if word == 'usa':
                cntr += 1
            word = ''
    print('The number of occurrences is: ', cntr)
    return

# ex 1.11
def ex1_11(string):
    cnt = 0
    mysum = 0
    for i in string:
        if i.isdigit():
            cnt += 1
            mysum += int(i)
        else:
            continue
        mymean = mysum / cnt
    print('The sum of the appeared digits is: ', mysum)
    print('The mean of the appeared digits is: ', mymean)
    return

def ex1_12(string):
    string = string.lower()
    word = ''
    for i in string:
        cnt = 0
        flag = 0
        if i == ' ':
            continue
        if i in word:
            continue
        word += i
        cnt = string.count(i)
        print('The number of occurrences of', i, 'is: ', cnt)
string = 'Emma is a good developer, Emma is also a writer. Emma sucks at everything else.'

# list1 = [9, 2, 3, 5, 1, 7, 2, 10]
# list2 = [4, 3, 8, 3, 2, 6, 8, 2]

# string = 'USA stands for United states of Murica. usa are a country in north america. UsA suck'
#string = 'I have 3 apples, 9 oranges, 4 bananas, 5 peas and 2 carrots'

ex1_12(string)