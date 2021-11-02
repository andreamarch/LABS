# ex2.1
def ex2_1():
    listOne = [3, 6, 9, 12, 15, 18, 21]
    listTwo = [4, 8, 12, 16, 20, 24, 28]
    listThree = list(listOne)
    for i in range(0, len(listThree)):
        if i % 2 == 0:
            listThree[i] = listTwo[i]
    print(listThree)
    return


# ex2.2
def ex2_2():
    sampleList = [34, 54, 67, 89, 11, 43, 94]
    temp = sampleList[3]
    sampleList.remove(temp)
    sampleList[1] += temp
    sampleList[len(sampleList) - 1] += temp
    print(sampleList)
    return


# ex2.3
def ex2_3():
    sampleList = [11, 45, 8, 23, 14, 12, 78, 45, 89]
    flag1 = 0
    flag2 = 0
    listOne = []
    listTwo = []
    listThree = []
    listOne = sampleList[0:3]
    listTwo = sampleList[3:6]
    listThree = sampleList[6:9]
    print(listOne)
    print(listTwo)
    print(listThree)
    return


# ex 2.4
def ex2_4():
    sampleList = [11, 45, 8, 11, 23, 45, 23, 45, 89]
    elements = []
    mydict = {}
    cnt_list = []
    x = 0
    for i in sampleList:
        cnt = 0
        flag = 0
        if i in elements:
            continue
        elements.append(i)
        cnt = sampleList.count(i)
        cnt_list.append(cnt)
        mydict[elements[x]] = cnt_list[x]
        x += 1
    print(mydict)
    return

# ex 2.5
# ??

# ex 2.6
def ex2_6():
    firstSet = {23, 42, 65, 57, 78, 83, 29}
    secondSet = {57, 83, 29, 67, 73, 43, 48}

    for i in secondSet:
        if i in firstSet:
            firstSet.remove(i)
    print(firstSet)
    return

# ex 2.7
def ex2_7():
    secondSet = {57, 83, 29}
    firstSet = {57, 83, 29, 67, 73, 43, 48}
    if firstSet.issubset(secondSet):
        firstSet.clear()
        print('firstSet deleted')
    else:
        secondSet.clear()
        print('secondSet deleted')
    return

# ex 2.8
def ex2_8():
    rollNumber = [47, 64, 69, 76, 37, 83, 98, 97]
    sampleDict = {'John': 47, 'Emma': 69, 'Kelly': 76, 'Jason': 97}
    for i in range(0, len(rollNumber)):
        if rollNumber[i] not in sampleDict.values():
            rollNumber[i] = False
    for i in rollNumber:
        rollNumber.remove(False)
    print(rollNumber)
    return

# ex 2.9
def ex2_9():
    speed = {'Jan': 47, 'Feb': 52, 'March': 47, 'April': 44, 'May': 52, 'June': 53, 'July': 54, 'Aug': 44, 'Sept': 54}
    mylist=[]
    for i in speed.values():
        if i not in mylist:
            mylist.append(i)
        else:
            continue
    print(mylist)
    return

# ex 2.10
def ex2_10():
    sampleList = [87, 52, 44, 53, 54, 87, 52, 53]
    sampleList = list(set(sampleList))
    t = tuple(sampleList)
    print('Maximum value is: ', max(t))
    print('Minimum value is: ', min(t))
    return

ex2_10()