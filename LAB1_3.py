import numpy as np

# ex 3.1
def ex3_1():
    arr = np.arange(8).reshape(4, 2)
    print(arr)
    return

# ex 3.2
def ex3_2():
    arr = np.arange(100, 200, 10).reshape(5, 2)
    print(arr)
    return

# ex 3.3
def ex3_3():
    arr = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
    arr2 = np.array([arr[0, 2], arr[1, 2], arr[2, 2]])
    print(arr2)
    return

# ex 3.4
def ex3_4():
    arr = np.array([[3, 6, 9, 12], [15, 18, 21, 24], [27, 30, 33, 36], [39, 42, 45, 48], [51, 54, 57, 60]])
    arr2 = np.array([0, 0, 0, 0])
    x = 0
    for i in range(0, 4):
        for j in range(0, 3):
            modi = i % 2
            modj = j % 2
            if (modi != 0) & (modj == 0):
                arr2[x] = arr[i, j]
                x += 1
    print(arr2)
    return

# ex 3.5
def ex3_5():
    arr1 = np.array([[5, 6, 9], [21, 18, 27]])
    arr2 = np.array([[15, 33, 24], [4, 7, 1]])
    arr3 = np.sqrt(arr1+arr2)
    print(arr3)
    return

# ex 3.6
def ex3_6():
    arr = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66]])
    print('Unsorted array:\n', arr)
    arr = np.sort(arr)
    i = list(np.shape(arr))
    for j in range(0, i[0]-1):
        while arr[j, 2] > arr[j + 1, 0]:
            temp = arr[j, 2]
            arr[j, 2] = arr[j+1, 0]
            arr[j+1, 0] = temp
            arr = np.sort(arr)
    print('Sorted array:\n', arr)
    return

# ex 3.7
def ex3_7():
    arr = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66]])
    print(max(arr.max(axis=0)))
    print(min(arr.min(axis=1)))
    return

# ex 3.8
def ex3_8():
    arr = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66]])
    newcol = np.array([10, 10, 10])
    arr = np.delete(arr, 1, 1)
    arr = np.c_[arr[:, 0], newcol, arr[:, 1]]
    print(arr)
