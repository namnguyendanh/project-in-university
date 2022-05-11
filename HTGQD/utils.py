import numpy as np

def gen_normal_candidate(n):
    arr = np.random.randint(10, size=n)
    for i in range(len(arr)):
        if arr[i] == 0:
            arr[i] = 1
    return arr

def gen_bad_candiate(n):
    arr = np.random.randint(6, size=n)
    for i in range(len(arr)):
        if arr[i] == 0:
            arr[i] = 1
    return arr

def gen_good_candidate(n):
    arr = np.random.randint(4, size=n)
    arr += 7
    return arr
