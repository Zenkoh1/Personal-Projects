def double_index(lst, index):
    try:
        lst[index] *= 2
        return lst
    except:
        return lst
print(double_index([1, 2, 3, 4], 10))