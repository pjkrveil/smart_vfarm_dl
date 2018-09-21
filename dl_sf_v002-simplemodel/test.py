def nearest_square(limit):
    limit = limit ** (0.5)
    y = int (limit)
    while y < limit :
         y = y*y
    return y

test1 = nearest_square(17000)
print("expected result: 36,actual result:{}".format(test1))