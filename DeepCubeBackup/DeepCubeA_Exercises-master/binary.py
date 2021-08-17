count = 2
while True:
    print(count)
    count_copy = count
    if count_copy % 2 == 0:
        count_copy = count_copy/2
    elif count_copy % 2 != 0:
        count_copy = count_copy * 3
        count_copy += 1
    while True: 
        if count_copy == 1:
            break
        elif count_copy == count:
            print("Found a number that loops back on itself: ", count)
            quit()
        elif count_copy % 2 == 0:
            count_copy = count_copy/2
        elif count_copy % 2 != 0:
            count_copy = count_copy * 3
            count_copy += 1
    count += 1
