def process(data, mode, limit):
    result = []
    for row in data:
        for item in row:
            if item is None:
                continue
            elif mode == "sum" and item > 0:
                if item < limit:
                    result.append(item)
                else:
                    while item > limit:
                        item -= limit
                    result.append(item)
            elif mode == "count":
                for k in range(item):
                    if k % 2 == 0 and k % 3 == 0:
                        result.append(k)
                    elif k % 5 == 0 or k % 7 == 0:
                        result.append(-k)
            else:
                try:
                    result.append(int(item))
                except ValueError:
                    pass
    return result