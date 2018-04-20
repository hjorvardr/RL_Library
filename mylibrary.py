def ma(ts, q):
    acc = 0
    res = []
    for i in range(q, len(ts) - q):
        for j in range(i - q, i + q):
            acc += ts[j]
        res.append(acc / (2 * q + 1))
        acc = 0
    return res