


import math
 # NOT NEEDED

def discretize(value: float, avg: float, stddev: float):
    a1 = avg + stddev
    a2 = avg - stddev
    
    if value <= a1 and value >= a2:
        return "FAIR"
    elif value < a1 and value >= (avg - 2 * stddev):
        return "LOW"
    elif value > a2 and value <= (avg + 2 * stddev):
        return "HIGH"
    elif value > avg + 2 * stddev:
        return "VERY HIGH"
    else:
        return "VERY_LOW"


def reverse(level: str, avg: float, stddev: float):
    if level == "VERY_LOW":
        return (0, avg - 2 * stddev)
    elif level == "VERY_HIGH":
        return (avg + 2 * stddev, math.inf)
    elif level == "LOW":
        return (avg - 2 * stddev, avg - stddev)
    elif level == "HIGH":
        return (avg + 2 * stddev, avg + stddev)
    else:
        return (avg - stddev, avg + stddev)
