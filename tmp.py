def multiply(x, y):
    return round(x * y)

def test1():
    assert multiply(4, 5) == 20
    assert multiply(100, 1.1) == 110
    assert multiply('mama', 5) == None