import enum

class Grade(enum.Enum):
    A = 'A'
    B = 'B'
    C = 'C'


class Age_group(enum.Enum):
    A = range(1, 11)
    B = range(10,21)
    C = range(20,31)
    D = range(30,41)
