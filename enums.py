import enum


class Age_group(enum.Enum):
    A = range(1, 11)
    B = range(10, 21)
    C = range(20, 31)
    D = range(30, 41)


class Disease_states(enum.Enum):
    S = "Suceptible"
    E = "Exposed"
    IAS = "Infected asymptomatic"
    IP = "Infected presymptomatic"
    IS = "Infected symptomatic"
    R = "Recovered"
    H = "Hospitalized"
    D = "Death"


class Traffic_light(enum.Enum):
    G = "G"
    O = "O"
    R = "R"
