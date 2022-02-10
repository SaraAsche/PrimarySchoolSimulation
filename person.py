import itertools
import random
import decimal
import math
import pickle

import numpy as np
from scipy.optimize import least_squares

from enums import Age_group
from enums import Grade
from layers import Grades, Klasse, Lunchbreak, Recess


# decimal.getcontext().prec = 12


class Person:
    newid = itertools.count()

    def __init__(self, grade, class_group):
        self.id = next(Person.newid)
        self.sex = self.get_gender()
        self.state = "R"
        self.vaccinated = self.get_vaccinated_status()
        self.grade = grade
        self.age = self.get_valid_age(grade)
        self.age_group = self.set_age_group()
        self.class_group = str(class_group)
        self.lunch_group = self.set_lunch_group()
        self.interactions = {}

        self.const_bias = 30 * (math.log10(1 / random.random()))

        self.bias_vector = {}
        self.p_vector = {}

    def get_valid_age(self, grade, is_teacher=False):
        return random.choice([grade + 4, grade + 5])

    def calc_probability_of_interacting(self, person):
        return random.random()

    def set_lunch_group(self):
        if self.grade < 4:
            return 1
        else:
            return 0

    def set_age_group(self):
        for group in Age_group:
            if self.age in group.value:
                return group
        return None

    def add_interaction(self, interaction):
        p1 = interaction.p1
        p2 = interaction.p2

        other = p1 if p1 != self else p2

        self.interactions[other] = interaction

    def get_interaction(self, p):
        return self.interactions.get(p, Interaction(self, p, 0))

    def interacted(self, p):
        return True

    def get_gender(self):
        return "F" if random.random() > 0.5 else "M"

    def get_vaccinated_status(self):
        return 1 if random.random() > 0.2 else 0

    def has_interacted_with(self, p):
        return p in self.interactions

    def getID(self):
        return self.id

    def getGrade(self):
        return self.grade

    def getClass(self):
        return self.class_group

    def get_class_and_grade(self):
        return str(self.grade) + self.class_group

    def getLunchgroup(self):
        return self.lunch_group

    def generate_bias_vector(self, students):
        for i in range(len(students)):
            self.bias_vector[students[i]] = self.const_bias

    def generate_p_vector(self, students, X):
        rand = True
        if len(X):
            rand = True
            ## Off-diagonal excluding lunch
            a1 = X[0]
            b1 = X[1]

            ## Off-diagonal with lunch
            a2 = X[2]
            b2 = X[3]

            ## Grade-Grade
            a3 = 40  # X[4]
            b3 = 0.1  # X[5]

            ## Class-Class
            a4 = 200  # X[6]
            b4 = 0.1  # X[7]
        else:
            ## Off-diagonal excluding lunch
            a1 = 0.2  # 1.5
            b1 = 3.5  # 1.25
            ##  Off-diagonal with lunch
            a2 = 0.01  # 0.001
            b2 = 0.01  # 0.07

            ## Grade-Grade
            a3 = 3  # 100
            b3 = 2  # 0.3

            ## Class-Class
            a4 = 6  # 10000
            b4 = 3  # 1

        for i in range(len(students)):
            same_lunch = self.lunch_group == students[i].lunch_group
            same_grade = self.grade == students[i].grade
            same_class = self.class_group == students[i].class_group and self.grade == students[i].grade

            p = a1 * np.random.power(b1)

            ### Lunch: Off-diagonal boosted with lunchgroups
            if same_lunch:
                p += a2 * np.random.power(b2)

            ### Grade-grade interaction layer
            if same_grade:
                p += a3 * np.random.power(b3)

            ### Class-class interaction layer. Assume no bias/low bias for class-class interactions. No free-time activity
            if same_class:
                p += a4 * np.random.power(b4)
                self.p_vector[students[i]] = p
                continue

            self.p_vector[students[i]] = p * self.bias_vector[students[i]] * students[i].bias_vector[self]

    def get_min_p_vector(self):
        return min(self.p_vector, key=self.p_vector.get)

    def renormalize(self):

        self.bias = self.const_bias + 150 * (math.log10(1 / random.random()))

        normTarget = self.bias

        oldMean = np.mean(list(self.bias_vector.values()))

        correction = normTarget / oldMean
        newVector = {}

        for i in self.bias_vector:
            newVector[i] = self.bias_vector[i] * correction

        self.bias_vector = newVector

    def __str__(self):
        return f"Person {self.id}, is in grade {self.grade} and class {self.class_group}, is a {self.sex} and is {self.age} years old"

    def __repr__(self) -> str:
        return str(self.id) + " ID"

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id


class Interaction:
    def __init__(self, p1, p2, count=1):
        self.p1 = p1
        self.p2 = p2
        self.count = count
        self.p1.add_interaction(self)
        self.p2.add_interaction(self)

    def getp1(self):
        return self.p1

    def getp2(self):
        return self.p2

    def getcount(self):
        return self.count

    def add_to_count(self, count=1) -> None:
        self.count += count

    def __repr__(self):
        return f"{self.p1.id} has interacted {self.count} times with {self.p2.id}"


if __name__ == "__main__":
    p1 = Person(1, "B")
    p2 = Person(1, "B")

    print(p1.id)
    print(p2.id)

    l = [p2, p1]
    print(l)
    print(sorted(l))
