import itertools
import random

from enums import Age_group
from enums import Grade
from layers import Grades, Klasse, Lunchbreak, Recess

class Person:
    newid = itertools.count()
    def __init__(self, grade, class_group):
        self.id = next(Person.newid)
        self.sex = self.get_gender()
        self.state = 'R'
        self.vaccinated = self.get_vaccinated_status()
        self.grade = grade
        self.age = self.get_valid_age(grade)
        self.age_group = self.set_age_group()
        self.class_group = str(class_group)
        self.lunch_group = 1
        self.interactions = []
    
    def get_valid_age(self, grade, is_teacher = False):
        return random.choice([grade + 4, grade + 5])

    def calc_probability_of_interacting(self, person):
        return random.random()

    def set_age_group(self):
        for group in Age_group:
            if self.age in group.value:
                return group
        return None

    def add_interaction(self, interaction):
        if interaction not in self.interactions:
            self.interactions.append(interaction)

    def interacted(self, p):
        return True

    def get_gender(self):
        return "F" if random.random() > .5 else "M"
    def get_vaccinated_status(self):
        return 1 if random.random() > .2 else 0

    def find_all_interactions_for_person(self, p, interactions):
        # interactions_t = filter(lambda x: x.p1 == p or x.p2 == p, interactions) 
        ids = []
        for interaction in self.interactions:
            if interaction.p1 == p:
                yield interaction.p2.id
                # ids.append(interaction.p2.id)
            else:
                yield interaction.p1.id
                # ids.append(interaction.p1.id)
        # return ids
    
    def getID(self):
        return self.id
    
    def getGrade(self):
        return self.grade
    
    def getClass(self):
        return self.class_group
    
    def getLunchgroup(self):
        return self.lunch_group

    def __str__(self):
        return f'Person {self.id}, is in grade {self.grade} and class {self.class_group}, is a {self.sex} and is {self.age} years old' 

    def __repr__(self) -> str:
        return str(self.id)

class Interaction:
    
    def __init__(self, p1, p2, count=1):
        self.p1 = p1
        self.p2 = p2
        self.count = count
        self.p1.add_interaction(self)
        self.p2.add_interaction(self)
    
    def getp1 (self):
        return self.p1

    def getp2 (self):
        return self.p2

    def getcount (self):
        return self.count

    def add_to_count(self, count = 1) -> None:
        self.count += count

    def __repr__(self):
        return f'{self.p1.id} has interacted {self.count} times with {self.p2.id}'


