import random
import itertools
import enum

from person import Person, Interaction
from enums import Grade, Age_group
from layers import Grades, Klasse, Lunchbreak, Recess

school =[] 
def weightedFlip(p):
    return random.random() < p

def interaction_between_persons(p1, p2):
    if p1.getGrade() == p2.getGrade() and (p1.getClass()==p2.getClass()):
        if random.random()<0.85:
           return True
    elif p1.getGrade()==p2.getGrade():
        if random.random()<0.60:
            return True
    else: 
        False # interaksjon, returnerer true/false. Henter inn layers og bruker sansynlighet til å si om en interaksjon finner sted eller ikke

available_grades = []


def generate_network(num_students, num_grades, num_classes, class_treshold = 20):
    available_grades = [i for i in range(1, num_grades + 1)]
    available_classes = [chr(i) for i in range(97, 97 + num_classes)]

    print(available_classes)

    students = []
    
    for grade in available_grades: # Loop igjennom antall grades
        i = 0
        has_filled = False # Bare et flagg for å sjekke om vi har gjort en random fylling av klasser eller ikke
        for pers in range(1, num_students//num_grades + 1): # Loop igjennom antall personer pr grade
                students.append(Person(grade, available_classes[i])) # Legg til person i students
                if pers % (num_students//num_grades//num_classes) == 0: # Dersom vi er kommet oss til ny klasse innenfor grade, må vi oppdatere i
                    i += 1
                    if i >= len(available_classes) and num_students//num_grades - pers >= class_treshold: 
                        # Dersom det ikke går å ha  like mange i hver klasse, og vi har igjen class_treshold antall studenter, lager vi en ny klasse
                        available_classes.append(chr(ord(available_classes[-1]) + 1))
                    elif i >= len(available_classes): # Hvis vi ikke har fler enn class_threshold studenter igjen legger vi de i en random klasse av de vi allerede har
                        has_filled = True # "Si ifra" til loopen at vi har gjort en random fylling av studentente.
                        for _ in range(num_students//num_grades - pers):
                            students.append(Person(grade, random.choice(available_classes))) # Legg til studenter i randome klasser
                if has_filled: # Break dersom vi har fylt random
                    break
    
    interactions = []

    for stud in students:
        print(stud)
        for pers in students:
            if pers.getID in stud.find_all_interactions_for_person(pers, interactions):
                break
            elif interaction_between_persons(stud, pers):
                Interaction(stud, pers)
    print(interactions)


netw = generate_network(225, 5, 2)
