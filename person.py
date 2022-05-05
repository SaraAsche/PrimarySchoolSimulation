"""Person class

Describes the Person classes and their functionality. A person
object is either a student or a teacher.  Is meant to be called from main.py,
network.py or analysis.py.  

  Typical usage example:

  person = Person(5, 'A')
  person2 = Person(3, 'B')
  
  person.set_state(Disease_states.IA)
  person.get_interaction(person2)

Author: Sara Johanne Asche
Date: 14.02.2022
File: person.py
"""

import itertools
import random
import math

import numpy as np

from enums import Age_group
from helpers import Helpers
from interaction import Interaction
from enums import Disease_states


class Person:
    """A class to collect the individuals and their attributes in the model

    Longer class information...

    Attributes
    ----------
    id : int
        A unique number attached to the person object ranging from 0 to the max capacity at each school.
    sex : str
        Denotes the sex of the person object. Can be "F" for female or "M" for male
    grade : int
        Number denoting the grade of the person object. Is limited by grades present at the school.
    age : int
        Number denoting the person's age
    age_group : Age_group enum
        Contains an enum describing the age_group of the individual.
    class_group : str
        Describes the exact class a Person object is registrered in. Could for instance be "A" or "B"
    lunch_group : bool
        True if the person object is in a grade lower than 4. Otherwise it is False. Can be changed based on schools and which grades interact most with each other.
    interactions : dictionary of Interaction objects
        Keeps a dictionary of each Interaction objects that occur between one Person object and another. The key is the Person object interacted with whilst the values are the Interaction object between the two Person objects.
    base_bias : float
        Each Person object is initiated with a float value that denotes their bias for interacting with any other Person object.
    bias_grade : float
        Each Person object is initiated with a float value that denotes their bias for interacting with any other Person objects of the same grades.
    bias_class : float
        Each Person object is initiated with a float value that denotes their bias for interacting with any other Person objects of the same class.
    p_vector : dictionary
        Dictionary that keeps track of the possibility that one Person object has to interact with another Person object
    state : str
        Disease spread model attribute.
    vaccinated: bool
        True if person object has been given a vaccine the last 6 months. False otherwise
    tested : bool
        True if person has tested positive for the disease
    states : dict
        Dict of the states an individual can be in, with states as the keys and number of days as values
    cohort : str
        String that determines the cohort an individual belongs in
    infected_on_day : int
        Integer that denotes the day an individual was infected
    recovered_on_day : int
        Integer that denotes the day an individual was recovered
    infected_by : Person
        The Person object that infected this individual

    Methods
    -------
    get_gender()
        Returns a string value representing a Person object's sex that is either "F" or "M" with a 50/50 percentage.
    get_ID()
        Returns a Person object's ID
    get_grade():
        Returns a Person object's Grade (i.e 5 or 1)
    get_class():
        Returns a Person Object's Class (i.e "A" or "B")
    get_class_and_grade():
        Returns a string consisting of the grade plus the class (i.e "5A" or "1B")
    get_lunch_group():
        Returns the lunch_group attribute of a Person object
    get_interaction( p)
        Returns the Interaction object of an interaction between Person object (self) and Person object (p).
    get_min_p_vector()
        Returns the lowest p value in a Person (self)'s p_vector dictionary.
    generate_valid_age(grade, is_teacher=False)
        Returns the age of an individual that is in grade grade, 50% likely that they have had their bithday before the school starts, 50% they have had their birthday after. If object is teacher, another method is used.
    generate_lunch_group()
        Generates the lunch_group attribute to a boolean depending on which grade the Person object is in.
    generate_age_group()
        Generates the age_group attribute to an Age_group enum value depending on the age of the Person object.
    generate_bias()
        Generated base_bias, bias_grade and bias_class
    add_interaction(self, interaction)
        Attaches an Interaction object to the Person object's interaction dictionary attribute.
    has_interacted_with(self, p)
        Returns True if Person object (p) is in Person object (self)'s interaction dictionary
    generate_p_vector(self, students, X)
        Generates a p_vector dictionary where keys are all possible Person objects Person (self) can interact with and the key is probability p that they will interact. Uses the similarity between two Person objects to generate a probability p.
    renormalize(self)
        Updates the bias_vector of a Person object (self) where the base_bias is affected by an additional random bias and then normalised to avoid a strictly growing bias.
    get_state()
        Returns the state of the Person object
    set_state(state)
        Sets the state of an individual to state
    clean_states()
        Sets the states dictionary to empty
    set_tested(tested: bool):
        Sets wheter or not the person has tested positive for the disease
    get_tested()
        Returns weter or not the person has tested positive for the disease
    get_vaccinated_status(self)
        Returns True if an individual has gotten a vaccine dose less than 6 months from now.
    set_cohort(cohort)
        Sets the cohort attribute of an individual
    get_cohort()
        Gets the cohort attribute of an individual
    set_diasease_state(state : Disease_states)
        Sets the disease state to state
    set_day_infected(day)
        Sets the day the individual was infected on
    get_day_infected()
        Gets the day the individual was infected on
    set_day_recovered(day)
        Sets the day the individual was recovered on
    get_day_recovered()
        Gets the day the individual was recovered on
    set_infected_by(pers)
        Sets the person pers this individual was infected by
    get_infected_by()
        Gets the person pers this individual was infected by
    is_symptomatic()
        Returns whether or not the person has a symptomatic or asymptomatic disease course
    add_day_in_state(self, pA=0.4, 0.6)
        For each day, the it will update the disease state of the person
    disease_state_start()
        Resets the states, cohort, infected_on_day, recovered_on_day, infected_by to 0 or None

    """

    newid = itertools.count()

    def __init__(self, grade, class_group, ID=None):
        """Inits Person object with grade and class parameters

        Parameters
        ----------
        grade : int
            The grade of the Person
        class_group : str
            the class of the Person
        ID : int/None
            If the person already has an ID, it should be input. Default is None, and the ID is generated
        """
        if ID != None:
            self.id = ID
        else:
            self.id = next(Person.newid)

        self.sex = self.get_gender()

        self.grade = grade
        self.age = self.generate_valid_age(grade)
        self.age_group = self.generate_age_group()
        self.class_group = str(class_group)
        self.lunch_group = self.generate_lunch_group()
        self.interactions = {}

        self.generate_bias()

        self.p_vector = {}

        self.state = self.disease_state_start()
        self.vaccinated = self.get_vaccinated_status()
        self.tested = False
        self.states = dict([(e, 0) for e in Disease_states])
        self.cohort = None
        self.infected_on_day = None
        self.recovered_on_day = None
        self.infected_by = -1

    def get_gender(self) -> str:  # Returns gender of Person
        return "F" if random.random() > 0.5 else "M"

    def get_ID(self) -> int:
        return self.id

    def get_grade(self) -> int:
        return self.grade

    def get_class(self) -> str:
        return self.class_group

    def get_class_and_grade(self) -> str:
        return str(self.grade) + self.class_group

    def get_lunch_group(self) -> bool:
        return self.lunch_group

    def get_interaction(self, p) -> Interaction:  # Returns interaction object between self and p
        return self.interactions.get(p, Interaction(self, p, 0))

    def remove_interactions(self):
        self.interactions.clear()

    def get_min_p_vector(self) -> float:
        """Returns minimum value of a given p-vector"""
        return min(self.p_vector, key=self.p_vector.get)

    def generate_valid_age(self, grade: int) -> int:

        """Generates and returns the age of the Person

        Uses grade to generate an appropriate age. Takes
        into account that students of the same grade can be
        +-1 year appart depending on when they are born.
        If the argument is_teacher isn't passed in, the default
        is_teacher = False is used.

        Parameters
        ----------
        grade : int
            The grade the Person object is in
        """

        return random.choice([grade + 4, grade + 5])

    def generate_lunch_group(self) -> bool:
        """Generates an appropriate bool value for lunch_group according to grade"""

        if self.grade < 4:
            return 1
        else:
            return 0

    def generate_age_group(self) -> Age_group:
        """Returns an age_group based on the Person objects' age"""

        for group in Age_group:
            if self.age in group.value:
                return group
        return None

    def generate_bias(self):
        # TODO: add documentation
        self.base_bias = 20 * (math.log10(1 / random.random()))
        self.bias_grade = 17 * (math.log10(1 / random.random()))
        self.bias_class = np.random.normal(loc=100, scale=5)

    def add_interaction(self, interaction: Interaction):
        """Adds an Interaction object to a Person's interactions dictionary

        Checks which of the p1 or p2 Person objects is self and denotes the
        other one other. Adds an entry to Person (self)'s dictionary with
        other as key and the Interaction object as value.

        Parameters
        ----------
        interaction : Interaction
            Interaction object that has two individuals saved
        """

        p1 = interaction.p1
        p2 = interaction.p2

        other = p1 if p1 != self else p2

        self.interactions[other] = interaction

    def has_interacted_with(self, p) -> bool:  # Returns True if two individuals have interacted. False otherwise.
        return p in self.interactions

    def generate_p_vector(self, students: list, X=[]) -> None:
        """Generates a p-vector dictionary for a Person object with respect to all other students

        If no X is given, the length of X is 0 and this function returns a p-vector with the given
        parameters: a1 = , a2 = a3 = a4 =, b1 = b2 = b3 = , b4 = . If an X is given, the parameters
        listed in it will be used.

        Parameters
        ----------
        students : list
            A list that contains all the Person objects that attend a certain school
        X : list
            A list containing parameters for a1,a2,a3,a4 and b1,b2,b3,b4.
        """

        if len(X):
            ## Off-diagonal excluding lunch
            a1 = X[0]
            b1 = X[1]

            ## Off-diagonal with lunch
            a2 = X[2]
            b2 = X[3]

            ## Grade-Grade
            a3 = X[4]
            b3 = X[5]

            ## Class-Class
            a4 = X[6]
            b4 = X[7]
        else:
            ## Off-diagonal excluding lunch
            a1 = 2  # 1.5
            b1 = 0.04  # 0.1

            ##  Off-diagonal with lunch
            a2 = 0.01  # 0.001
            b2 = 0.01  # 0.07

            ## Grade-Grade
            a3 = 2  # 100
            b3 = 0.4  # 0.3

            ## Class-Class
            a4 = 0.2  # 10000
            b4 = 11.5  # 1

        for i in range(len(students)):
            same_lunch = self.lunch_group == students[i].lunch_group
            same_grade = self.grade == students[i].grade
            same_class = self.class_group == students[i].class_group and self.grade == students[i].grade

            ## Default level of interaction between students
            p = a1 * np.random.power(b1) * self.base_bias * students[i].base_bias

            ### Lunch: Off-diagonal boosted with lunchgroups
            if same_lunch:
                p += a2 * np.random.power(b2) * self.base_bias * students[i].base_bias

            ### Grade-grade interaction layer
            if same_grade:
                p += a3 * np.random.power(b3) * self.bias_grade * students[i].bias_grade

            ### Class-class interaction layer. Assume no bias/low bias for class-class interactions. No free-time activity
            if same_class:
                p += np.random.gamma(a4, b4) * min(self.bias_class, students[i].bias_class)

            self.p_vector[students[i]] = p

    def renormalize(self) -> None:
        # TODO: remove renormalize
        """Renormalises the bias_vector of a Person object

        At the end of each day the bias_vector is updated so that students form connections differently
        from one day to the next. This is done by adding the new value, bias, to the const bias, and shifting it.
        The bias_vector is then multiplied by a correction value that is the old bias divided by the new bias vector
        to form a new value for each interaction.

        """

        self.bias = self.base_bias + 160 * (math.log10(1 / random.random()))

        normTarget = self.bias

        oldMean = np.mean(list(self.bias_vector.values()))

        correction = normTarget / oldMean
        newVector = {}

        for i in self.bias_vector:
            newVector[i] = self.bias_vector[i] * correction

        self.bias_vector = newVector.copy()

    def get_state(self) -> str:
        """Returns the state of the Person object"""
        return self.state

    def set_state(self, state: Disease_states) -> None:
        """Sets the state of the Person object"""
        self.state = state
        # self.states[state] += 1

    def clean_states(self) -> None:
        """Sets the states dictionary to empty"""
        self.state = None

    def set_tested(self, tested: bool) -> None:
        """Sets wheter or not the person has tested positive for the disease

        Parameters
        ----------
        tested : bool
            Denotes if the person has tested positive for the disease
        """
        self.tested = tested

    def get_tested(self) -> bool:
        """Returns weter or not the person has tested positive for the disease"""
        return self.tested

    def get_vaccinated_status(self) -> bool:
        """Returns True if an individual has gotten a vaccine dose less than 6 months from now"""

        return 1 if random.random() > 0.2 else 0

    def set_cohort(self, cohort: str) -> None:
        """Sets the cohort attribute of an individual"""
        self.cohort = cohort

    def get_cohort(self) -> str:
        """Gets the cohort attribute of an individual"""
        return self.cohort

    def set_diasease_state(self, state) -> None:
        """Sets the disease state to state"""
        self.state = state

    def set_day_infected(self, day):
        """Sets the day the individual was infected on"""
        self.infected_on_day = day

    def get_day_infected(self):
        """Gets the day the individual was infected on"""
        return self.infected_on_day

    def set_day_recovered(self, day):
        """Sets the day the individual was recovered on"""
        self.recovered_on_day = day

    def get_day_recovered(self):
        """Gets the day the individual was recovered on"""
        return self.recovered_on_day

    def set_infected_by(self, pers):
        """Sets the person pers this individual was infected by"""
        self.infected_by = pers

    def get_infected_by(self):
        """Gets the person pers this individual was infected by"""
        return self.infected_by

    def is_symptomatic(self) -> bool:
        """Returns whether or not the person has a symptomatic or asymptomatic disease course"""
        return bool(self.states[Disease_states.IS])

    def add_day_in_state(self, pA=0.4, pP=0.6):  # FHI: pA = 0.4
        """For each day, the add_day_in_state will update the disease state of the person. A day wil also be added to the states dict

        Parameters
        ----------
        pA : float
            Describes the percentage of individuals having a asymptomatic disease course
        pP : float
             Describes the percentage of individuals having a presymptomatic disease course
        """

        if self.state == Disease_states.E and self.states[self.state] == 3:
            # print("Hello")
            s = Helpers.get_infection_root(pA, pP)
            self.set_state(s)
        elif self.state == Disease_states.IP and self.states[self.state] == 3:
            self.set_state(Disease_states.IS)
        elif self.state == Disease_states.IAS and self.states[self.state] == 6:  # 4
            self.set_state(Disease_states.R)
            self.set_day_recovered(self.infected_on_day + 9)
        elif self.state == Disease_states.IS and self.states[self.state] == 4:
            self.set_state(Disease_states.R)
            self.set_day_recovered(self.infected_on_day + 10)
            self.tested = False
        else:
            self.states[self.state] += 1

        self.remove_interactions()

    def disease_state_start(self) -> str:
        """Resets the states, cohort, infected_on_day, recovered_on_day, infected_by to 0 or None"""
        self.states = dict([(e, 0) for e in Disease_states])
        self.cohort = None
        self.infected_on_day = None
        self.recovered_on_day = None
        self.infected_by = -1

        return Disease_states.S

    def __str__(self):
        return (
            f"Person ID: {self.id} (grade: {self.grade}, class: {self.class_group}, sex: {self.sex}, age: {self.age})"
        )

    def __repr__(self) -> str:
        return str(self.id) + " ID"

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id
