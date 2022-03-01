"""Interaction classes

Describes the Interaction classes and its functionality. An Interaction object
is meant to be called from main.py,
network.py or analysis.py.  

  Typical usage example:

  person = Person(5, 'A')
  person2 = Person(3, 'B')

  interaction = Interaction(person, person2)
  interaction.count --> 1
  interaction.add_to_count(10)
  interaction.count --> 11

Author: Sara Johanne Asche
Date: 14.02.2022
File: person.py
"""


class Interaction:
    """A class that saves interactions between two Person objects

    Attributes
    ----------
    p1 : Person
        Person object that is a part of the interaction
    p2 : Person
        Another Person object that is a part of the interaction
    count : int
        A count that keeps track of how many times two individuals have interacted

    Methods
    -------
    get_p1(self)
        Returns the p1 Person object
    get_p2(self)
        Returns the p2 Person object
    get_count(self)
        Returns the count variable for the Interaction object
    add_to_count(self, count=1)
        Increases the count variable with count (set to default to 1)
    """

    def __init__(self, p1, p2, count=1):
        """Inits Interaction object two Person objects (p1 and p2) and a count.

        Parameters
        ----------
        p1 : Person
            Person object that interacts with p2
        p2 : Person
            Person object that interacts with p1
        count : int
            The number of times p1 and p2 has interacted. Default = 1
        """

        self.p1 = p1
        self.p2 = p2
        self.count = count
        self.p1.add_interaction(self)
        self.p2.add_interaction(self)

    def get_p1(self):
        return self.p1

    def get_p2(self):
        return self.p2

    def get_count(self) -> int:
        return self.count

    def add_to_count(self, count=1) -> None:
        self.count += count

    def __repr__(self):
        return f"{self.p1.id} has interacted {self.count} times with {self.p2.id}"
