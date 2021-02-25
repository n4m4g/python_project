#!/usr/bin/env python3

class Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class YourBorg(Borg):
    def __init__(self, state=None):
        super().__init__()
        if state:
            self.state = state
        else:
            if not hasattr(self, "state"):
                self.state = "Init"

    def __str__(self):
        return self.state


def main():
    '''
    >>> What is this pattern about?
    a way to implement singleton behavior, but instead of
    having only one instance of a class, there are multiple
    instances that share the same state.
    In other words, the focus is on sharing state
    instead of sharing instance identity.

    >>> What does this example do?
    In Python, instance attributes are stored in a
    attribute dictionary called __dict__.
    Usually, each instance will have its own dictionary,
    but the Borg pattern modifies this so that all
    instances have the same dictionary.

    The __shared_state attribute will be the dictionary
    shared between all instances, and this is ensured by assigining
    __shared_state to the __dict__ variable when initializing a new
    instance (i.e., in the __init__ method).
    Other attributes are usually added to the instance's attribute
    dictionary, but, since the attribute dictionary itself is
    shared (which is __shared_state), all other attributes will also be shared.
    '''

    b1 = YourBorg()
    print(f"b1: {b1}")

    b2 = YourBorg()

    b1.state = 'Idle'
    b2.state = 'Running'

    print(f"b1: {b1}, b2: {b2}")

    b2.state = 'Zombie'
    print(f"b1: {b1}, b2: {b2}")

    print(f"b1 is b2: {b1 is b2}")

    b3 = YourBorg()
    print(f"b1: {b1}, b2: {b2}, b3: {b3}")

    b4 = YourBorg('Running')
    print(f"b1: {b1}, b2: {b2}, b3: {b3}, b4: {b4}")


if __name__ == "__main__":
    main()
