#!/usr/bin/env python3


class Building:
    def __init__(self):
        self.build_floor()
        self.build_size()

    def build_floor(self):
        raise NotImplementedError

    def build_size(self):
        raise NotImplementedError

    def __str__(self):
        return f"Floor: {self.floor} | Size: {self.size}"


class House(Building):
    def build_floor(self):
        self.floor = "One"

    def build_size(self):
        self.size = "Big"


class Flat(Building):
    def build_floor(self):
        self.floor = "More than One"

    def build_size(self):
        self.size = "Small"


class ComplexBuilding:
    def __str__(self):
        return f"Floor: {self.floor} | Size: {self.size}"


class ComplexHouse(ComplexBuilding):
    def build_floor(self):
        self.floor = "One"

    def build_size(self):
        self.size = "Big and fancy"


def construct_building(cls):
    building = cls()
    building.build_floor()
    building.build_size()
    return building


def main():
    """
    >>> What is this pattern about?
    It decouples the creation of a complex object and its representation,
    so that the same process can be reused to build objects from the same
    family.
    This is useful when you must separate the specification of an object
    from its actual representation (generally for abstraction).

    >>> What does this example do?
    The first example achieves this by using an abstract base
    class for a building, where the initializer (__init__ method) specifies the
    steps needed, and the concrete subclasses implement these steps.
    """

    house = House()
    print(house)

    flat = Flat()
    print(flat)

    complex_house = construct_building(ComplexHouse)
    print(complex_house)


if __name__ == "__main__":
    main()
