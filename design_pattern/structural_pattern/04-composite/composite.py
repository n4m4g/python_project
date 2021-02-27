#!/usr/bin/env python3

# all classes of inherited implementations must have certain methods
from abc import ABC, abstractmethod


class Graphic(ABC):
    @abstractmethod
    def render(self):
        raise NotImplementedError("You must implement this")


class CompositeGraphic(Graphic):
    def __init__(self):
        self.graphics = []

    def render(self):
        for graphic in self.graphics:
            graphic.render()

    def add(self, graphic):
        self.graphics.append(graphic)

    def remove(self, graphic):
        self.graphics.remove(graphic)


class Ellipse(Graphic):
    def __init__(self, name):
        self.name = name

    def render(self):
        print(f"Ellipse: {self.name}")


def main():
    """
    >>> What is this pattern about?
    The composite pattern describes a group of objects that is treated the
    same way as a single instance of the same type of object. The intent of
    a composite is to "compose" objects into tree structures to represent
    part-whole hierarchies. Implementing the composite pattern lets clients
    treat individual objects and compositions uniformly.
    >>> What does this example do?
    The example implements a graphic class，which can be either an ellipse
    or a composition of several graphics. Every graphic can be printed.
    """

    e1 = Ellipse('1')
    e2 = Ellipse('2')
    e3 = Ellipse('3')

    g1 = CompositeGraphic()
    g2 = CompositeGraphic()

    g1.add(e1)
    g1.add(e2)
    g2.add(e3)

    g = CompositeGraphic()
    g.add(g1)
    g.add(g2)

    g.render()


if __name__ == "__main__":
    main()
