#!/usr/bin/env python3


class Dog:
    def __init__(self):
        self.name = "Dog"

    def bark(self):
        return "woof!"


class Cat:
    def __init__(self):
        self.name = "Cat"

    def meow(self):
        return "meow!"


class Human:
    def __init__(self):
        self.name = "Human"

    def speak(self):
        return "Hello!"


class Car:
    def __init__(self):
        self.name = "Car"

    def make_noise(self, level):
        return f"vroom{'!'*level}"


class Adapter:
    def __init__(self, obj, **adapted_methods):
        self.obj = obj
        self.__dict__.update(adapted_methods)

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def original_dict(self):
        return self.obj.__dict__


def main():
    """
    >>> What is this pattern about?
    The Adapter pattern is useful to integrate classes that couldn't be
    integrated due to their incompatible interfaces.

    >>> What does this example do?
    The example has classes that represent entities (Dog, Cat, Human, Car)
    that make different noises. The Adapter class provides a different
    interface to the original methods that make such noises. So the
    original interfaces (e.g., bark and meow) are available under a
    different name: make_noise.
    """

    objects = []
    dog = Dog()
    print(dog.__dict__)
    objects.append(Adapter(dog, make_noise=dog.bark))
    print(objects[0].__dict__['obj'], objects[0].__dict__['make_noise'])
    print(objects[0].original_dict())

    cat = Cat()
    objects.append(Adapter(cat, make_noise=cat.meow))

    human = Human()
    objects.append(Adapter(human, make_noise=human.speak))

    car = Car()
    objects.append(Adapter(car, make_noise=lambda: car.make_noise(3)))

    for obj in objects:
        print(f"A {obj.name} goes {obj.make_noise()}")


if __name__ == "__main__":
    main()
