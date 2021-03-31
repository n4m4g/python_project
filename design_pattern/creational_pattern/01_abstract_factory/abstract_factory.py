#!/usr/bin/env python3

import random
from typing import Type


class Pet:
    def __init__(self, name: str) -> None:
        self.name = name

    def speak(self) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


class Cat(Pet):
    def speak(self) -> None:
        print('meow')

    def __str__(self) -> str:
        return f"Cat<{self.name}>"


class Dog(Pet):
    def speak(self) -> None:
        print('woof')

    def __str__(self) -> str:
        return f"Dog<{self.name}>"


class PetShop:
    def __init__(self, animal_factory: Type[Pet]) -> None:
        '''
        animal_factory
            a class object but not instantiate yet
        '''

        self.pet_factory = animal_factory

    def buy_pet(self, name: str) -> Pet:
        pet = self.pet_factory(name)
        print(f"Here is your lovely {pet}")
        return pet


def random_animal(name: str) -> Pet:
    return random.choice([Cat, Dog])(name)


def main():
    '''
    >>> What is this pattern about?

    Provide an interface for creating related/dependent objects
    without need to specify their actual class.


    >>> What does this example do?

    This particular implementation abstracts the creation of a pet and
    does so depending on the factory we chose (Dog or Cat, or random_animal)
    This works because both Dog/Cat and random_animal respect a common
    interface (callable for creation and .speak()).
    Now my application can create pets abstractly and decide later,
    based on my own criteria, dogs over cats.
    '''

    cat_shop = PetShop(Cat)
    pet = cat_shop.buy_pet("Lucy")
    pet.speak()
    print()

    shop = PetShop(random_animal)
    for name in ["Max", "Jack", "Buddy"]:
        pet = shop.buy_pet(name)
        pet.speak()
        print()


if __name__ == "__main__":
    main()
