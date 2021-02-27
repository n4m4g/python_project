#!/usr/bin/env python3


def count_to(count):
    numbers = ['one', 'two', 'three', 'four', 'five']
    yield from numbers[:count]

    """
    yield from something

    is equal to

    for thing in something:
        yield thing
    """


def count_to_two():
    return count_to(2)


def count_to_five():
    return count_to(5)


def main():
    """
    Traverses a container and accesses the container's elements.
    """

    for number in count_to_two():
        print(number)

    print()

    for number in count_to_five():
        print(number)


if __name__ == "__main__":
    main()
