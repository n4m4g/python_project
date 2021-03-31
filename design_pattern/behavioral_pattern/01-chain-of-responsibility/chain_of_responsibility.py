#!/usr/bin/env python3

from abc import ABC, abstractmethod


class Handler(ABC):
    def __init__(self, successor=None):
        self.successor = successor

    def handle(self, request: int):
        res = self.check_range(request)

        if not res and self.successor:
            self.successor.handle(request)

    @abstractmethod
    def check_range(self, request: int):
        raise NotImplementedError("You must implement this")


class ConcreteHandler0(Handler):
    start, end = 0, 10

    def check_range(self, request: int):
        if self.start <= request < self.end:
            print(f"request {request} handled in handler 0")
            return True


class ConcreteHandler1(Handler):
    start, end = 10, 20

    def check_range(self, request: int):
        if self.start <= request < self.end:
            print(f"request {request} handled in handler 1")
            return True


class ConcreteHandler2(Handler):
    start, end = 20, 30

    def check_range(self, request: int):
        if self.start <= request < self.end:
            print(f"request {request} handled in handler 2")
            return True


class FallbackHandler(Handler):
    def check_range(self, request: int):
        print(f"End of chain, no handler for {request}")


def main():
    """
    >>> What is this pattern about?
    The Chain of responsibility is an object oriented version of the
    `if ... elif ... elif ... else ...` idiom, with the
    benefit that the condition–action blocks can be dynamically rearranged
    and reconfigured at runtime.

    This pattern aims to decouple the senders of a request from its
    receivers by allowing request to move through chained
    receivers until it is handled.

    Request receiver in simple form keeps a reference to a single successor.
    As a variation some receivers may be capable of sending requests out
    in several directions, forming a `tree of responsibility`.

    >>> TL;DR
    Allow a request to pass down a chain of receivers until it is handled.
    """

    h0 = ConcreteHandler0()
    h1 = ConcreteHandler1()
    h2 = ConcreteHandler2()
    h0.successor = h1
    h1.successor = h2
    h2.successor = FallbackHandler()

    requests = [4, 2, 14, 24, 35, 24, 18, 7]

    for request in requests:
        h0.handle(request)


if __name__ == "__main__":
    main()
