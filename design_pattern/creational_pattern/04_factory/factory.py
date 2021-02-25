#!/usr/bin/env python3

class GreekLocalizer:
    def __init__(self):
        self.translations = {"dog": "σκύλος", "cat": "γάτα"}

    def localize(self, msg):
        return self.translations.get(msg, msg)


class EnglishLocalizer:
    def localize(self, msg):
        return msg


def get_localizer(language='English'):
    localizers = {'English': EnglishLocalizer,
                  'Greek': GreekLocalizer}

    return localizers[language]()


def main():
    """
    >>> What is this pattern about?
    Creates objects without having to specify the exact class.

    >>> What does this example do?
    The code shows a way to localize words in two languages: English and
    Greek. "get_localizer" is the factory function that constructs a
    localizer depending on the language chosen.
    """

    e, g = get_localizer('English'), get_localizer('Greek')

    for msg in "dog parrot cat bear".split(' '):
        print(e.localize(msg), g.localize(msg))


if __name__ == "__main__":
    main()
