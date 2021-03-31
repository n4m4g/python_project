#!/usr/bin/env python3


class HideFileCommand:
    def __init__(self):
        self.hidden_files = []

    def execute(self, filename):
        print(f"Hiding {filename}")
        self.hidden_files.append(filename)

    def undo(self):
        filename = self.hidden_files.pop()
        print(f"un-hiding {filename}")


class DeleteFileCommand:
    def __init__(self):
        self.delete_files = []

    def execute(self, filename):
        print(f"Deleting {filename}")
        self.delete_files.append(filename)

    def undo(self):
        filename = self.delete_files.pop()
        print(f"restoring {filename}")


class MenuItem:
    def __init__(self, command):
        self.command = command

    def on_do_press(self, filename):
        self.command.execute(filename)

    def on_undo_press(self):
        self.command.undo()


def main():
    """
    You have a menu that has lots of items.
    Each item is responsible for doing a
    special thing and you want your menu item
    just call the execute method when
    it is pressed.
    To achieve this you implement
    a command object with the execute
    method for each menu item and pass to it.

    *About the example
    We have a menu containing two items.
    Each item accepts a file name, one hides the file
    and the other deletes it.
    Both items have an undo option.
    Each item is a MenuItem class that accepts the
    corresponding command as input and executes
    it's execute method when it is pressed.
    """

    item1 = MenuItem(DeleteFileCommand())
    item2 = MenuItem(HideFileCommand())

    filename = 'test.txt'
    item1.on_do_press(filename)
    item1.on_undo_press()

    item2.on_do_press(filename)
    item2.on_undo_press()


if __name__ == "__main__":
    main()
