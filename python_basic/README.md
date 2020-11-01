python basic
============

Contents
--------
- [variable scope](#variable-scope)
- [class attribute](#class-attribute)
- [decorator](#decorator)

variable scope
--------------

global variables could be read in a function

```
>>> a = 10
>>> def read_var():
...     print(a)
...
>>> read_var()
10
>>>
```

change the global variable in a function, use 'global'

```
>>> a = 10
>>> def write_var():
...     global a
...     a = 20
...
>>> write_var()
>>> print(a)
20
>>>
```

class attribute
---------------

class attribute

```
>>> class Student:
...     count = 0
...     def __init__(self, name):
...         self.name = name
...         Student.count += 1
...
>>> Student.count
0
>>> a = Student('a')
>>> Student.count
1
>>>
```

instance attribute

```
>>> class Student:
...     name='Student'
...
>>> s = Student()
>>> s.name # no instance attribute, here is class attribute
'Student'
>>> Student.name # class attribute
'Student'
>>> s.name = 'John'
>>> s.name # priority of instance attribute higher than priority of class attribute
'John'
>>> Student.name
'Student'
>>> del s.name # delete instance attribute
>>> s.name
'Student'
>>>
```

decorator
---------

Add some feature to function without change it!!

For instance, a timer decorator

```
from time import time
from functools import wraps

def log(func):
    @wraps(func)
    def wrapper(*args, **kw):
        t = time()
        result = func(*args, **kw)
        print(f"{func.__name__}, {time()-t:.4f}")
        return result
    return wrapper

@log
def some_func():
    # do something
```
