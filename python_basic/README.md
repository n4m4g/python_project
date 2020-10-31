python basic
============

Contents
--------
- [variable scope](#variable-scope)

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
