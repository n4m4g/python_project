# pytest

## package install

   ```
   $ pip3 install --user pytest
   ```

## How to use pytest

1. Create some function in func.py  

2. Create test file with 'test_' prefix,  
   e.g., 'test_func.py' to test file 'func.py'

3. Create function with 'test_' prefix  
   e.g., 'test_my_add' to test function 'my_add'

4. Run pytest
```
$ py.test -v
$ pytest [test_file_name] -v # pytest test_func.py -v
$ pytest [test_file_name]::[test_func_name] -v # pytest test_func.py::test_my_add -v
```
