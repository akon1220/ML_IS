# python_code_templates.py: a file outlining code TEST standards for the cross-domain-learning repository
# SEE LICENSE STATEMENT AT THE END OF THE FILE
# See https://docs.python.org/3/library/unittest.html for reference

# dependency import statements
import unittest
from templates.python_code_templates import *


class TestGenericPythonClass(unittest.TestCase):
    """
    TestGenericPythonClass Description:
    - General Purpose: (describe what class is generally useful for; NAME THE CLASS BEING TESTED - python_code_templates.GenericPythonClass,
                       in this case - IN PARTICULAR)
    - Usage:
      * other_file_or_function.foo: (describe how TestGenericPythonClass is used in other_file_or_function.foo)
    """

    def setUp(self):
        """
        TestGenericPythonClass.setUp: (describe what function's general purpose/functionality is)
         - Inputs:
            * arg1 (type): (describe what arg1 represents / what it is for)
         - Outputs:
            * out1 (type): (describe what out1 represents / what it is for)
         - Usage:
            * other_file_or_function.foo: (describe how current function is used in other_file_or_function.foo)
        """
        self.generic_class = GenericPythonClass()
        pass

    def tearDown(self):
        """
        TestGenericPythonClass.tearDown: (describe what function's general purpose/functionality is)
         - Inputs:
            * arg1 (type): (describe what arg1 represents / what it is for)
         - Outputs:
            * out1 (type): (describe what out1 represents / what it is for)
         - Usage:
            * other_file_or_function.foo: (describe how current function is used in other_file_or_function.foo)
        """
        self.generic_class.dispose()
        pass


class TestGenericPythonFunction(unittest.TestCase):
    """
    TestGenericPythonFunction Description:
    - General Purpose: (describe what class is generally useful for; NAME THE FUNCTION BEING TESTED - python_code_templates.generic_python_function,
                       in this case - IN PARTICULAR)
    - Usage:
      * other_file_or_function.foo: (describe how TestGenericPythonFunction is used in other_file_or_function.foo)
    """

    def setUp(self):
        """
        TestGenericPythonClass.setUp: (describe what function's general purpose/functionality is)
         - Inputs:
            * arg1 (type): (describe what arg1 represents / what it is for)
         - Outputs:
            * out1 (type): (describe what out1 represents / what it is for)
         - Usage:
            * other_file_or_function.foo: (describe how current function is used in other_file_or_function.foo)
        """
        pass


if __name__ == "__main__":
    """
    python_code_templates_test.__main__: (describe what function's general purpose/functionality is)
     - Inputs:
        * arg1 (type): (describe what arg1 represents / what it is for)
     - Outputs:
        * out1 (type): (describe what out1 represents / what it is for)
     - Usage:
        * other_file_or_function.foo: (describe how current function is used in other_file_or_function.foo)
    """
    unittest.main()
    pass

