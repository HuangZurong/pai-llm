import unittest
from pydantic import validate_call


@validate_call
def add(a: int, b: int):
    return a + b


class TestPyDantic(unittest.TestCase):

    def test_add(self):
        ret = add(0, 2)
        self.assertEqual(ret, 2)
