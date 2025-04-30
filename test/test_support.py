import unittest
from pathlib import Path

from loguru import logger


def root_path() -> Path:
    return Path(__file__).parent.parent


class BaseTest(unittest.TestCase):

    def setUp(self):
        logger.patch(
            lambda record: record.update(level=logger.level("DEBUG"))
            if record["name"] == "sqlite3" else record
        )
