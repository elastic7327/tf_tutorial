#!/bin/sh


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TF_CPP_MIN_LOG_LEVEL=2
# pytest -s -v --log-format="%(asctime)s %(levelname)s %(message)s" --log-date-format="%Y-%m-%d %H:%M:%S"
pytest -s -v src/tests/test_chapter_04.py
# pytest -s -v
