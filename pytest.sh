#!/bin/sh


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TF_CPP_MIN_LOG_LEVEL=2
# pytest -s -v --log-format="%(asctime)s %(levelname)s %(message)s" --log-date-format="%Y-%m-%d %H:%M:%S"
# pytest -s -v src/tests/test_chapter_04.py

autopep8 . --recursive --in-place --pep8-passes 2000 --verbose
pytest -s -v src/tests/deep_learning_02.py
# pytest -s -v
