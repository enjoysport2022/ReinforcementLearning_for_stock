# -*- coding: utf-8 -*-
from distutils.core import setup
from setuptools import find_packages

setup(name="ReinforcementLearning_for_stock",
        version="0.0.1",
        description="rl for stock",
        url='https://github.com/poteman/ReinforcementLearning_for_stock',
        install_requires=[
            'lightgbm',
            'xgboost',
            'torch',
            'numpy',
            'pandas',
            'sklearn',
            'tqdm',
            '4paradigm/AutoX'
        ],
        python_requires='>=3.6',
        packages=find_packages(exclude=['data', 'img', 'log']),
        include_package_data=True,
        zip_safe=False
        )
# python setup.py sdist build
# twine upload dist/*