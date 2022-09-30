#!/usr/bin/env python3
from importlib_metadata import entry_points
from setuptools import find_packages
from setuptools import setup
import pathlib

import sarrarp50

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(name='sarrarp50-toolkit',
    version='0.0.2',
    description='Tools to preprocess and evaluate the SAR-RARP50 dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/surgical-vision/SAR_RARP50-evaluation',
    author='Dimitris Psychogyios',
    author_email='d.psychogyios@gmail.com',
    packages=find_packages(exclude=('scripts*',)),
    entry_points={
        'console_scripts': [
            'rarptk = scripts.sarrarp50:main'
        ]
    },
    install_requires=[
        'monai>=0.9',
        'numpy',
        'opencv-python>4',
        'pandas',
        'scipy>=1.6',
        'six',
        'torch',
        'tqdm',
        'typing_extensions>=4'
    ],
    python_requires='>=3.8, <4',
    classifiers=[
        'Environment :: Console',
        
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',

        'License :: OSI Approved :: MIT License',

        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='SAR-RARP50, toolkit',
    project_urls={
        'Bug Reports': 'https://github.com/surgical-vision/SAR_RARP50-evaluation',
        'Source': 'https://github.com/surgical-vision/SAR_RARP50-evaluation',
    },
)