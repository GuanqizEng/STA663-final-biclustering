import setuptools
from os import path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="STA663-final-biclustering-cxgq", # Replace with your own username
    version="1.0",
    author="Chengxin Yang, Guanqi Zeng",
    author_email="chengxin.yang@duke.edu, guanqi.zeng@duke.edu",
    description='A python package of Biclustering for Sparse matrix',
    url="https://github.com/Cheungsin/STA663-final-biclustering",
    py_modules = ['STA-663-final-biclustering-cxgq'],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"SSVD": "SSVD"},
    packages=setuptools.find_packages(),
    python_requires=">=3",
)
