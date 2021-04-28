from setuptools import setup, find_packages
from os import path

setup(
      name='STA663-final-biclustering',
      version=3.6,
      description='A python package of Biclustering for Sparse matrix',
      url='https://github.com/GuanqizEng/STA663-final-biclustering',
      author='Chengxin Yang, Guanqi Zeng',
      author_email='chengxin.yang@duke.edu, guanqi.zeng@duke.edu',
      classifiers=[
                  'Development Status :: 3 - Alpha',
                  'Intended Audience :: Developers',
                  'Topic :: Software Development :: Libraries :: Python Modules',
                  'License :: OSI Approved :: MIT License',
                  'Programming Language :: Python :: 3',
                  'Programming Language :: Python :: 3.4',
                  'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                  ],
      py_modules = ['STA663-final-biclustering'],
      packages=find_packages(),
      scripts = ['run_STA663-final-biclustering.py'],
      data_files=[('my_data', ['`two real world data sets`/bracsample.txt', 
                       '`two real world data sets`/bracsample.txt',
                       '`two real world data sets`/LungCancerData.txt'])],
      python_requires='>=3',
      )