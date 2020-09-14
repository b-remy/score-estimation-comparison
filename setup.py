from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

# taken from https://github.com/CEA-COSMIC/ModOpt/blob/master/setup.py
with open('requirements.txt') as open_file:
    install_requires = open_file.read()

setup(name='neural-score-estimation-comparison',
      version='0.1',
      description='Comparison of different methods estimating the score of distributions',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/b-remy/score-estimation-comparison',
      author='Benjamin Remy',
      author_email='benjamin.remy@cea.fr',
      license='MIT',
      packages=['nsec'],
      install_requires=install_requires,
      zip_safe=False)
