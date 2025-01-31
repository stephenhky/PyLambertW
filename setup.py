
from setuptools import setup


def package_description():
    return open('README.md', 'r').read()


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


setup(name='pylambertw',
      version='0.0.2a1',
      description="LambertW x Gaussian",
      long_description=package_description(),
      long_description_content_type='text/markdown',
      classifiers=[
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research"
      ],
      keywords="Gaussian, normal distribution, heavy tails",
      url="https://github.com/stephenhky/PyLamertW",
      author="Kwan-Yuet Ho",
      author_email="kwan-yuet.ho@nih.gov",
      license='MIT',
      packages=['pylambertw',],
      package_dir={'pylambertw': 'pylambertw'},
      python_requires='>=3.6',
      install_requires=install_requirements(),
      scripts=['script/simulate_heavytails'],
      test_suite="test",
      zip_safe=False)
