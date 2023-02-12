from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'Database tools for ppg and abp signal processing.'
LONG_DESCRIPTION = 'Extract and process photoplethysmography and arterial blood pressure data from mimic3-waveforms and vitaldb.'

setup(
    name="database_tools",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Cameron Johnson",
    author_email="cjohnson23@gwu.edu",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords=[],
    classifiers= [
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
