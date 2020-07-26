from setuptools import setup
setup(
    name=”junix”,
    version=”0.1.1",
    author=”Combined Intelligence”,
    description=”Simple library to export images from Jupyter notebook”,
    packages=[“junix”],
    entry_points={
        “console_scripts”: [‘junix = junix.cli:main’]
    }
)