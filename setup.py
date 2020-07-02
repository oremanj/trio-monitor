from setuptools import setup, find_packages

exec(open("trio_monitor/_version.py", encoding="utf-8").read())

LONG_DESC = open("README.rst", encoding="utf-8").read()

setup(
    name="trio-monitor",
    version=__version__,
    description="Trio introspection and debugging tools",
    url="https://github.com/oremanj/trio-monitor",
    long_description=LONG_DESC,
    author="Joshua Oreman",
    author_email="oremanj@gmail.com",
    license="MIT -or- Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["trio >= 0.16.0", "trio-typing"],
    keywords=["async", "trio", "debugging"],
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: Apache Software License",
        "Framework :: Trio",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
)
