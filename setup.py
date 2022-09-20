import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bioNC",
    version="0.0.1",
    author="Pierre Puchaud",
    author_email="pierre.puchaud@umontreal.ca",
    description="A library for biomechanics based on natural coordinates for forward and inverse approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
