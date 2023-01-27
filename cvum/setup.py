import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cvum", # Replace with your own username
    version="0.0.1",
    author="Nicolas Rondan",
    author_email="nrondan@correo.um.edu.uy",
    description="Python package for Computer Vision course Universidad de Montevideo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicolasrondan/cv-um-2021",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)