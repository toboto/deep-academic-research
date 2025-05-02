from setuptools import find_packages, setup

setup(
    name="deepsearcher",
    version="0.1.0",
    description="A tool for academic research assistance using RAG technology",
    author="murray",
    packages=find_packages(),
    install_requires=[
        "pymysql",
        "tqdm",
        "requests",
        "urllib3",
        "ruff",
    ],
    python_requires=">=3.10",
)
