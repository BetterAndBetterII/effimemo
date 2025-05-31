"""
effimemo package setup file
"""

from setuptools import setup, find_packages

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="effimemo",
    version="0.1.1",
    description="A Python package for managing LLM context windows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="betterandbetterii",
    author_email="betterandbetterii@gmail.com",
    url="https://github.com/betterandbetteri/effimemo",
    project_urls={
        "Bug Reports": "https://github.com/betterandbetteri/effimemo/issues",
        "Source": "https://github.com/betterandbetteri/effimemo",
    },
    packages=find_packages(),
    install_requires=[
        "tiktoken>=0.5.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    keywords="llm context management token window ai chatbot",
    license="MIT",
)
