# setup.py
from setuptools import setup, find_packages

setup(
    name="soccerlib",  # Имя вашего пакета
    version="0.1.0",  # Версия
    packages=find_packages(),  # Поиск всех пакетов
    install_requires=[],  # Зависимости, если есть
    author="Alexey Kozhakin",
    author_email="alexeykozhakin@gmail.com",
    description="Library for football predictions",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/AlexeyKozhakin/soccerlib.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
