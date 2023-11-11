# coding: utf8
from setuptools import setup  # type: ignore

setup(
    name='nicere',
    version='0.1',
    packages=['nicere'],
    url='https://github.com/mingminyu/nicere',
    license='Apache 2.0',
    author='mingminyu',
    author_email='yu_mingm623@163.com',
    description='更强大的Python正则匹配器',
    install_requires=[
        "rich",
        "pydantic",
        "pyyaml"
    ],
)
