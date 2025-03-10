from setuptools import setup, find_packages

setup(
    name='toxic-comment-classification',
    version='1.0.0',
    description='Multi-label classification of toxic comments using LLMs',
    long_description='',
    author='hades-init',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'pytorch',
        'transformers',
        'datasets',
        'fastapi',
        'pydantic'
    ]
)