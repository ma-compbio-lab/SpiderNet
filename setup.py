from setuptools import setup, find_packages

setup(


    name='spidernet',


    version='0.0.3',


    packages=find_packages(),


    install_requires=[
        "torch==2.0.1+cu117",
        "torchvision==0.15.2+cu117",
        "torchaudio==2.0.2+cu117",
        "torch_scatter==2.1.2",
        "torch-geometric==2.6.1",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "scipy==1.15.3",
        "pandas==2.2.3",
        "scanpy==1.11.1",
        "matplotlib==3.10.3",
        "seaborn==0.13.2",
        "scikit-misc==0.5.1",
        "igraph==0.11.8",
        "louvain==0.8.2",
        "gseapy==1.1.10",
        "squidpy==1.6.5"
    ],


    author='Junjie Tang',


    author_email='junjieta@andrew.cmu.edu',


    description='Interpretable framework for learning cell–cell meta-interactions from spatial omics data',


    long_description=open('README.md').read(),


    long_description_content_type='text/markdown',


    url='https://github.com/junjie-sml/SpiderNet',


    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],


)