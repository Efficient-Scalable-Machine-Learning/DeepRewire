from setuptools import setup, find_packages

setup(
    name='deep_rewire',
    version='0.1.0',
    packages=find_packages(include=['deep_rewire']),
    install_requires=[
        'torch',
    ],
    include_package_data=True,
    description='DeepRewire is a PyTorch-based project designed to simplify the creation and optimization of sparse neural networks with the concepts from the [Deep Rewiring](https://arxiv.org/abs/1711.05136) paper by Bellec et. al. ⚠️ Note: The implementation is not made by any of the authors. Please double-check everything before use.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Lukas König',
    author_email='luggistruggi@gmail.com',
    url='https://github.com/LuggiStruggi/DeepRewire',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

