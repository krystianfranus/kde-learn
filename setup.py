import setuptools

with open('VERSION', 'r') as f:
    version = f.read().strip()

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = f.read().split('\n')

setuptools.setup(
    name='kdelearn',
    version=version,
    author='Krystian Franus',
    author_email='krystian.franus@gmail.com',
    description='Short description of ml methods based on kernel density estimation',
    long_description=long_description,
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.6',
)
