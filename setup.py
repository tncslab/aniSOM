# from setuptools import setup

# setup( name='anisom',
#     version='0.1.0',
#     packages=['anisom'],
#     install_requires=[
#         'numpy',
#         'torch',
#         'tqdm',
#     ],
# )

from setuptools import setup, find_packages

setup(
    name='anisom',
    version='0.1.0',
    description='Anisotrope Self-Organizing Map and data generation',
    author='Zsigmond Benko',
    author_email='benko.zsigmond@wigner.hu',
    package_dir={'': 'src'},
    packages=find_packages(where='anisom'),
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'matplotlib',
        'scikit-learn',
        'pandas',
    ],
    python_requires='>=3.11',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)