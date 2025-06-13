from setuptools import setup, find_packages

setup(
    name='atrial-fibrillation-detection',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for detecting atrial fibrillation using complex-valued ResNet architectures.',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'PyYAML',
        'tqdm',
        'complexnumpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)