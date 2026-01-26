from setuptools import setup, find_packages

setup(
    name="lesion_tracking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "nibabel"
    ],
    entry_points={
        'console_scripts': [
            'process_lesions=lesion_tracking.main:process_lesion_masks',
        ],
    },
)
