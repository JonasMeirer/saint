from setuptools import setup, find_packages

setup(
    name='saint',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],  # Add dependencies here
    entry_points={
        'console_scripts': [
            # 'command_name = module:function'
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
