from setuptools import find_packages, setup

package_name = 'pddl_type_extractor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kai',
    maintainer_email='dorteel@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "type_extraction_node = pddl_type_extractor.ros_node:main",
            "test_behaviour_1k = pddl_type_extractor.test_behaviour_1k:main",
        ],
    },
)
