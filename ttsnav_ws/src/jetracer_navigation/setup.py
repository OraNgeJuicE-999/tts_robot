import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'jetracer_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='itri',
    maintainer_email='itri@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'nav_node = jetracer_navigation.nav_node:main',
            'teleop_keyboard = jetracer_navigation.teleop_keyboard:main',
            'a_star_planner = jetracer_navigation.a_star_planner:main',
            'dwa_controller = jetracer_navigation.dwa_controller:main',
            'goal_bridge = jetracer_navigation.goal_bridge:main',
            'vlm_bridge = jetracer_navigation.vlm_bridge:main',
            'set_initial_pose = jetracer_navigation.set_initial_pose:main'
        ],
    },
)
