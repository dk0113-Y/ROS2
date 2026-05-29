from setuptools import find_packages, setup

package_name = 'drl_explore_bridge'

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
    maintainer='dk',
    maintainer_email='2731967162@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'policy_probe_node = drl_explore_bridge.policy_probe_node:main',
            'scan_to_local_snap = drl_explore_bridge.scan_to_local_snap_node:main',
            'drl_policy_probe = drl_explore_bridge.drl_policy_probe_node:main',
            'drl_policy_step_once_node = drl_explore_bridge.drl_policy_step_once_node:main',
            'drl_policy_multi_step_node = drl_explore_bridge.drl_policy_multi_step_node:main',
            'drl_standalone_gazebo_bridge_node = drl_explore_bridge.drl_standalone_gazebo_bridge_node:main',
            'drl_trajectory_replay_node = drl_explore_bridge.drl_trajectory_replay_node:main',
        ],
    },
)
