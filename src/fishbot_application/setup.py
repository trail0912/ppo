from setuptools import find_packages, setup

package_name = 'fishbot_application'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wheeltec',
    maintainer_email='wheeltec@todo.todo',
    description='TODO: Package description',
    license='Apache_2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "init_robot_pose=fishbot_application.init_robot_pose:main",
            "get_robot_pose=fishbot_application.get_robot_pose:main",
            "nav_to_pose=fishbot_application.nav_to_pose:main",
            "lstm_train=fishbot_application.train:lstm_main",
            "ppo_train=fishbot_application.train:ppo_main",
            "bilstm_train=fishbot_application.train:bilstm_main",
            "predict=fishbot_application.predict:main",
            "passive_joint_pub=fishbot_application.passive_joint_pub:main",
            "odom_tf_broadcaster=fishbot_application.odom_tf_broadcaster:main",
            "lstm_train1=fishbot_application.train1:main",
            "lstm_train2=fishbot_application.train2:main",
        ],
    },
)
