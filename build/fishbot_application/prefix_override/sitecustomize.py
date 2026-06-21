import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/asus/ros2_ws/chapt6_ws/install/fishbot_application'
