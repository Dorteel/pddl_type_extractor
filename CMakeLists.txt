cmake_minimum_required(VERSION 3.8)
project(pddl_type_extractor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetTypes.srv"
)

ament_python_install_package(pddl_type_extractor_core)

install(PROGRAMS
  pddl_type_extractor_core/ros_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  config.yaml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()