#!/usr/bin/env python3
import json

import rclpy
from rclpy.node import Node

from pddl_type_extractor.srv import GetTypes
from pddl_type_extractor_core.pipeline import derive_types_for_instruction


class PDDLTypeExtractorNode(Node):
    def __init__(self):
        super().__init__("pddl_type_extractor")
        self.service = self.create_service(GetTypes, "get_types", self.handle_get_types)
        self.get_logger().info("PDDL type extractor service ready.")

    def handle_get_types(self, request, response):
        try:
            result = derive_types_for_instruction(request.instruction)

            response.success = True
            response.types = result["types"]

        except Exception as e:
            self.get_logger().error(str(e))

            response.success = False
            response.types = []

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PDDLTypeExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
