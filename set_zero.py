import sys


def set_zero():
    
    ########
    import rclpy
    import time
    
    from univla.genie_sim_ros import SimROSNode

    rclpy.init()
    sim_ros_node = SimROSNode()

    rclpy.spin_once(sim_ros_node)

    action = [
        -1.105, 0.534, 0.455, -1.247, 0.589, 1.426, -0.078, 
        0.0, 
        1.074, -0.611, -0.280, 1.284, -0.730, -1.495, 0.188, 
        0.0, 
        ]
    
    sim_ros_node.publish_joint_command(action)

if __name__ == "__main__":

    set_zero()
