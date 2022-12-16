import sys
import pathlib
root_dir = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from math import cos, sin, tan, pi

import matplotlib.pyplot as plt
import numpy as np

from utils.angle import rot_mat_2d

R  = 1.74   # robot raduis
WD = 1.07*2 # wheel distance
MAX_STEER = pi/4  # [rad] maximum steering angle

def check_collision(x_list, y_list, kd_tree):
    for i_x, i_y in zip(x_list, y_list):
        if (kd_tree.query_ball_point([i_x, i_y], R)):
            return True # collision
        
    return False # no collision

def plot_arrow(x, y, yaw, length=2.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)

def plot_robot(x, y, yaw):
    color = 'k'
    circle = plt.Circle((x, y), R, color=color, fill=False)
    plt.gca().add_patch(circle)
    plot_arrow(x, y, yaw)
    
def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi    

def move(x, y, yaw, distance, steer):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(steer) 

    return x, y, yaw

def main():
    x, y, yaw = 0., 0., 1.
    plt.axis('equal')
    plot_robot(x, y, yaw)
    plt.show()
    
if __name__ == '__main__':
    main()