"""Static gravity torque per joint, computed from a URDF.

The deflection this project keeps measuring is driven by gravitational load, and
for a stationary arm that load is a deterministic function of the configuration
and the mass distribution -- no torque sensor required. This module parses the
URDF's CAD-derived masses and centres of mass and evaluates, for each requested
joint, the moment of gravity about that joint's axis produced by everything
downstream of it.

Accuracy caveats, in order of importance: the gripper payload is not in the URDF
(capture with the payload you will deploy with); CAD masses miss cabling; and the
base is assumed level (gravity along -z of the root link). Scale errors in the
masses are absorbed by any learned per-joint coefficient -- only the shape of
tau(q) across configurations matters, and that is set by geometry.

Run directly for a sanity check against a dataset:

    python -m examples.gravity_torque <dataset directory>/end_effector.npz
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ElementTree

import numpy as np

DEFAULT_URDF = pathlib.Path(__file__).resolve().parent.parent / "models" / "rby1_no_world.urdf"
GRAVITY = np.array([0.0, 0.0, -9.81])


def rotation_rpy(rpy):
    """URDF fixed-axis roll-pitch-yaw to a rotation matrix (Rz @ Ry @ Rx)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rotation_axis(axis, angle):
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


class Robot:
    """Just enough of a URDF: the kinematic tree, joint axes, masses, and COMs."""

    def __init__(self, urdf_path=DEFAULT_URDF):
        root = ElementTree.parse(urdf_path).getroot()
        self.links = {}
        for link in root.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass = float(inertial.find("mass").get("value"))
            origin = inertial.find("origin")
            com = (np.zeros(3) if origin is None
                   else np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]))
            self.links[link.get("name")] = (mass, com)

        self.joints = {}
        self.children = {}
        child_links = set()
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz, rpy = np.zeros(3), np.zeros(3)
            if origin is not None:
                xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
                rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
            axis_element = joint.find("axis")
            axis = (np.array([1.0, 0.0, 0.0]) if axis_element is None
                    else np.array([float(v) for v in axis_element.get("xyz").split()]))
            record = {
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "xyz": xyz, "rpy": rpy, "axis": axis,
            }
            self.joints[joint.get("name")] = record
            self.children.setdefault(record["parent"], []).append(joint.get("name"))
            child_links.add(record["child"])
        parents = {j["parent"] for j in self.joints.values()}
        roots = parents - child_links
        if len(roots) != 1:
            raise ValueError(f"expected one root link, found {sorted(roots)}")
        self.root = roots.pop()

    def forward(self, positions):
        """World pose of every link and world axis/origin of every joint.

        ``positions`` maps joint name to value; unlisted joints sit at zero.
        """
        frames = {self.root: (np.eye(3), np.zeros(3))}
        joint_world = {}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            rotation_parent, position_parent = frames[parent]
            for name in self.children.get(parent, ()):
                joint = self.joints[name]
                rotation = rotation_parent @ rotation_rpy(joint["rpy"])
                origin = position_parent + rotation_parent @ joint["xyz"]
                value = positions.get(name, 0.0)
                axis_unit = joint["axis"] / np.linalg.norm(joint["axis"])
                joint_world[name] = (rotation @ axis_unit, origin)
                if joint["type"] in ("revolute", "continuous"):
                    child_rotation, child_origin = rotation @ rotation_axis(joint["axis"], value), origin
                elif joint["type"] == "prismatic":
                    child_rotation, child_origin = rotation, origin + rotation @ (axis_unit * value)
                else:
                    child_rotation, child_origin = rotation, origin
                frames[joint["child"]] = (child_rotation, child_origin)
                stack.append(joint["child"])
        return frames, joint_world

    def subtree(self, joint_name):
        """Every link at or below a joint's child -- what that joint carries."""
        out, stack = [], [self.joints[joint_name]["child"]]
        while stack:
            link = stack.pop()
            out.append(link)
            stack.extend(self.joints[n]["child"] for n in self.children.get(link, ()))
        return out

    def gravity_torques(self, positions, joint_names):
        """Gravitational moment about each named joint's axis, N m."""
        frames, joint_world = self.forward(positions)
        torques = np.empty(len(joint_names))
        for index, name in enumerate(joint_names):
            axis, origin = joint_world[name]
            moment = np.zeros(3)
            for link in self.subtree(name):
                if link not in self.links:
                    continue
                mass, com = self.links[link]
                rotation, position = frames[link]
                world_com = position + rotation @ com
                moment += np.cross(world_com - origin, mass * GRAVITY)
            torques[index] = moment @ axis
        return torques

    def link_pose(self, positions, link):
        frames, _ = self.forward(positions)
        rotation, position = frames[link]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, position
        return pose


def torques_for_dataset(joint_positions, joint_names, arm_joints, urdf_path=DEFAULT_URDF):
    """Torque features for an (n, n_joints) array of full joint vectors."""
    robot = Robot(urdf_path)
    names = [str(n) for n in joint_names]
    out = np.empty((len(joint_positions), len(arm_joints)))
    for row, q in enumerate(joint_positions):
        out[row] = robot.gravity_torques(dict(zip(names, q)), arm_joints)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=pathlib.Path, help="a target NPZ with joint_position")
    parser.add_argument("--urdf", type=pathlib.Path, default=DEFAULT_URDF)
    arguments = parser.parse_args()

    data = np.load(arguments.npz, allow_pickle=False)
    robot = Robot(arguments.urdf)
    names = [str(n) for n in data["joint_name"]]
    arm = [n for n in names if n.startswith("right_arm_")][:7]

    print(f"URDF root link: {robot.root}")
    print(f"total mass in model: {sum(m for m, _ in robot.links.values()):.2f} kg\n")

    # FK cross-check: the URDF chain evaluated at the recorded configuration should
    # reproduce the dataset's own A = base_T_link. This validates the parser far more
    # strongly than any torque eyeballing.
    link = str(data["tf_link_name"])
    errors = []
    for row in range(len(data["A"])):
        positions = dict(zip(names, data["joint_position"][row]))
        predicted = robot.link_pose(positions, link)
        errors.append(np.linalg.norm(predicted[:3, 3] - data["A"][row][:3, 3]))
    errors = np.array(errors) * 1e3
    print(f"FK cross-check against A ({link}): "
          f"median {np.median(errors):.3f} mm, max {errors.max():.3f} mm")

    torques = torques_for_dataset(data["joint_position"], names, arm, arguments.urdf)
    print("\ngravity torque ranges across the capture (N m):")
    for index, name in enumerate(arm):
        print(f"  {name:14s} {torques[:, index].min():+8.2f} .. {torques[:, index].max():+8.2f}")


if __name__ == "__main__":
    main()
