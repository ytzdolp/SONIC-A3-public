#!/usr/bin/env python3
"""Build the 29-DOF floating-base A3 MJCF used by SONIC.

MuJoCo's URDF importer merges the URDF root into the world.  This script wraps
the imported tree back in a floating pelvis, preserves the pelvis inertia, and
locks the two head joints while retaining their bodies and mass.
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


def build(urdf: Path, output: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf))
    with tempfile.NamedTemporaryFile(suffix=".xml") as imported:
        mujoco.mj_saveLastXML(imported.name, model)
        tree = ET.parse(imported.name)

    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "../urdf/a3/meshes")
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", Path(mesh.attrib["file"]).name)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Imported MJCF has no worldbody")

    pelvis = ET.Element("body", {"name": "pelvis", "pos": "0 0 1.063"})
    pelvis.append(ET.Element("freejoint", {"name": "floating_base_joint"}))
    # Values are copied from the root link of the manufacturer URDF.
    pelvis.append(
        ET.Element(
            "inertial",
            {
                "pos": "-0.00090674 0.00002058 -0.12728789",
                "mass": "4.86725331",
                "fullinertia": "0.02194473 0.01757488 0.01349901 0.0000022 0.00003787 0.00001476",
            },
        )
    )
    for child in list(worldbody):
        worldbody.remove(child)
        pelvis.append(child)
    worldbody.append(pelvis)

    for joint_name in ("head_yaw_joint", "head_pitch_joint"):
        joint = root.find(f".//joint[@name='{joint_name}']")
        # The checked-in SONIC URDF already locks these axes.  Keep this
        # removal for compatibility with the original 31-DOF manufacturer URDF.
        if joint is None:
            continue
        for body in root.iter("body"):
            if joint in list(body):
                body.remove(joint)
                break

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if name and joint.attrib.get("type") != "free":
            ET.SubElement(actuator, "motor", {"name": name, "joint": name})

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode", xml_declaration=False)

    check = mujoco.MjModel.from_xml_path(str(output))
    if (check.nq, check.nv, check.njnt) != (36, 35, 30):
        raise RuntimeError(
            f"Unexpected A3 model dimensions: nq={check.nq}, nv={check.nv}, njnt={check.njnt}"
        )
    print(f"Wrote {output}: nq={check.nq}, nv={check.nv}, bodies={check.nbody}, joints={check.njnt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.urdf.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
