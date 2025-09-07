import pybullet as p, pybullet_data
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
rid = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
print("------JOINTS------")
for j in range(p.getNumJoints(rid)):
    info = p.getJointInfo(rid, j)
    print(j, info[1].decode(), "-> parent link:", info[12].decode(), "child link:", info[12].decode())
    # info[12] is link name (child), info[1] is joint name


print("------LINKS------")
for j in range(p.getNumJoints(rid)):
    print(j, p.getJointInfo(rid, j)[12].decode())


p.disconnect()