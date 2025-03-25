import real_control
import math
import realsenseD435_multi
import time

class AssembleEnv(object):
    def __init__(self):
        self.RAD2DEG = 180/math.pi
        self.DEG2RAD = math.pi/180
        self.rob = real_control.UR5_Real()
        self.rob.RTDE_SOFT_F_THRESHOLD = 15
        self.rob.RTDE_SOFT_RETURN = 0.003
        self.camera=realsenseD435_multi.RealsenseD435Multi(exposure=192, gain=50, contrast=25, gamma=120)



env = AssembleEnv()

env.rob.openRG2()
# time.sleep(2)
# 位置上方
env.rob.moveIK_changePosOrt(d_pos=[0e-3,0e-3,100e-3], d_ort=[0,0,0], wait=True)

