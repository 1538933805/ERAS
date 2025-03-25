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

"设置被抓取的位姿"
pos_c = [203.89e-3, -449.28e-3, 87.44e-3, 0, -3.1415926, 1.5707963]
# env.rob.moveIK(pos_c[0:3], pos_c[3:6], wait=True)
pos_c_up = pos_c.copy()
pos_c_up[2] += 100e-3
"设置被装配的位姿"
pos_a = [-93.55e-3, -531.03e-3, 94e-3, 0, -3.1415, 0]
# env.rob.moveIK(pos_a[0:3], pos_a[3:6], wait=True)
pos_a_up = pos_a.copy()
pos_a_up[2] += 100e-3

env.rob.openRG2()
"catch"
# 被抓取位置上方
env.rob.moveIK(pos_c_up[0:3], pos_c_up[3:6], wait=True)
# 被抓取位置
env.rob.moveIK(pos_c[0:3], pos_c[3:6], wait=True)
# 抓取
env.rob.closeRG2()
# time.sleep(2)
# 被抓取位置上方
env.rob.moveIK(pos_c_up[0:3], pos_c_up[3:6], wait=True)
env.rob.moveIK_changePosOrt(d_pos=[0e-3,0e-3,50e-3], d_ort=[0,0,0], wait=True)


"go_to"
# 被装配位置上方
env.rob.moveIK(pos_a_up[0:3], pos_a_up[3:6], wait=True)
