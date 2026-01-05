from ClassFly import *
from ClassMission import *
from ClassSmoke import *
from clc import *


if __name__ == "__main__":
    init()
    
    smoke_class_list[0].flag = 1
    mission_class_list[0].flag = 1
    
    fly1.angle = 180
    fly1.v = 120
    smoke_class_list[0].fly_time = 1.5
    smoke_class_list[0].fall_time = 3.6

    clc()

    print(total_time[0])

