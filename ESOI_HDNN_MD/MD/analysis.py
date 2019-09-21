from parmed.amber import Rst7
import numpy as np

def coords_from_rst7_AMBER(filename,natom):
    rst7=Rst7(filename,natom)
    coords=[list(i._value) for i in rst7.positions]
    coords=np.array(coords)
    #if rst7.hasvels:
    #    print (rst7.velocities)
    #    velocities=rst7.velocities/1000/20.455
    #    return coords,velocities
    return coords

def coords_from_rst7_TM(filename,natom):
    rst7=Rst7(filename,natom)
    coords=[list(i._value) for i in rst7.positions]
    coords=np.array(coords)
    if rst7.hasvels:
        print (rst7.velocities)
        velocities=rst7.velocities
        return coords,velocities
    return coords

