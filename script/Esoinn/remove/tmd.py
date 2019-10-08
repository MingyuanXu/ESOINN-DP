import numpy as np
import os
def write_restrain_to_block(i,j,dis,restrain):
    block=[]
    block.append('  &rst\n')
    block.append('    iat=%d,%d\n'%(i,j))
    block.append('    r1=%.2f , r2=%.2f , r3=%.2f , r4=%.2f ,\n'%(dis-0.5,dis-0.25,dis+0.25,dis+0.5))
    block.append('    rk2=%.2f , rk3=%.2f ,\n' %(restrain,restrain))
    block.append('  /\n')
    s=''
    for i in block:
        s=s+i
    return s
def create_restrain_file(MOL,S_struc,E_struc,path='./',if_consider_bond=False):
    natom=MOL.GetNumAtoms()
    E_index=[]
    for i in MOL.GetAtoms():
        E_index.append(i.GetAtomicNum())
    if '/' not in path:
        print ('Error: / not in path variable!!!')
        stop
    else:
        restrain_file=open(path+'DIS_restrain.in','w')
        restrain_file.write('Restrain')
        Bond_list=MOL.GetBonds()
        for i in range(0,natom-1):
            for j in range(i+1,natom):
                if MOL.GetBondBetweenAtoms(i,j):
                    if if_consider_bond:
                        dis=np.sqrt(np.sum((E_struc[i]-E_struc[j])**2))
                        restrain=20
                        restrain_block=write_restrain_to_block(i,j,dis,restrain)
                        restrain_file.write(restrain_block)
                else:
                    dis=np.sqrt(np.sum((E_struc[i]-E_struc[j])**2))
                    restrain=15
                    restrain_block=write_restrain_to_block(i,j,dis,restrain)
                    restrain_file.write(restrain_block)
        restrain_file.close()
    return
def write_Leapin(path,pdbname,prmname,inpname):
    if '/' not in path:
        print ('Error: / not in path variable!!!')
        stop
    else:
        file=open(path+'Leap.in','w')
        file.write('source oldff/leaprc.ff14SB\n')
        file.write('loadoff ../lib/zn.lib\n')
        file.write('loadamberparams ../lib/zn.dat\n')
        file.write('m=loadpdb %s\n'%pdbname)
        file.write('saveamberparm m %s %s \n'%(prmname,inpname))
        file.write('quit\n')
        file.close()
    return

def write_mdin(path):
    if '/' not in path:
        print ('Error: / not in path variable!!!')
        stop
    else:
        mdinfile=open(path+'md.in','w')
        mdin='''100 ps NPT production for 120 deg
 &cntrl
  imin = 0, ntx = 1, irest = 0,
  ntpr = 10, ntwr = 10, ntwx = 10,
  ntf = 1, ntc = 1, cut = 999.0,
  ntb = 0,igb=0, nstlim = 20000, dt = 0.001,
  nmropt = 1, ioutfm = 0,
 &end
 &wt
  type='DUMPFREQ', istep1=50,
 &end
 &wt
  type='END',
 &end
DISANG=DIS_restrain.in
DUMPAVE=DIS_restrain.out
            '''
        mdinfile.write(mdin)
        mdinfile.close()
    return
def mdcrd2pdb(path):
    if '/' not in path:
        print ('Error: / not in path variable!!!')
        stop
    else:
        trajinfile=open(path+'mdcrd2pdb.in','w')
        trajin='''
trajin md.mdcrd 1 1000 10
trajout md.pdb PDB
        '''
        trajinfile.write(trajin)
        trajinfile.close()
    return





