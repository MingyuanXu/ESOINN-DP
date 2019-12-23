import numpy as np
import os
import argparse



parser=argparse.ArgumentParser(description='Deal with training Set')
parser.add_argument('-i','--log',help="log")
args=parser.parse_args()

def read_esplogfile(filename):
    file=open(filename,'r')
    line=file.readline()
    natom=0;crd=[]
    espnum=0;espdat=[]
    fitdat=[];ele_index=[]
    Normal_flag=False
    dipole=[]
    charge=[]
    force=[]
    unit=0.529177249
    
    while line:
        if 'Charge' in line and 'Multiplicity' in line:
            totalcharge=int(line.split()[2])

        if 'SCF Done' in line:
            var=line.split()
            energy=float(var[4])
        if 'Atomic Center' in line:
            natom+=1
            var=line.split()
            tmpcrd=[float(line[31:42]),float(line[42:52]),float(line[52:61])]
            #print (tmpcrd)
            crd.append(tmpcrd)
        if 'ESP Fit' in line:
            espnum+=1
            tmpesp=[float(line[31:42]),float(line[42:52]),float(line[52:61])]
            espdat.append(tmpesp)
        if "Fit    " in line:
            fitdat.append(float(line.split()[-1]))
        if "Forces (Hartrees/Bohr)" in line:
            line=file.readline()
            line=file.readline()
            for i in range(natom):
                line=file.readline()
                tmp_eleindex=int(line.split()[1])
                var=line.split()
                tmpforce=[float(var[2]),float(var[3]),float(var[4])]
                force.append(tmpforce)
                ele_index.append(tmp_eleindex)
        if 'Predicted change' in line:
                DBLOCK=''
                while 'Normal termination' not in line:
                    if 'Error termination' in line:
                        print (filename+' is end with error')
                    DBLOCK=DBLOCK+line.strip('\n').strip()
                    line=file.readline()
                var=DBLOCK.split('\\')
                for i in var:
                    if 'Dipole' in i:
                        dipole_str=i.strip(' Dipole=')
                        print (filename,dipole_str)
                        dipole=[float(m) for m in dipole_str.split(',')]
        if 'Normal termination' in line:
            Normal_flag=True
        line=file.readline()
    file.close()

    dir_name = filename.strip('.log')+'_resp'
    os.system ('mkdir '+dir_name)

    file=open(dir_name+'/esp.dat','w')
    file.write('%5d%5d\n'%(natom,espnum))
    for i in range(natom):
        file.write('%s%16.6E%16.6E%16.6E\n'\
                %('                ',crd[i][0]/unit,crd[i][1]/unit,crd[i][2]/unit))
    for i in range(espnum):
        file.write('%16.6E%16.6E%16.6E%16.6E\n'\
                %(fitdat[i],espdat[i][0]/unit,espdat[i][1]/unit,espdat[i][2]/unit))
    print (filename,natom,espnum)
    file.close()
    if Normal_flag==True:
        file=open (dir_name+'/resp.in','w')
        file.write('resp for MP\n')
        file.write(' &cntrl nmol=1, ihfree=1\n')
        file.write(' &end\n')
        file.write('1.0\n')
        file.write('zinc metalloprotein\n')
        file.write('%5d%5d\n'%(totalcharge,natom))
        for i in range(natom):
            file.write('%5d%5d\n'%(ele_index[i],0))
        file.write('\n')
        file.close()
        file=open(dir_name+'_efd.dat','w')
        file.write('Natom: %d\n'%natom)
        file.write('Coords in angstrom:\n')
        for i in range(natom ):
            file.write('%8d %8d %15.6f %15.6f %15.6f\n'%(i,ele_index[i],crd[i][0],crd[i][1],crd[i][2]))
        file.write('Energy in Hartree: %f\n'%energy)
        file.write('Force in hartree/bohr:\n')
        for i in range(natom):
            file.write('%15.6f %15.6f %15.6f\n'%(force[i][0],force[i][1],force[i][2]))
        file.write('Dipole in e*bohr:\n')
        file.write('%15.6f %15.6f %15.6f\n'%(dipole[0],dipole[1],dipole[2]))
        file.write('Total charge in e:\n')
        file.write('%d\n'%(totalcharge))
        file.close()
        
    return Normal_flag,natom

def cal_resp_charge(filename):
    flag,natom=read_esplogfile(filename=filename)
    if flag==True:
        dir_name=filename.strip('.log')+'_resp'
        os.system('resp -O -i '+dir_name+'/resp.in -o '+dir_name+'/resp.out -p '+dir_name\
                +'/resp.pch -t '+dir_name+'/resp.chg -e '+dir_name+'/esp.dat\n')
        file=open(dir_name+'/resp.out','r')
        line=file.readline()
        resp_charge=[]
        while line:
            if 'Point Charges Before' in line:
                line=file.readline()
                line=file.readline()
                for i in range(natom):
                    line=file.readline()
                    tmpq=float(line.split()[3])
                    resp_charge.append(tmpq)
            line=file.readline()
        resp_charge=np.array(resp_charge)
        file.close()
        file=open(dir_name+'_efd.dat','a')
        file.write('RESP charge in e:\n')
        for i in range(natom):
            file.write('%15.6f\n'%resp_charge[i])
        file.close()
        #os.system('rm -r '+dir_name)
    else :
        resp_charge=np.zeros(1)
        file=open(dirname+'_efd.dat','a')
        file.write('RESP charge calculation failed!\n')
    return [flag,resp_charge]
try:
    flag,resp_c=cal_resp_charge(args.log)
except:
    print (args.log+' calculation failed!')

