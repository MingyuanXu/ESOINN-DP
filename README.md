
                      
          _---~~(~~-_.        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      _{        )   )         ========================================================================
    ,   ) -~~- ( ,-' )_                    ___________ ____  _____   ___   __      ____  ____
   (  `-,_..`., )-- '_,)                  / ____/ ___// __ \/  _/ | / / | / /     / __ \/ __ \
  ( ` _)  (  -~( -_ `,  }                / __/  \__ \/ / / // //  |/ /  |/ /_____/ / / / /_/ /
  (_-  _  ~_-~~~~`,  ,' )               / /___ ___/ / /_/ // // /|  / /|  /_____/ /_/ / ____/
    `~ -^(    __;-,((()))              /_____//____/\____/___/_/ |_/_/ |_/     /_____/_/
          ~~~~ {_ -_(())
                 `\  }       ======================  Author:  Mingyuan Xu, Tong Zhu ==================
                   { }      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ab initio MD simulation packages with enhanced self organized increment high dimensional neural network 

Installation:

#Installation of requirements:

conda create -n esoihdnn python=3.6

conda activate esoihdnn

pip –-upgrade install pip

pip install parmed tensorflow-gpu==1.14

conda install numpy scipy scikit-learn paramiko matplotlib seaborn

conda install -c omnia openmm


tar zxvf ESOI-HDNN-MD.tar.gz

cd ESOI-HDNN-MD

#Installation of Library and TensorMol

cd packages

we need to install cuda_10.0.*_linux.run to an user-defined path and add cudnn-10.0-linux-x64*.tar to user-defined cuda path

&& tar zxvf TensorMol.tar.gz

cd TensorMol && pip install -e . && cd ..

#Installation of ESOI-HDNN-MD

cd ../../ && python setup.py install


