from  TensorMol import * 
import numpy as np
import  pickle
from ..Comparm import *
class BP_HDNN():
    """
    Electrostatic embedding Behler Parinello with van der waals interaction implemented with Grimme C6 scheme.
    """
    def __init__(self, TData_, Name_=None, Trainable_=True,ForceType_="LJ"):
        """
        Args:
            TData_: A TensorMolData instance.
            Name_: A name for this instance.
        """
        self.path = GPARAMS.Neuralnetwork_setting.Networkprefix 
        self.Trainable = Trainable_
        self.need_Newtrain=True
        if Name_!=None:
            self.name=Name_
            self.Load()
            self.recorder=open('./'+GPARAMS.Compute_setting.Traininglevel+'/'+self.name+'.record','a')
            if self.TData==None and TData_ ==None and Trainable_==True:
                print ("ERROR: A Trainable BP_HDNN Instance don't have trainable dataset!!")
                exit
            if self.TData!=None and TData_== None and Trainable_==True:
                self.TData.LoadDataToScratch(self.tformer)
                self.max_steps = math.ceil(GPARAMS.Neuralnetwork_setting.Maxsteps*GPARAMS.Neuralnetwork_setting.Batchsize/self.TData.NTrain)
                self.switch_steps=math.ceil(self.max_steps*GPARAMS.Neuralnetwork_setting.Switchrate)
                self.batch_size = GPARAMS.Neuralnetwork_setting.Batchsize
                self.real_steps=math.ceil(self.max_steps*self.TData.NTrain/self.batch_size)
                self.real_switch_steps=math.ceil(self.real_steps*GPARAMS.Neuralnetwork_setting.Switchrate)
                self.learning_rate = np.array(GPARAMS.Neuralnetwork_setting.Learningrate)
                self.learning_rate_dipole = np.array(GPARAMS.Neuralnetwork_setting.Learningratedipole)
                self.learning_rate_energy = np.array(GPARAMS.Neuralnetwork_setting.Learningrateenergy)
                self.LR_boundary=[int(m) for m in np.array(GPARAMS.Neuralnetwork_setting.Learningrateboundary)*self.real_steps]
                self.LR_boundary_dipole=[int(m) for m in np.array(GPARAMS.Neuralnetwork_setting.Learningrateboundary)*self.real_switch_steps]
                self.LR_boundary_EF=[int(m) for m in np.array(GPARAMS.Neuralnetwork_setting.Learningrateboundary)*(self.real_steps-self.real_switch_steps)]#+self.real_switch_steps]
                self.Training_Target = "Dipole"
                self.recorder.write('MaxNAtoms: %d Max Epoch step: %d\tSwtich step: %d\t Training Target: %s\n'%(self.MaxNAtoms,self.max_steps,self.switch_steps, self.Training_Target))
            if TData_!=None:
                if Trainable_==True:
                    if TData_.eles==self.eles and TData_.MaxNAtoms==self.TData.MaxNAtoms:
                        self.TData=TData_
                        print ('TM has get the Training data')
                        self.tformer = Transformer(GPARAMS.Neuralnetwork_setting.Innormroutine, \
                                       GPARAMS.Neuralnetwork_setting.Outnormroutine,\
                                       self.element, self.TData.dig.name,\
                                       self.TData.dig.OType)
                        if (self.Trainable):
                            self.TData.LoadDataToScratch(self.tformer)
                        self.need_Newtrain=False
                        self.TData.ele=self.eles_np
                        self.TData.elep=self.eles_pairs_np
                        self.max_steps = math.ceil(GPARAMS.Neuralnetwork_setting.Maxsteps*GPARAMS.Neuralnetwork_setting.Batchsize/self.TData.NTrain)
                        self.switch_steps=math.ceil(self.max_steps*GPARAMS.Neuralnetwork_setting.Switchrate)
                        self.batch_size = GPARAMS.Neuralnetwork_setting.Batchsize
                        self.real_steps=math.ceil(self.max_steps*self.TData.NTrain/self.batch_size)
                        self.real_switch_steps=math.ceil(self.real_steps*GPARAMS.Neuralnetwork_setting.Switchrate)
                        self.learning_rate = np.array(GPARAMS.Neuralnetwork_setting.Learningrate)
                        self.learning_rate_dipole = np.array(GPARAMS.Neuralnetwork_setting.Learningratedipole)
                        self.learning_rate_energy = np.array(GPARAMS.Neuralnetwork_setting.Learningrateenergy)
                        self.LR_boundary=[int(m) for m in np.array(GPARAMS.Neuralnetwork_setting.Learningrateboundary)*self.real_steps]
                        self.LR_boundary_dipole=[int(m) for m in np.array(GPARAMS.Neuralnetwork_setting.Learningrateboundary)*self.real_switch_steps]
                        self.LR_boundary_EF=[int(m) for m in np.array(GPARAMS.Neuralnetwork_setting.Learningrateboundary)*(self.real_steps-self.real_switch_steps)]#+self.real_switch_steps]
                        self.Training_Target = "Dipole"
                        self.recorder.write('MaxNAtoms: %d Max Epoch step: %d\tSwtich step: %d\t Training Target: %s\n'%(self.MaxNAtoms,self.max_steps,self.switch_steps, self.Training_Target))
                        return
                    else:
                        print ('The New Training set is different in Ntypes of element or MaxNAtoms has changed!')
                        print (TData_.eles,self.eles)
                        print (TData_.MaxNAtoms,self.TData.MaxNAtoms)
                        self.need_Newtrain=True
            if Trainable_==False:
                return
        self.NetType = "RawBP_EE_Charge_DipoleEncode_Update_vdw_DSF_elu_Normalize_Dropout"
        self.TData = TData_
        self.element=0
        self.name = self.TData.name+"_"+self.TData.dig.name+"_"+self.NetType+"_"+str(self.element)
        #print ("./"+GPARAMS.Compute_setting.Traininglevel+'/'+self.name+".record")
        self.recorder=open('./'+GPARAMS.Compute_setting.Traininglevel+'/'+self.name+'.record','a')
        self.train_dir = GPARAMS.Neuralnetwork_setting.Networkprefix+self.name
        self.PreparedFor=0
        self.Training_Target="Dipole"
        self.eles = self.TData.eles
        self.n_eles = len(self.eles)
        self.eles_np = np.asarray(self.eles).reshape((self.n_eles,1))
        self.eles_pairs = []
        for i in range (len(self.eles)):
            for j in range(i, len(self.eles)):
                self.eles_pairs.append([self.eles[i], self.eles[j]])
        self.eles_pairs_np = np.asarray(self.eles_pairs)
        self.tformer = Transformer(GPARAMS.Neuralnetwork_setting.Innormroutine, \
                                   GPARAMS.Neuralnetwork_setting.Outnormroutine,\
                                   self.element, self.TData.dig.name,\
                                   self.TData.dig.OType)
        if (self.Trainable):
            self.TData.LoadDataToScratch(self.tformer)
        self.TData.ele = self.eles_np
        self.TData.elep = self.eles_pairs_np
        self.inshape = np.prod(self.TData.dig.eshape)
        self.outshape = np.prod(self.TData.dig.lshape)
        self.MeanStoich = self.TData.MeanStoich
        self.MeanNumAtoms = np.sum(self.MeanStoich)
        self.TData.PrintStatus()
        try:
            self.tf_prec
        except:
            self.tf_prec = eval(GPARAMS.Neuralnetwork_setting.tfprec)
        self.MaxNAtoms = self.TData.MaxNAtoms
        self.momentum = GPARAMS.Neuralnetwork_setting.Momentum

        self.max_steps = math.ceil(GPARAMS.Neuralnetwork_setting.Maxsteps*GPARAMS.Neuralnetwork_setting.Batchsize/self.TData.NTrain)
        self.switch_steps=math.ceil(self.max_steps*GPARAMS.Neuralnetwork_setting.Switchrate)
        self.batch_size = GPARAMS.Neuralnetwork_setting.Batchsize
        self.real_steps=math.ceil(self.max_steps*self.TData.NTrain/self.batch_size)
        self.real_switch_steps=math.ceil(self.real_steps*GPARAMS.Neuralnetwork_setting.Switchrate)
        self.learning_rate = np.array(GPARAMS.Neuralnetwork_setting.Learningrate)
        self.learning_rate_dipole = np.array(GPARAMS.Neuralnetwork_setting.Learningratedipole)
        self.learning_rate_energy = np.array(GPARAMS.Neuralnetwork_setting.Learningrateenergy)
        self.LR_boundary=[int(m) for m in np.array(GPARAMS.Neuralnetwork_setting.Learningrateboundary)*self.real_steps]
        self.LR_boundary_dipole=[int(m) for m in np.array(GPARAMS.Neuralnetwork_setting.Learningrateboundary)*self.real_switch_steps]
        self.LR_boundary_EF=[int(m) for m in np.array(GPARAMS.Neuralnetwork_setting.Learningrateboundary)*(self.real_steps-self.real_switch_steps)]#+self.real_switch_steps]

        self.max_checkpoints = GPARAMS.Neuralnetwork_setting.Maxcheckpoints
        self.activation_function_type = GPARAMS.Neuralnetwork_setting.Neuraltype
        self.activation_function = None
        self.AssignActivation()
        #Training Setting
        self.keep_prob = np.asarray(GPARAMS.Neuralnetwork_setting.Keepprob)
        self.nlayer = len(GPARAMS.Neuralnetwork_setting.Keepprob) - 1
        self.monitor_mset =  GPARAMS.Neuralnetwork_setting.Monitorset
        self.GradScalar = GPARAMS.Neuralnetwork_setting.Scalar["F"]
        self.EnergyScalar = GPARAMS.Neuralnetwork_setting.Scalar["E"]
        self.DipoleScalar = GPARAMS.Neuralnetwork_setting.Scalar["D"]
        self.HiddenLayers = GPARAMS.Neuralnetwork_setting.Structure
        self.batch_size = GPARAMS.Neuralnetwork_setting.Batchsize
        if (not os.path.isdir(self.path)):
            os.mkdir(self.path)
        self.chk_file = ''
        self.current_step=0
        #DSF Setting
        self.Ree_on  = GPARAMS.Neuralnetwork_setting.EEcutoffon
        self.Ree_off  = GPARAMS.Neuralnetwork_setting.EEcutoffoff
        self.DSFAlpha = GPARAMS.Neuralnetwork_setting.DSFAlpha
        if self.Ree_on!=0.0:
            raise Exception("EECutoffOn should equal to zero in DSF_elu")
        self.elu_width = GPARAMS.Neuralnetwork_setting.Eluwidth
        self.elu_shift=DSF(self.elu_width*BOHRPERA, self.Ree_off*BOHRPERA, self.DSFAlpha/BOHRPERA)
        self.elu_alpha=DSF_Gradient(self.elu_width*BOHRPERA, self.Ree_off*BOHRPERA, self.DSFAlpha/BOHRPERA)
        #Vdw Setting
        self.vdw_R = np.zeros(self.n_eles)        
        self.C6=np.zeros(self.n_eles)
        for i,ele in enumerate(self.eles):
            self.C6[i]=C6_coff[ele]* (BOHRPERA*10.0)**6.0 / JOULEPERHARTREE # convert into a.u.
            self.vdw_R[i] = atomic_vdw_radius[ele]*BOHRPERA
        #MolInstance_DirectBP_EE_ChargeEncode_Update
        self.SFPa = None
        self.SFPr = None
        self.Ra_cut = None
        self.Rr_cut = None
        self.HasANI1PARAMS = False
        if not self.HasANI1PARAMS:
            self.SetANI1Param()
        #MolInstance_fc_sqdiff_BP
        self.inp_pl=None
        self.inp_pl=None
        self.mats_pl=None
        self.label_pl=None
        self.batch_size_output = 0
        #MolInstance_fc_sqdiff
        self.summary_op =None
        self.summary_writer=None
        #MolInstance(Instance):
        #LOGGER.info("MolInstance.inshape %s MolInstance.outshape %s", str(self.inshape) , str(self.outshape))
        self.recorder.write('MaxNAtoms: %d Max Epoch step: %d\tSwtich step: %d\t Training Target: %s\n'%(self.MaxNAtoms,self.max_steps,self.switch_steps, self.Training_Target))
        #self.tf_precision = eval("tf.float64")
        #self.set_symmetry_function_params()

    def AssignActivation(self):
        LOGGER.debug("Assigning Activation... %s", GPARAMS.Neuralnetwork_setting.Neuraltype)
        try:
            if self.activation_function_type == "relu":
                self.activation_function = tf.nn.relu
            elif self.activation_function_type == "elu":
                self.activation_function = tf.nn.elu
            elif self.activation_function_type == "selu":
                self.activation_function = self.selu
            elif self.activation_function_type == "softplus":
                self.activation_function = tf.nn.softplus
            elif self.activation_function_type == "tanh":
                self.activation_function = tf.tanh
            elif self.activation_function_type == "sigmoid":
                self.activation_function = tf.sigmoid
            elif self.activation_function_type == "sigmoid_with_param":
                self.activation_function = sigmoid_with_param
            elif self.activation_function_type == "gaussian":
                self.activation_function = guassian_act
            elif self.activation_function_type == "gaussian_rev_tozero":
                self.activation_function = guassian_rev_tozero
            elif self.activation_function_type == "gaussian_rev_tozero_tolinear":
                self.activation_function = guassian_rev_tozero_tolinear
            elif self.activation_function_type == "square_tozero_tolinear":
                self.activation_function = square_tozero_tolinear
            else:
                print ("unknown activation function, set to relu")
                self.activation_function = tf.nn.relu
        except Exception as Ex:
            print(Ex)
            print ("activation function not assigned, set to relu")
            self.activation_function = tf.nn.relu
        return
    def _variable_with_weight_decay(self, var_name, var_shape, var_stddev, var_wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

        Returns:
        Variable Tensor
        """
        var = tf.Variable(tf.truncated_normal(var_shape, stddev=var_stddev, dtype=self.tf_prec), name=var_name)
        if var_wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), var_wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var
    def _get_weight_variable(self, name, shape):
        return tf.get_variable(name, shape, self.tf_prec, tf.truncated_normal_initializer(stddev=0.01))
    def _get_bias_variable(self, name, shape):
        return tf.get_variable(name, shape, self.tf_prec, tf.constant_initializer(0.01, dtype=self.tf_prec))
    def SetANI1Param(self, prec=np.float64):
        self.Ra_cut = GPARAMS.Neuralnetwork_setting.AN1_a_Rc
        self.Rr_cut = GPARAMS.Neuralnetwork_setting.AN1_r_Rc
        zetas = np.array([[GPARAMS.Neuralnetwork_setting.AN1_zeta]], dtype = prec)
        etas = np.array([[GPARAMS.Neuralnetwork_setting.AN1_eta]], dtype = prec)
        AN1_num_a_As = GPARAMS.Neuralnetwork_setting.AN1_num_a_As
        AN1_num_a_Rs = GPARAMS.Neuralnetwork_setting.AN1_num_a_Rs
        thetas = np.array([ 2.0*Pi*i/AN1_num_a_As for i in range (0, AN1_num_a_As)], dtype = prec)
        rs =  np.array([ self.Ra_cut*i/AN1_num_a_Rs for i in range (0, AN1_num_a_Rs)], dtype = prec)
        # Create a parameter tensor. 4 x nzeta X neta X ntheta X nr
        p1 = np.tile(np.reshape(zetas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
        p2 = np.tile(np.reshape(etas,[1,1,1,1,1]),[1,1,AN1_num_a_As,AN1_num_a_Rs,1])
        p3 = np.tile(np.reshape(thetas,[1,1,AN1_num_a_As,1,1]),[1,1,1,AN1_num_a_Rs,1])
        p4 = np.tile(np.reshape(rs,[1,1,1,AN1_num_a_Rs,1]),[1,1,AN1_num_a_As,1,1])
        SFPa = np.concatenate([p1,p2,p3,p4],axis=4)
        self.SFPa = np.transpose(SFPa, [4,0,1,2,3])
        etas_R = np.array([[GPARAMS.Neuralnetwork_setting.AN1_eta]], dtype = prec)
        AN1_num_r_Rs = GPARAMS.Neuralnetwork_setting.AN1_num_r_Rs
        rs_R =  np.array([ self.Rr_cut*i/AN1_num_r_Rs for i in range (0, AN1_num_r_Rs)], dtype = prec)
        # Create a parameter tensor. 2 x  neta X nr
        p1_R = np.tile(np.reshape(etas_R,[1,1,1]),[1,AN1_num_r_Rs,1])
        p2_R = np.tile(np.reshape(rs_R,[1,AN1_num_r_Rs,1]),[1,1,1])
        SFPr = np.concatenate([p1_R,p2_R],axis=2)
        self.SFPr = np.transpose(SFPr, [2,0,1])
        self.inshape = int(len(self.eles)*AN1_num_r_Rs + len(self.eles_pairs)*AN1_num_a_Rs*AN1_num_a_As)
        self.inshape_withencode = int(self.inshape + AN1_num_r_Rs)
        #self.inshape = int(len(self.eles)*AN1_num_r_Rs)
        p1 = np.tile(np.reshape(thetas,[AN1_num_a_As,1,1]),[1,AN1_num_a_Rs,1])
        p2 = np.tile(np.reshape(rs,[1,AN1_num_a_Rs,1]),[AN1_num_a_As,1,1])
        SFPa2 = np.concatenate([p1,p2],axis=2)
        self.SFPa2 = np.transpose(SFPa2, [2,0,1])
        p1_new = np.reshape(rs_R,[AN1_num_r_Rs,1])
        self.SFPr2 = np.transpose(p1_new, [1,0])
        self.zeta = GPARAMS.Neuralnetwork_setting.AN1_zeta
        self.eta = GPARAMS.Neuralnetwork_setting.AN1_eta
        self.HasANI1PARAMS = True
        #print ("self.inshape:", self.inshape)
    
    def Clean(self):
        #MolInstance_DirectBP_EE_ChargeEncode_Update_vdw_DSF_elu_Normalize.Clean(self)
        self.keep_prob_pl = None
        #self.elu_width = None
        #self.elu_shift = None
        #self.elu_alpha = None     
        self.Ebp_atom = None
        self.Evdw = None   
        self.Radp_Ele_pl = None
        self.Angt_Elep_pl = None
        self.mil_jk_pl = None
        self.Radius_Qs_Encode = None
        self.Radius_Qs_Encode_Index = None
        self.Elabel_pl = None
        self.Dlabel_pl = None
        self.Qlabel_pl =None
        self.Radp_pl = None
        self.Angt_pl = None
        self.Reep_pl = None
        self.natom_pl = None
        self.AddEcc_pl = None
        self.Etotal = None
        self.Ebp = None
        self.Ecc = None
        self.dipole = None
        self.charge = None
        self.energy_wb = None
        self.dipole_wb = None
        self.dipole_loss = None
        self.gradient = None
        self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = None, None, None, None, None
        self.train_op_dipole, self.train_op_EandG = None, None
        self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = None, None, None, None, None
        self.lr,self.lr_dipole,self.lr_ef=None,None,None
        self.run_metadata = None
        self.Radp_pl = None
        self.Angt_pl = None
        self.xyzs_pl=None
        self.check = None
        self.Zs_pl=None
        self.label_pl=None
        self.grads_pl = None
        self.atom_outputs = None
        self.energy_loss = None
        self.grads_loss = None
        self.Scatter_Sym = None
        self.Sym_Index = None
        self.options = None
        self.run_metadata = None
        self.graph=None
        self.sess = None
        self.loss = None
        self.output = None
        self.total_loss = None
        self.train_op = None
        self.embeds_placeholder = None
        self.labels_placeholder = None
        self.saver = None
        self.gradient =None
        self.summary_writer = None
        self.PreparedFor = 0
        self.summary_op = None
        #self.activation_function = None
        self.options = None
        self.run_metadata = None
        self.batch_size_ctrl=None 

    def TrainPrepare(self,  continue_training =False,chk_file=''):
        """
        Get placeholders, graph and losses in order to begin training.
        Also assigns the desired padding.

        Args:
            continue_training: should read the graph variables from a saved checkpoint.
        """
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([None, self.MaxNAtoms,3]),name="InputCoords")
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
            #MING ADDED
            self.batch_size_ctrl=tf.placeholder(dtype=tf.int64,shape=[None],name="Batch_size_ctrl")
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
            self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([None, self.MaxNAtoms]),name="InputZs")
            self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([None]),name="DesEnergy")
            self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([None, 3]),name="DesDipoles")
            self.Qlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([None, self.MaxNAtoms]),name="DesCharge")
            self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([None, self.MaxNAtoms,3]),name="DesGrads")
            self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
            self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
            self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
            self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]),name="RadialElectros")
            self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([None]))
            self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
            #self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=())
            self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
            Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
            Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
            #SFPa = tf.Variable(self.SFPa, trainable=False, dtype = self.tf_prec)
            #SFPr = tf.Variable(self.SFPr, trainable=False, dtype = self.tf_prec)
            SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
            SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
            Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
            Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
            Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
            elu_width  = tf.Variable(self.elu_width, trainable=False, dtype = self.tf_prec)
            Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
            zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
            eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
            C6 = tf.Variable(self.C6,trainable=False, dtype = self.tf_prec)
            vdw_R = tf.Variable(self.vdw_R,trainable=False, dtype = self.tf_prec)
            #self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear(self.xyzs_pl, self.Zs_pl, Ele, self.SFPr2_vary, Rr_cut, Elep, self.SFPa2_vary, zeta, eta, Ra_cut, self.Radp_pl, self.Angt_pl)
#            with tf.name_scope("MakeDescriptors"):
            #with tf.device('/job:localhost/replica:0/task:0/gpu:1'):
            #with tf.device('/cpu:0'):
            self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_jk_pl)
            self.Ecc, self.dipole, self.charge, self.dipole_wb = self.dipole_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, elu_width, Ree_off, self.Reep_pl, self.AddEcc_pl, self.keep_prob_pl,self.batch_size_ctrl)
            self.Radp_pl  = self.Radp_Ele_pl[:,:3]
#            with tf.name_scope("behler"):
            self.Etotal, self.Ebp, self.Evdw,  self.energy_wb, self.Ebp_atom = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.Ecc, self.xyzs_pl, self.Zs_pl, Ele, C6, vdw_R, self.Reep_pl, Ree_on, Ree_off, self.keep_prob_pl,self.batch_size_ctrl)
            #self.Etotal,  self.energy_wb = self.inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, Ree_on, Ree_off, self.Reep_pl)
            #self.check = tf.add_check_numerics_ops()
            self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
            #self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad", colocate_gradients_with_ops=True)
#            with tf.name_scope("losses"):
            self.total_loss, self.loss, self.energy_loss, self.grads_loss, self.dipole_loss = self.loss_op(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl, self.natom_pl)
            self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole = self.loss_op_dipole(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl, self.natom_pl)
            self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG = self.loss_op_EandG(self.Etotal, self.gradient, self.dipole, self.Elabel_pl, self.grads_pl, self.Dlabel_pl,self.natom_pl)
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("loss_dip", self.loss_dipole)
            tf.summary.scalar("loss_EG", self.loss_EandG)
#            with tf.name_scope("training"):
            self.train_op,self.lr = self.training(self.total_loss, self.learning_rate,self.LR_boundary,self.momentum, )
            self.train_op_dipole,self.lr_dipole = self.training(self.total_loss_dipole, self.learning_rate_dipole,self.LR_boundary_dipole, self.momentum, self.dipole_wb)
            self.train_op_EandG,self.lr_ef = self.training(self.total_loss_EandG, self.learning_rate_energy, self.LR_boundary_EF,self.momentum, self.energy_wb)
            self.summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            # please do not use the totality of the GPU memory
            config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.90
            self.sess = tf.Session(config=config)
           
            all_variables_list = tf.global_variables()
            restore_variables_list = []
            for item in all_variables_list: 
                if "global_step" not in item.name:
                    restore_variables_list.append(item) 
            self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
            self.sess.run(init)
            #Ming+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if chk_file!='':
                self.chk_file=chk_file
            if continue_training==True:
                self.saver.restore(self.sess, self.chk_file)
            #Ming---------------------------------------------------------------

            self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
            if (GPARAMS.Neuralnetwork_setting.Profiling>0):
                print("logging with FULL TRACE")
                self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                self.run_metadata = tf.RunMetadata()
                self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
            self.sess.graph.finalize()

    def fill_feed_dict(self, batch_data):
        """
        Fill the tensorflow feed dictionary.

        Args:
            batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
            and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

        Returns:
            Filled feed dictionary.
        """
        # Don't eat shit.
        if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
            print("I was fed shit")
            raise Exception("DontEatShit")
        feed_dict={i: d for i, d in zip([self.xyzs_pl]+[self.Zs_pl]+[self.Elabel_pl] + [self.Dlabel_pl] + [self.grads_pl] + [self.Radp_Ele_pl] + [self.Angt_Elep_pl] + [self.Reep_pl] + [self.mil_jk_pl] + [self.natom_pl] + [self.AddEcc_pl] + [self.keep_prob_pl], batch_data)}
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        feed_dict[self.batch_size_ctrl]=[batch_data[2].shape[0]]
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        return feed_dict

    def energy_inference(self, inp, indexs,  cc_energy, xyzs, Zs, eles, c6, R_vdw, Reep, EE_cuton, EE_cutoff, keep_prob,batch_size_ctrl):
        """
        Builds a Behler-Parinello graph

        Args:
            inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
            index: a list of (num_of atom type X batchsize) array which linearly combines the elements
        Returns:
            The BP graph output
        """
        # convert the index matrix from bool to float
        xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
        Ebranches=[]
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #output = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
        output=tf.zeros(tf.concat([batch_size_ctrl,[self.MaxNAtoms]],axis=0),dtype=self.tf_prec)
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        atom_outputs = []
        with tf.name_scope("EnergyNet"):
            for e in range(len(self.eles)):
                Ebranches.append([])
                inputs = inp[e]
                shp_in = tf.shape(inputs)
                index = tf.cast(indexs[e], tf.int64)
                for i in range(len(self.HiddenLayers)):
                    if i == 0:
                        with tf.name_scope(str(self.eles[e])+'_hidden1'):
                            weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
                            biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
                            Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(inputs, keep_prob[i]), weights) + biases))
                            #Ebranches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
                    else:
                        with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)):
                            weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
                            biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biaseslayer'+str(i))
                            Ebranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[i]), weights) + biases))
                            #Ebranches[-1].append(self.activation_function(tf.matmul(Ebranches[-1][-1], weights) + biases))
                with tf.name_scope(str(self.eles[e])+'_regression_linear'):
                    shp = tf.shape(inputs)
                    weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
                    biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
                    Ebranches[-1].append(tf.matmul(tf.nn.dropout(Ebranches[-1][-1], keep_prob[-1]), weights) + biases)
                    shp_out = tf.shape(Ebranches[-1][-1])
                    cut = tf.slice(Ebranches[-1][-1],[0,0],[shp_out[0],1])
                    rshp = tf.reshape(cut,[1,shp_out[0]])
                    atom_outputs.append(rshp)
                    rshpflat = tf.reshape(cut,[shp_out[0]])
                    atom_indice = tf.slice(index, [0,1], [shp_out[0],1])
                    #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
                    #ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[-1, self.MaxNAtoms])
                    ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, batch_size_ctrl*self.MaxNAtoms),[-1, self.MaxNAtoms]) 
                    #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
                    output = tf.add(output, ToAdd)
                tf.verify_tensor_all_finite(output,"Nan in output!!!")
            bp_energy = tf.reshape(tf.reduce_sum(output, axis=1), [-1])
        total_energy = tf.add(bp_energy, cc_energy)
        vdw_energy = TFVdwPolyLR(xyzsInBohr, Zs, eles, c6, R_vdw, EE_cuton*BOHRPERA, Reep)
        total_energy_with_vdw = tf.add(total_energy, vdw_energy)
        energy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="EnergyNet")
        return total_energy_with_vdw, bp_energy, vdw_energy, energy_vars, output

    def dipole_inference(self, inp, indexs, xyzs, natom, Elu_Width, EE_cutoff, Reep, AddEcc, keep_prob,batch_size_ctrl):
        """
        Builds a Behler-Parinello graph

        Args:
            inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
            index: a list of (num_of atom type X batchsize) array which linearly combines the elements
        Returns:
            The BP graph output
        """
        # convert the index matrix from bool to float
        xyzsInBohr = tf.multiply(xyzs,BOHRPERA)
        Dbranches=[]
        atom_outputs_charge = []
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #output_charge = tf.zeros([self.batch_size, self.MaxNAtoms], dtype=self.tf_prec)
        output_charge=tf.zeros(tf.concat([batch_size_ctrl,[self.MaxNAtoms]],axis=0),dtype=self.tf_prec)
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        dipole_wb = []
        with tf.name_scope("DipoleNet"):
            for e in range(len(self.eles)):
                Dbranches.append([])
                charge_inputs = inp[e]
                charge_shp_in = tf.shape(charge_inputs)
                charge_index = tf.cast(indexs[e], tf.int64)
                for i in range(len(self.HiddenLayers)):
                    if i == 0:
                        with tf.name_scope(str(self.eles[e])+'_hidden1_charge'):
                            weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.inshape, self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.inshape))), var_wd=0.001)
                            biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
                            Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(charge_inputs, keep_prob[i]), weights) + biases))
                            #Dbranches[-1].append(self.activation_function(tf.matmul(charge_inputs, weights) + biases))
                            dipole_wb.append(weights)
                            dipole_wb.append(biases)
                    else:
                        with tf.name_scope(str(self.eles[e])+'_hidden'+str(i+1)+"_charge"):
                            weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[i-1], self.HiddenLayers[i]], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[i-1]))), var_wd=0.001)
                            biases = tf.Variable(tf.zeros([self.HiddenLayers[i]], dtype=self.tf_prec), name='biases')
                            Dbranches[-1].append(self.activation_function(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[i]), weights) + biases))
                            #Dbranches[-1].append(self.activation_function(tf.matmul(Dbranches[-1][-1], weights) + biases))
                            dipole_wb.append(weights)
                            dipole_wb.append(biases)
                with tf.name_scope(str(self.eles[e])+'_regression_linear_charge'):
                    charge_shp = tf.shape(charge_inputs)
                    weights = self._variable_with_weight_decay(var_name='weights', var_shape=[self.HiddenLayers[-1], 1], var_stddev=1.0/(10+math.sqrt(float(self.HiddenLayers[-1]))), var_wd=None)
                    biases = tf.Variable(tf.zeros([1], dtype=self.tf_prec), name='biases')
                    dipole_wb.append(weights)
                    dipole_wb.append(biases)
                    Dbranches[-1].append(tf.matmul(tf.nn.dropout(Dbranches[-1][-1], keep_prob[-1]), weights) + biases)
                    shp_out = tf.shape(Dbranches[-1][-1])
                    cut = tf.slice(Dbranches[-1][-1],[0,0],[shp_out[0],1])
                    rshp = tf.reshape(cut,[1,shp_out[0]])
                    atom_outputs_charge.append(rshp)
                    rshpflat = tf.reshape(cut,[shp_out[0]])
                    atom_indice = tf.slice(charge_index, [0,1], [shp_out[0],1])
                    #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
                    #ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, [self.batch_size*self.MaxNAtoms]),[self.batch_size, self.MaxNAtoms])
                    ToAdd = tf.reshape(tf.scatter_nd(atom_indice, rshpflat, batch_size_ctrl*self.MaxNAtoms),[-1, self.MaxNAtoms]) 
                    #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
                    output_charge = tf.add(output_charge, ToAdd)
            tf.verify_tensor_all_finite(output_charge,"Nan in output!!!")
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
            #netcharge = tf.reshape(tf.reduce_sum(output_charge, axis=1), [self.batch_size])
            netcharge = tf.reshape(tf.reduce_sum(output_charge, axis=1), [-1])
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
            delta_charge = tf.multiply(netcharge, natom)
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
            #delta_charge_tile = tf.tile(tf.reshape(delta_charge,[self.batch_size,1]),[1, self.MaxNAtoms])
            delta_charge_tile = tf.tile(tf.reshape(delta_charge,[-1,1]),[1, self.MaxNAtoms])
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
            scaled_charge =  tf.subtract(output_charge, delta_charge_tile)
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
            # Ming :output_charge=tf.zeros(tf.concat([batch_size_ctrl,[self.MaxNAtoms]],axis=0),dtype=self.tf_prec)
            #flat_dipole = tf.multiply(tf.reshape(xyzsInBohr,[self.batch_size*self.MaxNAtoms, 3]), tf.reshape(scaled_charge,[self.batch_size*self.MaxNAtoms, 1]))
            #dipole = tf.reduce_sum(tf.reshape(flat_dipole,[self.batch_size, self.MaxNAtoms, 3]), axis=1)
            flat_dipole = tf.multiply(tf.reshape(xyzsInBohr,tf.concat([batch_size_ctrl*self.MaxNAtoms,[3]],axis=0)), tf.reshape(scaled_charge,tf.concat([batch_size_ctrl*self.MaxNAtoms,[1]],axis=0)))
            dipole = tf.reduce_sum(tf.reshape(flat_dipole,tf.concat([batch_size_ctrl,[self.MaxNAtoms,3]],axis=0)), axis=1)
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII

        def f1(): return TFCoulombEluSRDSFLR(xyzsInBohr, scaled_charge, Elu_Width*BOHRPERA, Reep, tf.cast(self.DSFAlpha, self.tf_prec), tf.cast(self.elu_alpha,self.tf_prec), tf.cast(self.elu_shift,self.tf_prec))
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #def f2(): return  tf.zeros([self.batch_size], dtype=self.tf_prec)
        def f2(): return  tf.zeros(batch_size_ctrl, dtype=self.tf_prec)
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        cc_energy = tf.cond(AddEcc, f1, f2)
        #dipole_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DipoleNet")
        return  cc_energy, dipole, scaled_charge, dipole_wb

    def loss_op(self, energy, energy_grads, dipole, Elabels, grads, Dlabels, natom):
        maxatom=tf.cast(tf.shape(energy_grads)[2], self.tf_prec)
        energy_diff  = tf.multiply(tf.subtract(energy, Elabels,name="EnDiff"), natom*maxatom)
        energy_loss = tf.nn.l2_loss(energy_diff,name="EnL2")
        
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #grads_diff = tf.multiply(tf.subtract(energy_grads, grads,name="GradDiff"), tf.reshape(natom*maxatom, [1, self.batch_size, 1, 1]))
        grads_diff = tf.multiply(tf.subtract(energy_grads, grads,name="GradDiff"), tf.reshape(natom*maxatom, [1, -1, 1, 1]))
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        grads_loss = tf.nn.l2_loss(grads_diff,name="GradL2")
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels,name="DipoleDiff"), tf.reshape(natom*maxatom,[self.batch_size,1]))
        dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels,name="DipoleDiff"), tf.reshape(natom*maxatom,[-1,1]))
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        
        dipole_loss = tf.nn.l2_loss(dipole_diff,name="DipL2")
        #loss = tf.multiply(grads_loss, energy_loss)
        EandG_loss = tf.add(tf.multiply(energy_loss, self.EnergyScalar), tf.multiply(grads_loss, self.GradScalar),name="MulLoss")
        loss = tf.add(EandG_loss, tf.multiply(dipole_loss, self.DipoleScalar))
        #loss = tf.identity(dipole_loss)
        tf.add_to_collection('losses', loss)
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss, dipole_loss

    def loss_op_dipole(self, energy, energy_grads, dipole, Elabels, grads, Dlabels, natom):
        maxatom=tf.cast(tf.shape(energy_grads)[2], self.tf_prec)
        energy_diff  = tf.multiply(tf.subtract(energy, Elabels), natom*maxatom)
        energy_loss = tf.nn.l2_loss(energy_diff)
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #grads_diff = tf.multiply(tf.subtract(energy_grads, grads), tf.reshape(natom*maxatom, [1, self.batch_size, 1, 1]))
        grads_diff = tf.multiply(tf.subtract(energy_grads, grads), tf.reshape(natom*maxatom, [1, -1, 1, 1]))
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        grads_loss = tf.nn.l2_loss(grads_diff)
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels), tf.reshape(natom*maxatom,[self.batch_size,1]))
        dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels), tf.reshape(natom*maxatom,[-1,1]))
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        dipole_loss = tf.nn.l2_loss(dipole_diff)
        #loss = tf.multiply(grads_loss, energy_loss)
        EandG_loss = tf.add(tf.multiply(energy_loss, self.EnergyScalar), tf.multiply(grads_loss, self.GradScalar))
        loss = tf.identity(dipole_loss)
        tf.add_to_collection('losses', loss)
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss, dipole_loss

    def loss_op_EandG(self, energy, energy_grads, dipole, Elabels, grads, Dlabels, natom):
        maxatom=tf.cast(tf.shape(energy_grads)[2], self.tf_prec)
        energy_diff  = tf.multiply(tf.subtract(energy, Elabels), natom*maxatom)
        energy_loss = tf.nn.l2_loss(energy_diff)
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #grads_diff = tf.multiply(tf.subtract(energy_grads, grads), tf.reshape(natom*maxatom, [1, self.batch_size, 1, 1]))
        grads_diff = tf.multiply(tf.subtract(energy_grads, grads), tf.reshape(natom*maxatom, [1, -1, 1, 1]))
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        grads_loss = tf.nn.l2_loss(grads_diff)
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels), tf.reshape(natom*maxatom,[self.batch_size,1]))
        dipole_diff = tf.multiply(tf.subtract(dipole, Dlabels), tf.reshape(natom*maxatom,[-1,1]))
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        dipole_loss = tf.nn.l2_loss(dipole_diff)
        #loss = tf.multiply(grads_loss, energy_loss)
        EandG_loss = tf.add(tf.multiply(energy_loss, self.EnergyScalar), tf.multiply(grads_loss, self.GradScalar))
        #loss = tf.add(EandG_loss, tf.multiply(dipole_loss, self.DipoleScalar))
        loss = tf.identity(EandG_loss)
        #loss = tf.identity(energy_loss)
        tf.add_to_collection('losses', loss)
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss, energy_loss, grads_loss, dipole_loss

    def training(self, loss, LR,LR_boundary, momentum, update_var=None):
        """Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
        Returns:
        train_op: The Op for training.
        """
        tf.summary.scalar(loss.op.name, loss)
        #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, list(LR_boundary), list(LR))
        optimizer = tf.train.AdamOptimizer(learning_rate,name="Adam")
        if update_var == None:
            train_op = optimizer.minimize(loss, global_step=global_step, name="trainop")
        else:
            train_op = optimizer.minimize(loss, global_step=global_step, var_list=update_var, name="trainop")
        return train_op,learning_rate


    def train_step_EandG(self, step):
        """
        Perform a single training step (complete processing of all input), using minibatches of size self.batch_size
        Args:
            step: the index of this step.
        """
        Ncase_train = self.TData.NTrain
        #print ('NTrain:',self.TData.NTrain,'N ministep per epoch:',int(Ncase_train/self.batch_size))
        start_time = time.time()
        train_loss =  0.0
        train_energy_loss = 0.0
        train_dipole_loss = 0.0
        train_grads_loss = 0.0
        num_of_mols = 0
        print_per_mini = 100
        print_loss = 0.0
        print_energy_loss = 0.0
        print_dipole_loss = 0.0
        print_grads_loss = 0.0
        print_time = 0.0
        time_print_mini = time.time()
        duration=time.time()-start_time
        for ministep in range (0, int(Ncase_train/self.batch_size)):
            
            t_mini = time.time()
            batch_data = self.TData.GetTrainBatch(self.batch_size)+[GPARAMS.Neuralnetwork_setting.AddEcc] + [self.keep_prob]
            actual_mols  = self.batch_size
            t = time.time()
            dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, Evdw, mol_dipole, atom_charge,lr_ef = self.sess.run([self.train_op_EandG, self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG, self.Etotal, self.Ecc, self.Evdw,  self.dipole, self.charge,self.lr_ef], feed_dict=self.fill_feed_dict(batch_data))
            print_loss += loss_value
            print_energy_loss += energy_loss
            print_grads_loss += grads_loss
            print_dipole_loss += dipole_loss
            if (ministep%print_per_mini == 0 and ministep!=0):
                #print ("time:", (time.time() - time_print_mini)/print_per_mini ,  " loss_value: ",  print_loss/print_per_mini, " energy_loss:", print_energy_loss/print_per_mini, " grads_loss:", print_grads_loss/print_per_mini, " dipole_loss:", print_dipole_loss/print_per_mini)
                print_loss = 0.0
                print_energy_loss = 0.0
                print_dipole_loss = 0.0
                print_grads_loss = 0.0
                print_time = 0.0
                time_print_mini = time.time()
            train_loss = train_loss + loss_value
            train_energy_loss += energy_loss
            train_grads_loss += grads_loss
            train_dipole_loss += dipole_loss
            num_of_mols += actual_mols
        duration = time.time() - start_time
        self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration,lr_ef)
        return


    def train_step_dipole(self, step):
        """
        Perform a single training step (complete processing of all input), using minibatches of size self.batch_size
        Args:
            step: the index of this step.
        """
        Ncase_train = self.TData.NTrain
        #print (Ncase_train)
        start_time = time.time()
        train_loss =  0.0
        train_energy_loss = 0.0
        train_dipole_loss = 0.0
        train_grads_loss = 0.0
        num_of_mols = 0
        pre_output = np.zeros((self.batch_size),dtype=np.float64)
        print_per_mini = 100
        print_loss = 0.0
        print_energy_loss = 0.0
        print_dipole_loss = 0.0
        print_grads_loss = 0.0
        print_time = 0.0
        time_print_mini = time.time()
        #print (int(Ncase_train/self.batch_size))
        duration=time.time()-start_time
        for ministep in range (0, int(Ncase_train/self.batch_size)):
            t_mini = time.time()
            batch_data = self.TData.GetTrainBatch(self.batch_size) + [False] + [self.keep_prob]
            actual_mols  = self.batch_size
            t = time.time()
            dump_2, total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge,lr_dipole = self.sess.run([self.train_op_dipole, self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole, self.Etotal, self.Ecc,  self.dipole,self.charge,self.lr_dipole], feed_dict=self.fill_feed_dict(batch_data))
            print_loss += loss_value
            print_energy_loss += energy_loss
            print_grads_loss += grads_loss
            print_dipole_loss += dipole_loss
            if (ministep%print_per_mini == 0 and ministep!=0):
                #print ("time:", (time.time() - time_print_mini)/print_per_mini ,  " loss_value: ",  print_loss/print_per_mini, " energy_loss:", print_energy_loss/print_per_mini, " grads_loss:", print_grads_loss/print_per_mini, " dipole_loss:", print_dipole_loss/print_per_mini)
                print_loss = 0.0
                print_energy_loss = 0.0
                print_dipole_loss = 0.0
                print_grads_loss = 0.0
                print_time = 0.0
                time_print_mini = time.time()
            train_loss = train_loss + loss_value
            train_energy_loss += energy_loss
            train_grads_loss += grads_loss
            train_dipole_loss += dipole_loss
            duration = time.time() - start_time
            num_of_mols += actual_mols
        self.print_training(step, train_loss, train_energy_loss, train_grads_loss, train_dipole_loss, num_of_mols, duration,lr_dipole)
        return

    def test_dipole(self, step):
        """
        Perform a single test step (complete processing of all input), using minibatches of size self.batch_size
        Args:
            step: the index of this step.
        """
        Ncase_test = self.TData.NTest
        start_time = time.time()
        test_loss =  0.0
        test_energy_loss = 0.0
        test_dipole_loss = 0.0
        test_grads_loss = 0.0
        num_of_mols = 0
        duration = time.time() - start_time
        for ministep in range (0, int(Ncase_test/self.batch_size)):
            batch_data = self.TData.GetTestBatch(self.batch_size)+[False] + [np.ones(self.nlayer+1)]
            actual_mols  = self.batch_size
            t = time.time()
            total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.total_loss_dipole, self.loss_dipole, self.energy_loss_dipole, self.grads_loss_dipole, self.dipole_loss_dipole, self.Etotal, self.Ecc, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
            test_loss = test_loss + loss_value
            test_energy_loss += energy_loss
            test_grads_loss += grads_loss
            test_dipole_loss += dipole_loss
            duration = time.time() - start_time
            num_of_mols += actual_mols
        #print ("testing...")
        self.print_training(step, test_loss, test_energy_loss, test_grads_loss, test_dipole_loss, num_of_mols, duration,0,False)
        return  test_loss

    def test_EandG(self, step):
        """
        Perform a single test step (complete processing of all input), using minibatches of size self.batch_size
        Args:
            step: the index of this step.
        """
        Ncase_test = self.TData.NTest
        start_time = time.time()
        test_loss =  0.0
        test_energy_loss = 0.0
        test_dipole_loss = 0.0
        test_grads_loss = 0.0
        num_of_mols = 0
        duration = time.time() - start_time
        for ministep in range (0, int(Ncase_test/self.batch_size)):
            batch_data = self.TData.GetTestBatch(self.batch_size)+[GPARAMS.Neuralnetwork_setting.AddEcc] + [np.ones(self.nlayer+1)]
            actual_mols  = self.batch_size
            t = time.time()
            total_loss_value, loss_value, energy_loss, grads_loss,  dipole_loss,  Etotal, Ecc, mol_dipole, atom_charge = self.sess.run([self.total_loss_EandG, self.loss_EandG, self.energy_loss_EandG, self.grads_loss_EandG, self.dipole_loss_EandG, self.Etotal, self.Ecc, self.dipole, self.charge], feed_dict=self.fill_feed_dict(batch_data))
            test_loss = test_loss + loss_value
            test_energy_loss += energy_loss
            test_grads_loss += grads_loss
            test_dipole_loss += dipole_loss
            duration = time.time() - start_time
            num_of_mols += actual_mols
        #print ("testing...")
        self.print_training(step, test_loss, test_energy_loss, test_grads_loss, test_dipole_loss, num_of_mols, duration,0,False)
        return  test_loss

    def train(self, mxsteps, continue_training= False,chk_file=''):
        """
        This the training loop for the united model.
        """
        LOGGER.info("running the TFMolInstance.train()")
        if self.need_Newtrain==True:
            continue_training=False
            chk_file=''
        self.TrainPrepare(continue_training,chk_file)
        test_freq = GPARAMS.Neuralnetwork_setting.Testfreq
        mini_dipole_test_loss = float('inf') # some big numbers
        mini_energy_test_loss = float('inf')
        mini_test_loss = float('inf')
        for step in  range (0, mxsteps):
            if self.Training_Target == "EandG":
                self.train_step_EandG(step)
                if step%test_freq==0 and step!=0 :
                    if self.monitor_mset != None:
                        self.InTrainEval(self.monitor_mset, self.Rr_cut, self.Ra_cut, self.Ree_off, step=step)
                    test_energy_loss = self.test_EandG(step)
                    if test_energy_loss < mini_energy_test_loss:
                        mini_energy_test_loss = test_energy_loss
                        self.save_chk(step)
            elif self.Training_Target == "Dipole":
                self.train_step_dipole(step)
                if step%test_freq==0 and step!=0 :
                    if self.monitor_mset != None:
                        self.InTrainEval(self.monitor_mset, self.Rr_cut, self.Ra_cut, self.Ree_off, step=step)
                    test_dipole_loss = self.test_dipole(step)
                    if test_dipole_loss < mini_dipole_test_loss:
                        mini_dipole_test_loss = test_dipole_loss
                        self.save_chk(step)
                    if step >= self.switch_steps:
                        self.saver.restore(self.sess, self.chk_file)
                        self.Training_Target = "EandG"
                        self.recorder.write("Switching to Energy and Gradient Learning...\n")
            else:
                self.train_step(step)
                if step%test_freq==0 and step!=0 :
                    if self.monitor_mset != None:
                        self.InTrainEval(self.monitor_mset, self.Rr_cut, self.Ra_cut, self.Ree_off, step=step)
                    test_loss = self.test(step)
                    if test_loss < mini_test_loss:
                        mini_test_loss = test_loss
                        self.save_chk(step)
        self.SaveAndClose()
        return

    def InTrainEval(self, mol_set, Rr_cut, Ra_cut, Ree_cut, step=0):
        """
        The energy, force and dipole routine for BPs_EE.
        """
        nmols = len(mol_set.mols)
        for i in range(nmols, self.batch_size):
            mol_set.mols.append(mol_set.mols[-1])
        nmols = len(mol_set.mols)
        dummy_energy = np.zeros((nmols))
        dummy_dipole = np.zeros((nmols, 3))
        xyzs = np.zeros((nmols, self.MaxNAtoms, 3), dtype = np.float64)
        dummy_grads = np.zeros((nmols, self.MaxNAtoms, 3), dtype = np.float64)
        Zs = np.zeros((nmols, self.MaxNAtoms), dtype = np.int32)
        natom = np.zeros((nmols), dtype = np.int32)
        for i, mol in enumerate(mol_set.mols):
            xyzs[i][:mol.NAtoms()] = mol.coords
            Zs[i][:mol.NAtoms()] = mol.atoms
            natom[i] = mol.NAtoms()
        NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
        rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(Rr_cut, Ra_cut, self.eles_np, self.eles_pairs_np)
        NLEE = NeighborListSet(xyzs, natom, False, False,  None)
        rad_eep = NLEE.buildPairs(Ree_cut)
        batch_data = [xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom]
        feed_dict=self.fill_feed_dict(batch_data+[GPARAMS.Neuralnetwork_setting.AddEcc]+[np.ones(self.nlayer+1)])
        Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient= self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.Ecc, self.Evdw, self.dipole, self.charge, self.gradient], feed_dict=feed_dict)
        monitor_data = [Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient]
        f = open(self.name+"_monitor_"+str(step)+".dat","wb")
        pickle.dump(monitor_data, f)
        f.close()
        #print ("calculating monitoring set..")
        return Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient

    def print_training(self, step, loss, energy_loss, grads_loss, dipole_loss, Ncase, duration,lr=0, Train=True):
        if Train:
            self.recorder.write("step: %7d L_rate: %.6f duration: %.5f  Train Loss: Total: %.6f  E: %.6f  F: %.6f, D: %.6f "%(step,lr, duration, (float(loss)/(Ncase)), (float(energy_loss)/(Ncase)), (float(grads_loss)/(Ncase)), (float(dipole_loss)/(Ncase))))
            self.recorder.flush()
            if step==0:
                self.recorder.write("\n")
                self.recorder.flush()
        else:
            self.recorder.write("Test Loss: Total %.6f E: %.6f F: %.6f, D: %.6f\n"%( float(loss)/(Ncase), float(energy_loss)/(Ncase), float(grads_loss)/Ncase, float(dipole_loss)/Ncase))
            self.recorder.flush()
        return

    def evaluate(self, batch_data):
        """
        Evaluate the energy, atom energies, and IfGrad = True the gradients
        of this Direct Behler-Parinello graph.
        """
        # Check sanity of input
        nmol = batch_data[2].shape[0]
        self.activation_function_type = GPARAMS.Neuralnetwork_setting.Neuraltype
        self.AssignActivation()
        #print ("self.activation_function:\n\n", self.activation_function)
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        #if (batch_data[0].shape[1] != self.MaxNAtoms or self.batch_size != nmol):
        #    self.MaxNAtoms = batch_data[0].shape[1]
        #    self.batch_size = nmol
        #    print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
        #    print ("loading the session..")
        #    self.EvalPrepare()
        #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII

        LOGGER.debug("nmol: %i", batch_data[2].shape[0])
        self.batch_size = nmol
        if not self.sess:
            print ("self.batch_size:", self.batch_size, "  self.MaxNAtoms:", self.MaxNAtoms)
            print ("loading the session..")
            self.EvalPrepare()
        feed_dict=self.fill_feed_dict(batch_data+[GPARAMS.Neuralnetwork_setting.AddEcc]+[np.ones(self.nlayer+1)])
        Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient = self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.Ecc, self.Evdw, self.dipole, self.charge, self.gradient], feed_dict=feed_dict)
        #Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient, bp_gradient, syms= self.sess.run([self.Etotal, self.Ebp, self.Ebp_atom, self.Ecc, self.Evdw, self.dipole, self.charge, self.gradient, self.bp_gradient, self.Scatter_Sym], feed_dict=feed_dict)
        #print ("Etotal:", Etotal, " bp_gradient", bp_gradient)
        #return Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient, bp_gradient, syms
        return Etotal, Ebp, Ebp_atom, Ecc, Evdw, mol_dipole, atom_charge, gradient

    def EvalPrepare(self,  continue_training =False):
        """
        Get placeholders, graph and losses in order to begin training.
        Also assigns the desired padding.

        Args:
            continue_training: should read the graph variables from a saved checkpoint.
        """
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.xyzs_pl=tf.placeholder(self.tf_prec, shape=tuple([None, self.MaxNAtoms,3]),name="InputCoords")
            self.Zs_pl=tf.placeholder(tf.int64, shape=tuple([None, self.MaxNAtoms]),name="InputZs")
            self.Elabel_pl = tf.placeholder(self.tf_prec, shape=tuple([None]),name="DesEnergy")
            self.Dlabel_pl = tf.placeholder(self.tf_prec, shape=tuple([None, 3]),name="DesDipoles")
            self.grads_pl=tf.placeholder(self.tf_prec, shape=tuple([None, self.MaxNAtoms,3]),name="DesGrads")
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
            #MING ADDED
            self.batch_size_ctrl=tf.placeholder(dtype=tf.int64,shape=[None],name="Batch_size_ctrl")
            #IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
            self.Radp_Ele_pl=tf.placeholder(tf.int64, shape=tuple([None,4]))
            self.Angt_Elep_pl=tf.placeholder(tf.int64, shape=tuple([None,5]))
            self.mil_jk_pl = tf.placeholder(tf.int64, shape=tuple([None,4]))
            self.Reep_pl=tf.placeholder(tf.int64, shape=tuple([None,3]),name="RadialElectros")
            self.natom_pl = tf.placeholder(self.tf_prec, shape=tuple([None]))
            self.keep_prob_pl =  tf.placeholder(self.tf_prec, shape=tuple([self.nlayer+1]))
            self.AddEcc_pl = tf.placeholder(tf.bool, shape=())
            Ele = tf.Variable(self.eles_np, trainable=False, dtype = tf.int64)
            Elep = tf.Variable(self.eles_pairs_np, trainable=False, dtype = tf.int64)
            SFPa2 = tf.Variable(self.SFPa2, trainable= False, dtype = self.tf_prec)
            SFPr2 = tf.Variable(self.SFPr2, trainable= False, dtype = self.tf_prec)
            Rr_cut = tf.Variable(self.Rr_cut, trainable=False, dtype = self.tf_prec)
            Ra_cut = tf.Variable(self.Ra_cut, trainable=False, dtype = self.tf_prec)
            Ree_on = tf.Variable(self.Ree_on, trainable=False, dtype = self.tf_prec)
            elu_width  = tf.Variable(self.elu_width, trainable=False, dtype = self.tf_prec)
            Ree_off = tf.Variable(self.Ree_off, trainable=False, dtype = self.tf_prec)
            zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_prec)
            eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_prec)
            C6 = tf.Variable(self.C6,trainable=False, dtype = self.tf_prec)
            vdw_R = tf.Variable(self.vdw_R,trainable=False, dtype = self.tf_prec)
            self.Scatter_Sym, self.Sym_Index  = TFSymSet_Scattered_Linear_WithEle(self.xyzs_pl, self.Zs_pl, Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut, self.Radp_Ele_pl, self.Angt_Elep_pl, self.mil_jk_pl)
            self.Ecc, self.dipole, self.charge, self.dipole_wb = self.dipole_inference(self.Scatter_Sym, self.Sym_Index, self.xyzs_pl, self.natom_pl, elu_width, Ree_off, self.Reep_pl, self.AddEcc_pl, self.keep_prob_pl,self.batch_size_ctrl)
            self.Radp_pl  = self.Radp_Ele_pl[:,:3]
            self.Etotal, self.Ebp, self.Evdw,  self.energy_wb, self.Ebp_atom = self.energy_inference(self.Scatter_Sym, self.Sym_Index, self.Ecc, self.xyzs_pl, self.Zs_pl, Ele, C6, vdw_R, self.Reep_pl, Ree_on, Ree_off, self.keep_prob_pl,self.batch_size_ctrl)
            self.gradient  = tf.gradients(self.Etotal, self.xyzs_pl, name="BPEGrad")
            self.bp_gradient  = tf.gradients(self.Ebp, self.xyzs_pl, name="BPGrad")
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
            self.saver.restore(self.sess, self.chk_file)
            if (GPARAMS.Neuralnetwork_setting.Profiling>0):
                print("logging with FULL TRACE")
                self.summary_writer = tf.summary.FileWriter('./networks/PROFILE', self.sess.graph)
                self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                self.run_metadata = tf.RunMetadata()
                self.summary_writer.add_run_metadata(self.run_metadata, "init", global_step=None)
            self.sess.graph.finalize()

    def SaveAndClose(self):
        self.summary_op =None
        self.summary_writer=None
        self.check=None
        self.label_pl = None
        self.mats_pl = None
        self.prob = None
        self.correct = None
        self.inp_pl = None
        self.graph=None
        self.recorder.close()
        self.recorder=None
        if (self.TData!=None):
            self.TData.CleanScratch()
        if (self.sess != None):
            self.sess.close()
        
        print("Saving TFInstance...")
        self.Clean()
        #print("Going to pickle...\n",[(attr,type(ins)) for attr,ins in self.__dict__.items()])
        f=open(self.path+self.name+".tfn","wb")
        #print (self.__dict__)
        pickle.dump(self.__dict__, f ,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        return

    def Load(self):
        print ("Unpickling TFInstance...")
        from TensorMol.Containers.PickleTM import UnPickleTM as UnPickleTM
        tmpname=self.name
        tmp = UnPickleTM(self.path+self.name+".tfn")
        self.Clean()
        self.__dict__.update(tmp)
        # Simple hack to fix checkpoint path.
        self.name=tmpname
        self.train_dir = GPARAMS.Neuralnetwork_setting.Networkprefix+self.name
        self.chk_file = os.path.join(self.train_dir,self.name+'-chk')
        self.chk_file=self.chk_file.replace("./networks/",GPARAMS.Neuralnetwork_setting.Networkprefix)
        print("self.chk_file:", self.chk_file)
        return 

    def save_chk(self, step):  # We need to merge this with the one in TFInstance
        #self.chk_file = os.path.join(self.train_dir,self.name+'-chk-'+str(step))
        self.chk_file = os.path.join(self.train_dir,self.name+'-chk')
        self.recorder.write("Saving Checkpoint file in the TFMoInstance\n")
        self.saver.save(self.sess,  self.chk_file)
        return

