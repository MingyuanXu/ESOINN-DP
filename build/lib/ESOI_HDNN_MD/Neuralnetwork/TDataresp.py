from __future__ import absolute_import
from __future__ import print_function
import os, gc
from TensorMol import *
from ..Comparm import *
class TData_BP_Direct_EE_WithCharge():
#class TensorMolData_BP_Direct_EE(TensorMolData_BP_Direct_Linear):
#class TensorMolData_BP_Direct_Linear(TensorMolData_BP_Direct):
#class TensorMolData_BP_Direct(TensorMolData):
#class TensorMolData(TensorData):
#class TensorData():
    """
    A Training Set is a Molecule set, with a sampler and an embedding
    The sampler chooses points in the molecular volume.
    The embedding turns that into inputs and labels for a network to regress.
    This tensordata serves up batches digested within TensorMol.
    """
    def __init__(self, MSet_=None,  Dig_=None, Name_=None, order_=3, num_indis_=1, type_="mol", WithGrad_ = False,MaxNAtoms=None):
        """
        make a tensordata object
        Several arguments of PARAMS affect this classes behavior

        Args:
            MSet_: A MoleculeSet
            Dig_: A Digester
            Name_: A Name
        """
                #TensorMolData_BP_Direct_EE.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_, WithGrad_)
                #TensorMolData_BP_Direct_Linear.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_, WithGrad_)
                #TensorMolData_BP_Direct.__init__(self, MSet_, Dig_, Name_, order_, num_indis_, type_, WithGrad_)
        self.HasGrad = WithGrad_ # whether to pass around the gradient.
        self.path = "./trainsets/"
        self.suffix = ".pdb"
        self.set = MSet_
        self.set_name = None
        if (self.set != None):
            print("loading the set...")
            self.set_name = MSet_.name # Check to make sure the name can recall the set.
            print("finished loading the set..")
        self.dig = Dig_
        self.type = type_
        self.CurrentElement = None # This is a mode switch for when TensorData provides training data.
        self.SamplesPerElement = []
        self.AvailableElements = []
        self.AvailableDataFiles = []
        self.NTest = 0  # assgin this value when the data is loaded
        self.TestRatio = GPARAMS.Neuralnetwork_setting.Testratio # number of cases withheld for testing.
        self.Random = GPARAMS.Neuralnetwork_setting.Randomizedata # Whether to scramble training data (can be disabled for debugging purposes)
        self.ScratchNCase = 0
        self.ScratchState=None
        self.ScratchPointer=0 # for non random batch iteration.
        self.scratch_inputs=None
        self.scratch_outputs=None
        self.scratch_test_inputs=None # These should be partitioned out by LoadElementToScratch
        self.scratch_test_outputs=None
        self.Classify=GPARAMS.Neuralnetwork_setting.Classify # should be moved to transformer.
        self.MxTimePerElement=GPARAMS.Neuralnetwork_setting.Maxtimeperelement
        self.MxMemPerElement=GPARAMS.Neuralnetwork_setting.Maxmemperelement
        self.ChopTo = GPARAMS.Neuralnetwork_setting.Chopto
        self.ExpandIsometriesAltogether = False
        self.ExpandIsometriesBatchwise = False

        # Ordinarily during training batches will be requested repeatedly
        # for the same element. Introduce some scratch space for that.
        if (not os.path.isdir(self.path)):
            os.mkdir(self.path)
        if (Name_!= None):
            self.name = Name_
            self.Load()
            return
        elif (MSet_==None or Dig_==None):
            raise Exception("I need a set and Digester if you're not loading me.")
        self.name = ""

        self.order = order_
        self.num_indis = num_indis_
        self.NTrain = 0
        TensorData.__init__(self, MSet_,Dig_,Name_, type_=type_)
        try:
            LOGGER.info("TensorMolData.type: %s",self.type)
            LOGGER.info("TensorMolData.dig.name: %s",self.dig.name)
            LOGGER.info("NMols in TensorMolData.set: %i", len(self.set.mols))
            self.raw_it = iter(self.set.mols)
        except:
            print(" do not include MSet")

        self.MaxNAtoms=MaxNAtoms
        self.MaxNAtoms_real=None

        try:
            if (MSet_ != None):
                self.MaxNAtoms = MSet_.MaxNAtoms()
        except:
            print("fail to load self.MaxNAtoms")

        self.eles = []
        if (MSet_ != None):
            self.eles = list(MSet_.AtomTypes())
            self.eles.sort()
            self.MaxNAtoms_real = np.max([m.NAtoms() for m in self.set.mols])
            print("self.MaxNAtoms:", self.MaxNAtoms)
            self.Nmols = len(self.set.mols)

        if self.MaxNAtoms==None:
            self.MaxNAtoms=self.MaxNAtoms_real
        elif self.MaxNAtoms!=None and self.MaxNAtoms<self.MaxNAtoms_real:
            self.MaxNAtoms=self.MaxNAtoms_real
            print ('Warning: Pre set MaxNatoms is lower then real MaxNatoms in trainingset')
        
        self.MeanStoich=None
        self.MeanNAtoms=None
        self.test_mols_done = False
        self.test_begin_mol  = None
        self.test_mols = []
        self.MaxN3 = None # The most coordinates in the set.
        self.name = self.set.name
        print("TensorMolData_BP.eles", self.eles)
        print("self.HasGrad:", self.HasGrad)
        self.Rr_cut = GPARAMS.Neuralnetwork_setting.AN1_r_Rc
        self.Ra_cut = GPARAMS.Neuralnetwork_setting.AN1_a_Rc
        self.Ree_cut = GPARAMS.Neuralnetwork_setting.EEcutoffoff
        self.ele = None #  determine later
        self.elep = None # determine later
        return

    def QueryAvailable(self):
        """ If Tensordata has already been made, this looks for it under a passed name."""
        # It should probably check the sanity of each input/outputfile as well...
        return
    def ReloadSet(self):
        """
        Recalls the MSet to build training data etc.
        """
        self.set = MSet(self.set_name)
        self.set.Load()
        return

    def CheckShapes(self):
        # Establish case and label shapes.
        if self.type=="frag":
            tins,touts = self.dig.Emb(test_mol.mbe_frags[self.order][0],False,False)
        elif self.type=="mol":
            if (self.set != None):
                test_mol = self.set.mols[0]
                tins,touts = self.dig.Emb(test_mol,True,False)
            else:
                return
        else:
            raise Exception("Unknown Type")
        print("self.dig ", self.dig.name)
        print("self.dig input shape: ", self.dig.eshape)
        print("self.dig output shape: ", self.dig.lshape)
        if (self.dig.eshape == None or self.dig.lshape ==None):
            raise Exception("Ain't got no fucking shape.")
        return 

    def BuildTrain(self, name_="gdb9",  append=False):
        self.CheckShapes()
        self.name=name_
        total_case = 0
        for mi in range(len(self.set.mols)):
            total_case += len(self.set.mols[mi].mbe_frags[self.order])
        cases = np.zeros(tuple([total_case]+list(self.dig.eshape)))
        labels = np.zeros(tuple([total_case]+list(self.dig.lshape)))
        casep=0
        insname = self.path+"Mol_"+name_+"_"+self.dig.name+"_"+str(self.order)+"_in.npy"
        outsname = self.path+"Mol_"+name_+"_"+self.dig.name+"_"+str(self.order)+"_out.npy"
        if self.type=="frag":
            for mi in range(len(self.set.mols)):
                for frag in self.set.mols[mi].mbe_frags[self.order]:
                    #print  frag.dist[0], frag.frag_mbe_energy
                    ins,outs = self.dig.TrainDigest(frag)
                    cases[casep:casep+1] += ins
                    labels[casep:casep+1] += outs
                    casep += 1
        elif self.type=="mol":
            for mi in range(len(self.set.mols)):
                if (mi%10000==0):
                    LOGGER.debug("Mol: "+str(mi))
                ins,outs = self.dig.TrainDigest(mi)
                cases[casep:casep+1] += ins
                labels[casep:casep+1] += outs
                casep += 1
        else:
            raise Exception("Unknown Type")
        alreadyexists = (os.path.isfile(insname) and os.path.isfile(outsname))
        if (append and alreadyexists):
            ti=None
            to=None
            inf = open(insname,"rb")
            ouf = open(outsname,"rb")
            ti = np.load(inf)
            to = np.load(ouf)
            inf.close()
            ouf.close()
            cases = np.concatenate((cases[:casep],ti))
            labels = np.concatenate((labels[:casep],to))
            inf = open(insname,"wb")
            ouf = open(outsname,"wb")
            np.save(inf,cases)
            np.save(ouf,labels)
            inf.close()
            ouf.close()
            self.AvailableDataFiles.append([insname,outsname])
            #self.SamplesPerElement.append(casep*self.dig.NTrainSamples)
        else:
            inf = open(insname,"wb")
            ouf = open(outsname,"wb")
            np.save(inf,cases[:casep])
            np.save(ouf,labels[:casep])
            inf.close()
            ouf.close()
            self.AvailableDataFiles.append([insname,outsname])
            #self.SamplesPerElement.append(casep*self.dig.NTrainSamples)
        self.Save() #write a convenience pickle.
        return

    def RawBatch(self,nmol = 4096):
        """
            Shimmy Shimmy Ya Shimmy Ya Shimmy Yay.
            This type of batch is not built beforehand
            because there's no real digestion involved.

            Args:
                    nmol: number of molecules to put in the output.

            Returns:
                    Ins: a #atomsX4 tensor (AtNum,x,y,z)
                    Outs: output of the digester
                    Keys: (nmol)X(MaxNAtoms) tensor listing each molecule's place in the input.
        """
        ndone = 0
        natdone = 0
        self.MaxNAtoms = self.set.MaxNAtoms()
        Ins = np.zeros(tuple([nmol,self.MaxNAtoms,4]))
        Outs = np.zeros(tuple([nmol,self.MaxNAtoms,3]))
        while (ndone<nmol):
            try:
                m = next(self.raw_it)
#                print "m props", m.properties.keys()
#                print "m coords", m.coords
                ti, to = self.dig.Emb(m, True, False)
                n=ti.shape[0]

                Ins[ndone,:n,:] = ti.copy()
                Outs[ndone,:n,:] = to.copy()
                ndone += 1
                natdone += n
            except StopIteration:
                self.raw_it = iter(self.set.mols)
        return Ins,Outs

    def CleanScratch(self):
        TensorData.CleanScratch(self)
        self.raw_it = None
        self.xyzs = None
        self.Zs = None
        self.labels = None
        self.grads = None
        return

    def LoadData(self):
        if (self.set == None):
            try:
                self.ReloadSet()
            except Exception as Ex:
                print("TData doesn't have a set.", Ex)
        random.shuffle(self.set.mols)
        xyzs = np.zeros((self.Nmols, self.MaxNAtoms, 3), dtype = np.float64)
        Zs = np.zeros((self.Nmols, self.MaxNAtoms), dtype = np.int32)
        natom = np.zeros((self.Nmols), dtype = np.int32)

        if (self.dig.OType == "EnergyAndDipole"):
            Elabels  = np.zeros((self.Nmols), dtype = np.float64)
            Dlabels  = np.zeros((self.Nmols, 3),  dtype = np.float64)
            Qlabels  = np.zeros((self.Nmols,self.MaxNAtoms),dtype=np.float64)
            Qtlabels = np.zeros((self.Nmols),dtype=np.float64)
            masks    = np.zeros((self.Nmols,self.MaxNAtoms),dtype=np.float64)
        else:
            raise Exception("Output Type is not implemented yet")

        if (self.HasGrad):
            grads = np.zeros((self.Nmols, self.MaxNAtoms, 3), dtype=np.float64)
        for i, mol in enumerate(self.set.mols):
            try:
                xyzs[i][:mol.NAtoms()] = mol.coords
                Zs[i][:mol.NAtoms()] = mol.atoms
                natom[i] = mol.NAtoms()
            except Exception as ex:
                print(mol.coords, mol.atoms, mol.coords.shape[0], mol.atoms.shape[0])
                raise Exception("Bad data2")
            if (self.dig.OType  == "EnergyAndDipole"):
                Elabels[i] = mol.properties["atomization"]
                Dlabels[i] = mol.properties["dipole"]*AUPERDEBYE
                if GPARAMS.Esoinn_setting.Ifresp==True:
                    Qlabels[i][:mol.NAtoms()] = mol.properties["resp_charge"]
                elif GPARAMS.Esoinn_setting.Ifadch==True:
                    Qlabels[i][:mol.NAtoms()] = mol.properties["adch_charge"]

                Qtlabels[i]=np.sum(Qlabels[i])
                masks[i][:mol.NAtoms()]=np.ones(mol.NAtoms(),dtype=np.float64)
            else:
                raise Exception("Output Type is not implemented yet")
            if (self.HasGrad):
                grads[i][:mol.NAtoms()] = mol.properties["gradients"]

        if (self.HasGrad):
            return xyzs, Zs, Elabels, Dlabels, natom, grads, Qlabels, Qtlabels,masks
        else:
            return xyzs, Zs, Elabels, Dlabels, natom, Qlabels, Qtlabels,masks
    
    def LoadDataToScratch(self, tformer):
        """
        Reads built training data off disk into scratch space.
        Divides training and test data.
        Normalizes inputs and outputs.
        note that modifies my MolDigester to incorporate the normalization
        Initializes pointers used to provide training batches.

        Args:
                random: Not yet implemented randomization of the read data.

        Note:
                Also determines mean stoichiometry
        """
        try:
            self.HasGrad
        except:
            self.HasGrad = False
        if (self.ScratchState == 1):
            return

        if (self.HasGrad):
            print('HHHHHHHHHH')
            self.xyzs, self.Zs, self.Elabels, self.Dlabels, self.natom, self.grads, self.Qlabels, self.Qtlabels, self.masks = self.LoadData()
        else:
            self.xyzs, self.Zs, self.Elabels, self.Dlabels, self.natom, self.Qlabels, self.Qtlabels, self.masks  = self.LoadData()

        self.NTestMols = int(self.TestRatio * self.Zs.shape[0])
        self.LastTrainMol = int(self.Zs.shape[0]-self.NTestMols)
        self.NTrain = self.LastTrainMol
        self.NTest = self.NTestMols
        self.test_ScratchPointer = self.LastTrainMol
        self.ScratchPointer = 0
        self.ScratchState = 1
        LOGGER.debug("LastTrainMol in TensorMolData: %i", self.LastTrainMol)
        LOGGER.debug("NTestMols in TensorMolData: %i", self.NTestMols)

        #Ming+++++++++++++++++++++++++++++++++++++++++++++++++++++
        atomcount=np.zeros(len(self.eles))
        self.MeanStoich=np.zeros(len(self.eles))
        for j in range(len(self.eles)):
            for i in range(len(self.xyzs)):
                for k in range(len(self.Zs[i])):
                    if self.Zs[i][k]==self.eles[j]:
                        atomcount[j]=atomcount[j]+1
            self.MeanStoich[j]=np.ceil(atomcount[j]/len(self.xyzs))
        self.MeanNumAtoms=np.sum(self.MeanStoich)
        #Ming-------------------------------------------------------

        return

    def GetTrainBatch(self, ncases):
        if (self.ScratchState == 0):
            self.LoadDataToScratch()
        reset = False
        if (ncases > self.NTrain):
            raise Exception("Insufficent training data to fill a batch"+str(self.NTrain)+" vs "+str(ncases))
        if (self.ScratchPointer+ncases >= self.NTrain):
            self.ScratchPointer = 0
        self.ScratchPointer += ncases
        xyzs     = self.xyzs[self.ScratchPointer-ncases:self.ScratchPointer]
        Zs       = self.Zs[self.ScratchPointer-ncases:self.ScratchPointer]
        Dlabels  = self.Dlabels[self.ScratchPointer-ncases:self.ScratchPointer]
        Elabels  = self.Elabels[self.ScratchPointer-ncases:self.ScratchPointer]
        Qlabels  = self.Qlabels[self.ScratchPointer-ncases:self.ScratchPointer]
        Qtlabels = self.Qtlabels[self.ScratchPointer-ncases:self.ScratchPointer]
        masks    = self.masks[self.ScratchPointer-ncases:self.ScratchPointer]
        natom = self.natom[self.ScratchPointer-ncases:self.ScratchPointer]
        NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
        rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(self.Rr_cut, self.Ra_cut, self.ele, self.elep)
        NLEE = NeighborListSet(xyzs, natom, False, False,  None)
        rad_eep = NLEE.buildPairs(self.Ree_cut)
        if (self.HasGrad):
            return [xyzs, Zs, Elabels, Dlabels, Qlabels, Qtlabels, masks,\
                     self.grads[self.ScratchPointer-ncases:self.ScratchPointer], rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom]
        else:
            return [xyzs, Zs, Elabels, Dlabels, Qlabels, Qtlabels, masks,\
                     rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom]

    def GetTestBatch(self,ncases):
        if (self.ScratchState == 0):
            self.LoadDataToScratch()
        reset = False
        if (ncases > self.NTest):
            raise Exception("Insufficent training data to fill a batch"+str(self.NTest)+" vs "+str(ncases))
        if (self.test_ScratchPointer+ncases > self.Zs.shape[0]):
            self.test_ScratchPointer = self.LastTrainMol
        self.test_ScratchPointer += ncases
        xyzs     = self.xyzs[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
        Zs       = self.Zs[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
        Elabels  = self.Elabels[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
        Dlabels  = self.Dlabels[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
        Qlabels  = self.Qlabels[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
        Qtlabels = self.Qtlabels[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
        masks    = self.masks[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
        natom = self.natom[self.test_ScratchPointer-ncases:self.test_ScratchPointer]
        NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
        rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(self.Rr_cut, self.Ra_cut, self.ele, self.elep)
        NLEE = NeighborListSet(xyzs, natom, False, False,  None)
        rad_eep = NLEE.buildPairs(self.Ree_cut)
        if (self.HasGrad):
            return [xyzs, Zs, Elabels, Dlabels, Qlabels, Qtlabels, masks,\
                     self.grads[self.test_ScratchPointer-ncases:self.test_ScratchPointer], rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom]
        else:
            return [xyzs, Zs, Elabels, Dlabels, Qlabels, Qtlabels, masks,\
                     rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom]

    def GetBatch(self, ncases, Train_=True):
        if Train_:
            return self.GetTrainBatch(ncases)
        else:
            return self.GetTestBatch(ncases)


    def Randomize(self, ti, to, group):
        ti = ti.reshape((ti.shape[0]/group, group, -1))
        to = to.reshape((to.shape[0]/group, group, -1))
        random.seed(0)
        idx = np.random.permutation(ti.shape[0])
        ti = ti[idx]
        to = to[idx]
        ti = ti.reshape((ti.shape[0]*ti.shape[1],-1))
        to = to.reshape((to.shape[0]*to.shape[1],-1))
        return ti, to


    def KRR(self):
        from sklearn.kernel_ridge import KernelRidge
        ti, to = self.LoadData(True)
        print("KRR: input shape", ti.shape, " output shape", to.shape)
        #krr = KernelRidge()
        krr = KernelRidge(alpha=0.0001, kernel='rbf')
        trainsize = int(ti.shape[0]*0.5)
        krr.fit(ti[0:trainsize,:], to[0:trainsize])
        predict  = krr.predict(ti[trainsize:, : ])
        print(predict.shape)
        krr_acc_pred  = np.zeros((predict.shape[0],2))
        krr_acc_pred[:,0] = to[trainsize:].reshape(to[trainsize:].shape[0])
        krr_acc_pred[:,1] = predict.reshape(predict.shape[0])
        np.savetxt("krr_acc_pred.dat", krr_acc_pred)
        print("KRR train R^2:", krr.score(ti[0:trainsize, : ], to[0:trainsize]))
        print("KRR test  R^2:", krr.score(ti[trainsize:, : ], to[trainsize:]))
        return 


    def PrintSampleInformation(self):
        print("From files: ", self.AvailableDataFiles)
        return


    def EvaluateTestBatch(self, desired, predicted, tformer, nmols_=100):
        if (tformer.outnorm != None):
            desired = tformer.UnNormalizeOuts(desired)
            predicted = tformer.UnNormalizeOuts(predicted)
        LOGGER.info("desired.shape "+str(desired.shape)+" predicted.shape "+str(predicted.shape)+" nmols "+str(nmols_))
        LOGGER.info("Evaluating, "+str(len(desired))+" predictions... ")
        if (self.dig.OType=="GoEnergy" or self.dig.OType == "Energy" or self.dig.OType == "AtomizationEnergy"):
            predicted=predicted.flatten()[:nmols_]
            desired=desired.flatten()[:nmols_]
            LOGGER.info( "NCases: "+str(len(desired)))
            #LOGGER.info( "Mean Energy "+str(self.unscld(desired)))
            #LOGGER.info( "Mean Predicted Energy "+str(self.unscld(predicted)))
            for i in range(min(50,nmols_)):
                    LOGGER.info( "Desired: "+str(i)+" "+str(desired[i])+" Predicted "+str(predicted[i]))
            LOGGER.info("MAE "+str(np.average(np.abs(desired-predicted))))
            LOGGER.info("STD "+str(np.std(desired-predicted)))
        else:
            raise Exception("Unknown Digester Output Type.")
        return

    def PrintStatus(self):
        print("self.ScratchState",self.ScratchState)
        print("self.ScratchPointer",self.ScratchPointer)
        #print "self.test_ScratchPointer",self.test_ScratchPointer

    def Save(self):
        self.CleanScratch()
        f=open(self.path+self.name+"_"+self.dig.name+".tdt","wb")
        pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        return


    def BuildTrainMolwise(self, name_="gdb9", atypes=[], append=False, MakeDebug=False):
        """
        Generates inputs for all training data using the chosen digester.
        This version builds all the elements at the same time.
        The other version builds each element separately
        If PESSamples = [] it may use a Go-model (CITE:http://dx.doi.org/10.1016/S0006-3495(02)75308-3)
        """
        if (((self.dig.name != "GauInv" and self.dig.name !="GauSH" and self.dig.name !="ANI1_Sym")) or (self.dig.OType != "GoForce" and self.dig.OType!="GoForceSphere"
                                 and self.dig.OType!="Force" and self.dig.OType!="Del_Force" and self.dig.OType !="ForceSphere" and self.dig.OType !="ForceMag")):
            raise Exception("Molwise Embedding not supported")
        if (self.set == None):
            try:
                self.ReloadSet()
            except Exception as Ex:
                print("TData doesn't have a set.", Ex)
        self.CheckShapes()
        self.name=name_
        LOGGER.info("Generating Train set: %s from mol set %s of size %i molecules", self.name, self.set.name, len(self.set.mols))
        if (len(atypes)==0):
            atypes = self.set.AtomTypes()
        LOGGER.debug("Will train atoms: "+str(atypes))
        # Determine the size of the training set that will be made.
        nofe = [0 for i in range(MAX_ATOMIC_NUMBER)]
        for element in atypes:
            for m in self.set.mols:
                nofe[element] = nofe[element]+m.NumOfAtomsE(element)
        truncto = [nofe[i] for i in range(MAX_ATOMIC_NUMBER)]
        cases_list = [np.zeros(shape=tuple([nofe[element]*self.dig.NTrainSamples]+list(self.dig.eshape)), dtype=np.float64) for element in atypes]
        labels_list = [np.zeros(shape=tuple([nofe[element]*self.dig.NTrainSamples]+list(self.dig.lshape)), dtype=np.float64) for element in atypes]
        casep_list = [0 for element in atypes]
        t0 = time.time()
        ord = len(self.set.mols)
        mols_done = 0
        try:
            for mi in xrange(ord):
                m = self.set.mols[mi]
                ins,outs = self.dig.TrainDigestMolwise(m)
                for i in range(m.NAtoms()):
                    # Route all the inputs and outputs to the appropriate place...
                    ai = atypes.tolist().index(m.atoms[i])
                    cases_list[ai][casep_list[ai]] = ins[i]
                    labels_list[ai][casep_list[ai]] = outs[i]
                    casep_list[ai] = casep_list[ai]+1
                if (mols_done%10000==0 and mols_done>0):
                    print(mols_done)
                if (mols_done==400):
                    print("Seconds to process 400 molecules: ", time.time()-t0)
                mols_done = mols_done + 1
        except Exception as Ex:
                print("Likely you need to re-install MolEmb.", Ex)
        for element in atypes:
            # Write the numpy arrays for this element.
            ai = atypes.tolist().index(element)
            insname = self.path+name_+"_"+self.dig.name+"_"+str(element)+"_in.npy"
            outsname = self.path+name_+"_"+self.dig.name+"_"+str(element)+"_out.npy"
            alreadyexists = (os.path.isfile(insname) and os.path.isfile(outsname))
            if (append and alreadyexists):
                ti=None
                to=None
                inf = open(insname,"rb")
                ouf = open(outsname,"rb")
                ti = np.load(inf)
                to = np.load(ouf)
                inf.close()
                ouf.close()
                try:
                    cases = np.concatenate((cases_list[ai][:casep_list[ai]],ti))
                    labels = np.concatenate((labels_list[ai][:casep_list[ai]],to))
                except Exception as Ex:
                    print("Size mismatch with old training data, clear out trainsets")
                inf = open(insname,"wb")
                ouf = open(outsname,"wb")
                np.save(inf,cases)
                np.save(ouf,labels)
                inf.close()
                ouf.close()
                self.AvailableDataFiles.append([insname,outsname])
                self.AvailableElements.append(element)
                self.SamplesPerElement.append(casep_list[ai]*self.dig.NTrainSamples)
            else:
                inf = open(insname,"wb")
                ouf = open(outsname,"wb")
                np.save(inf,cases_list[ai][:casep_list[ai]])
                np.save(ouf,labels_list[ai][:casep_list[ai]])
                inf.close()
                ouf.close()
                self.AvailableDataFiles.append([insname,outsname])
                self.AvailableElements.append(element)
                self.SamplesPerElement.append(casep_list[ai]*self.dig.NTrainSamples)
        self.Save() #write a convenience pickle.
        return

    def MergeWith(self,ASet_):
        '''
        Augments my training data with another set, which for example may have been generated on another computer.
        '''
        self.QueryAvailable()
        ASet_.QueryAvailable()
        print("Merging", self.name, " with ", ASet_.name)
        for ele in ASet_.AvailableElements:
            if (self.AvailableElements.count(ele)==0):
                raise Exception("WriteME192837129874")
            else:
                mti,mto = self.LoadElement(ele)
                ati,ato = ASet_.LoadElement(ele)
                labelshapes = list(mti.shape)[1:]
                eshapes = list(mto.shape)[1:]
                ASet_labelshapes = list(ati.shape)[1:]
                ASet_eshapes = list(ato.shape)[1:]
                if (labelshapes != ASet_labelshapes or eshapes != ASet_eshapes):
                    raise Exception("incompatible")
                if (self.dig.name != ASet_.dig.name):
                    raise Exception("incompatible")
                print("Merging ", self.name, " element, ", ele ," with ", ASet_.name)
                mti=np.concatenate((mti,ati),axis=0)
                mto=np.concatenate((mto,ato),axis=0)
                print("The new element train set will have", mti.shape[0], " cases in it")
                insname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_in.npy"
                outsname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_out.npy"
                inf = open(insname,"wb")
                ouf = open(outsname,"wb")
                np.save(inf,mti)
                np.save(ouf,mto)
                inf.close()
                ouf.close()

    def Load(self):
        print("Unpickling Tensordata")
        f = open(self.path+self.name+".tdt","rb")
        tmp=pickle.load(f)
        self.__dict__.update(tmp)
        f.close()
        self.CheckShapes()
        print("Training data manager loaded.")
        if (self.set != None):
            print("Based on ", len(self.set.mols), " molecules ")
        print("Based on files: ",self.AvailableDataFiles)
        self.QueryAvailable()
        self.PrintSampleInformation()
        self.dig.Print()
        return

    def LoadElement(self, ele, Random=True, DebugData_=False):
        insname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_in.npy"
        outsname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_out.npy"
        dbgname = self.path+self.name+"_"+self.dig.name+"_"+str(ele)+"_dbg.tdt"
        try:
            inf = open(insname,"rb")
            ouf = open(outsname,"rb")
            ti = np.load(inf)
            to = np.load(ouf)
            inf.close()
            ouf.close()
        except Exception as Ex:
            print("Failed to read:",insname, " or ",outsname)
            raise Ex
        if (ti.shape[0] != to.shape[0]):
            raise Exception("Bad Training Data.")
        if (self.ChopTo!=None):
            ti = ti[:self.ChopTo]
            to = to[:self.ChopTo]
        if (DebugData_):
            print("DEBUGGING, ", len(ti), " cases..")
            f = open(dbgname,"rb")
            dbg=pickle.load(f)
            f.close()
            print("Found ", len(dbg), " pieces of debug information for this element... ")
            for i in range(len(dbg)):
                print("CASE:", i, " was for ATOM", dbg[i][1], " At Point ", dbg[i][2])
                ds=GRIDS.Rasterize(ti[i])
                GridstoRaw(ds, GRIDS.NPts, "InpCASE"+str(i))
                print(dbg[i][0].coords)
                print(dbg[i][0].atoms)
        #ti = ti.reshape((ti.shape[0],-1))  # flat data to [ncase, num_per_case]
        #to = to.reshape((to.shape[0],-1))  # flat labels to [ncase, 1]
        if (Random):
            idx = np.random.permutation(ti.shape[0])
            ti = ti[idx]
            to = to[idx]
        self.ScratchNCase = to.shape[0]
        return ti, to

    def LoadElementToScratch(self,ele,tformer):
        """
        Reads built training data off disk into scratch space.
        Divides training and test data.
        Normalizes inputs and outputs.
        note that modifies my MolDigester to incorporate the normalization
        Initializes pointers used to provide training batches.

        Args:
            random: Not yet implemented randomization of the read data.
        """
        ti, to = self.LoadElement(ele, self.Random)
        if (self.dig.name=="SensoryBasis" and self.dig.OType=="Disp" and self.ExpandIsometriesAltogether):
            print("Expanding the given set over isometries.")
            ti,to = GRIDS.ExpandIsometries(ti,to)
        if (tformer.outnorm != None):
            to = tformer.NormalizeOuts(to)
        if (tformer.innorm != None):
            ti = tformer.NormalizeIns(ti)
        self.NTest = int(self.TestRatio * ti.shape[0])
        self.scratch_inputs = ti[:ti.shape[0]-self.NTest]
        self.scratch_outputs = to[:ti.shape[0]-self.NTest]
        self.scratch_test_inputs = ti[ti.shape[0]-self.NTest:]
        self.scratch_test_outputs = to[ti.shape[0]-self.NTest:]
        self.ScratchState = ele
        self.ScratchPointer=0
        LOGGER.debug("Element "+str(ele)+" loaded...")
        return

    def NTrainCasesInScratch(self):
        if (self.ExpandIsometriesBatchwise):
            return self.scratch_inputs.shape[0]*GRIDS.NIso()
        else:
            return self.scratch_inputs.shape[0]

    def NTestCasesInScratch(self):
        return self.scratch_inputs.shape[0]

    def PrintSampleInformation(self):
        lim = min(len(self.AvailableElements),len(self.SamplesPerElement),len(self.AvailableDataFiles))
        for i in range(lim):
            print("AN: ", self.AvailableElements[i], " contributes ", self.SamplesPerElement[i] , " samples ")
            print("From files: ", self.AvailableDataFiles[i])
        return


