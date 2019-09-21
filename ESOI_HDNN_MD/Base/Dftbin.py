#!/usr/bin/env python
# coding=utf-8

DFTBin="""Geometry=GenFormat{
%s
}

Hamiltonian=DFTB{
    SCC=Yes
    SCCTolerance=1.0e-6
    MaxSCCIterations=10000
    Mixer=Broyden{
        MixingParameter=0.1
#        CacheIterations=-1
        InverseJacobiweight=0.01
        MinimalWeight=1
        MaximalWeight=100000.
        WeightFactor=0.01
    }
    SlaterKosterFiles=Type2FileNames{
        Prefix="%s"
        Separator="-"
        Suffix=".skf"
        LowerCaseTypeName=No
    }

    MaxAngularMomentum={
%s
    }

    charge=   %d
    SpinPolarisation={}
    Filling=Fermi{
        Temperature[k]=300
    }

    OrbitalResolvedSCC=No
    ReadInitialCharges=No
    Eigensolver=DivideAndConquer{}
    OldSKInterpolation=No
    ThirdOrderFull=Yes
    DampXH=Yes
    DampXHExponent=4.00

    HubbardDerivs={
%s
    }
    Dispersion = DftD3{
        Damping = BeckeJohnson{
            a1 = 0.746
            a2 = 4.191
        }
    }
}

Options={
    WriteAutotestTag=Yes
    WriteDetailedXML=Yes
    writeResultsTag=Yes
    RandomSeed=0
}

Analysis={
    CalculateForces=Yes
    WriteEigenvectors=No
    AtomResolvedEnergies=No
    writeBandOut=Yes
}
ParserOptions={
    ParserVersion=5
}
"""
