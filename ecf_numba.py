import numpy as np
import awkward as ak
from numba import njit
import awkward.numba
import math

@njit
def e23_loop(pt, eta, phi):
    e23_val = 0
    for i in range(0,len(pt)):
        z_i = (pt[i]/sum(pt))
        for j in range(1, len(pt)):
            if j <= i:
                continue
            if j > i:
                z_j = (pt[j]/sum(pt))
                for k in range(2, len(pt)):
                    if k <= j:
                        continue
                    if k > j:
                        z_k = (pt[k]/sum(pt))
                        delta_rij = math.sqrt((phi[i] - phi[j])**2 + (eta[i] - eta[j])**2)
                        delta_rik = math.sqrt((phi[i] - phi[k])**2 + (eta[i] - eta[k])**2)
                        delta_rjk = math.sqrt((phi[j] - phi[k])**2 + (eta[j] - eta[k])**2)
                        e23_partial = z_i*z_j*z_k*min(delta_rij*delta_rik, delta_rij*delta_rjk, delta_rik*delta_rjk)
                        e23_val += e23_partial
    return e23_val

def e23_numba(fatjetscands, pfcands, fatjets): 
    jets_pt = ak.unflatten(ak.flatten(pfcands.pt[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))
    jets_eta = ak.unflatten(ak.flatten(pfcands.eta[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))
    jets_phi = ak.unflatten(ak.flatten(pfcands.phi[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))   
    e23_jetwise = np.zeros(len(jets_pt))
    for i in range(0, len(jets_pt)):
        peetee = jets_pt[i].to_numpy()
        eeta = jets_eta[i].to_numpy()
        fi = jets_phi[i].to_numpy()
        e23_jetwise[i] = e23_loop(peetee, eeta, fi)
    intermed = ak.from_numpy(e23_jetwise)
    e23_eventwise = ak.unflatten(e23_jetwise, ak.count(fatjets.nConstituents, axis=1))
    return e23_eventwise


@njit
def e2_loop(pt, eta, phi):
    e2_val = 0
    for i in range(0,len(pt)):
        z_i = (pt[i]/sum(pt))
        for j in range(1, len(pt)):
            if j <= i:
                continue
            if j > i:
                z_j = (pt[j]/sum(pt))
                delta_r  = math.sqrt((phi[i] - phi[j])**2 + (eta[i] - eta[j])**2)
                e2_partial = z_i*z_j*delta_r
                e2_val += e2_partial
    return e2_val

def e2_numba(fatjetscands, pfcands, fatjets): 
    jets_pt = ak.unflatten(ak.flatten(pfcands.pt[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))
    jets_eta = ak.unflatten(ak.flatten(pfcands.eta[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))
    jets_phi = ak.unflatten(ak.flatten(pfcands.phi[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))   
    e2_jetwise = np.zeros(len(jets_pt))
    for i in range(0, len(jets_pt)):
        peetee = jets_pt[i].to_numpy()
        eeta = jets_eta[i].to_numpy()
        fi = jets_phi[i].to_numpy()
        e2_jetwise[i] = e2_loop(peetee, eeta, fi)
    intermed = ak.from_numpy(e2_jetwise)
    e2_eventwise = ak.unflatten(e2_jetwise, ak.count(fatjets.nConstituents, axis=1))
    return e2_eventwise