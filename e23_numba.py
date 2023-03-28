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
    jets_eta = ak.unflatten(ak.flatten(pfcands.eta[fatjetpfcands.pFCandsIdx]),ak.flatten(fatjet.nConstituents))
    jets_phi = ak.unflatten(ak.flatten(pfcands.phi[fatjetpfcands.pFCandsIdx]),ak.flatten(fatjet.nConstituents))   
    e23_jetwise = np.array([])
    for i in range(0, len(jets_pt)):
        peetee = jets_pt[i].to_numpy()
        eeta = jets_eta[i].to_numpy()
        fi = jets_phi[i].to_numpy()
        e23_jetwise = np.append(e23_jetwise, e23_loop(peetee, eeta, fi))
    intermed = ak.from_numpy(e23_jetwise)
    e23_eventwise = ak.unflatten(e23_jetwise, ak.count(fatjet.nConstituents, axis=1))
    return e23_eventwise
