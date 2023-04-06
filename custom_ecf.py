import numpy as np
import awkward as ak

def e2(fatjetscands, pfcands, fatjets):
    jets_pt = ak.unflatten(ak.flatten(pfcands.pt[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))
    sums = ak.sum(jets_pt, axis=1)
    z = (jets_pt/sums)
    
    del jets_pt, sums

    eta = ak.unflatten(ak.flatten(pfcands.eta[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))

    phi = ak.unflatten(ak.flatten(pfcands.phi[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))

    z_comb = ak.combinations(z, 2, axis=1)
    z_ij = np.multiply(z_comb['0'],z_comb['1'])
    
    del z, z_comb

    eta_comb = ak.combinations(eta, 2, axis=1)
    eta_diff = np.square(eta_comb['0']-eta_comb['1'])
    
    del eta, eta_comb

    phi_comb = ak.combinations(phi, 2, axis=1)
    phi_diff = np.square(phi_comb['0']-phi_comb['1'])
    
    del phi, phi_comb

    delta_r = np.sqrt(eta_diff+phi_diff)
    product = np.multiply(delta_r,z_ij)

    del delta_r, z_ij, phi_diff, eta_diff
    
    e2_jetwise = ak.sum(product, axis=1)
    e2_eventwise = ak.unflatten(e2_jetwise, ak.count(fatjets.nConstituents, axis=1))
    
    del e2_jetwise, product
    
    return e2_eventwise


def distance(array, a, b):
    return np.sqrt(np.square(array['phi'][a] - array['phi'][b])
            + np.square(array['eta'][a] - array['eta'][b]))


def e23(fatjetscands, pfcands, fatjets):
    jets_pt = ak.unflatten(ak.flatten(pfcands.pt[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))
    sums = ak.sum(jets_pt, axis=1)
    z = (jets_pt/sums)
    
    del jets_pt, sums

    eta = ak.unflatten(ak.flatten(pfcands.eta[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))

    phi = ak.unflatten(ak.flatten(pfcands.phi[fatjetscands.pFCandsIdx]),ak.flatten(fatjets.nConstituents))

    z_comb = ak.combinations(z, 3, axis=1)
    z_ijk = z_comb['0']*z_comb['1']*z_comb['2']
    
    del z, z_comb

    eta_comb = ak.combinations(eta, 3, axis=1)
    phi_comb = ak.combinations(phi, 3, axis=1)
    
    del eta, phi

    coords = ak.zip({'phi': phi_comb, 'eta': eta_comb})
    
    del eta_comb, phi_comb

    a = distance(coords, '0', '1')
    b = distance(coords, '0', '2')
    c = distance(coords, '1', '2')
    
    del coords

    ij_ik = a*b
    ij_jk = a*c
    ik_jk = b*c
    
    del a, b, c
    
    del_comb = ak.flatten(ak.zip({'ijik': ij_ik, 'ijjk': ij_jk, 'ikjk': ik_jk}))
    
    del ij_ik, ij_jk, ik_jk
    
    temp_numpy = np.vstack((del_comb.ijik.to_numpy(),
        del_comb.ijjk.to_numpy(), 
        del_comb.ikjk.to_numpy())).T
    
    del del_comb
    
    delta_r = ak.unflatten(np.amin(temp_numpy,axis=1), ak.count(z_ijk,axis=1))

    del temp_numpy

    e23_jetwise = ak.sum(np.multiply(z_ijk, delta_r), axis=1)
    
    del z_ijk, delta_r
    
    e23_eventwise = ak.unflatten(e23_jetwise, ak.count(fatjets.nConstituents, axis=1))
    
    del e23_jetwise
    
    return e23_eventwise
