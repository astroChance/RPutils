"""
Elastic moduli and velocity functions
"""

import numpy as np


def shear_mod(vs, dens):
    """
    Calculate shear modulus
    
    Inputs:
        vs (float): s wave velocity in km/s
        dens (float): bulk density in g/cm3
    Returns:
        Shear Modulus (float) in GPa
    """
    dens = np.array(dens)
    vs = np.array(vs)
    u = dens * vs**2
    return u


def bulk_mod(vp, vs, dens):
    """
    Calculate bulk modulus

    Inputs:
        vp (float): p wave velocity in km/s
        vs (float): s wave velocity in km/s
        dens (float): bulk density in g/cm3
    Returns:
        Bulk Modulus (float) in GPa
    """
    vp = np.array(vp)
    vs = np.array(vs)
    dens = np.array(dens)
    k = dens*(vp**2 - ((4/3)*(vs**2)))
    return k


def p_vel(k, mu, dens):
    """
    Calculate P velocity from moduli and density

    Inputs:
        k (float): bulk modulus in GPa
        mu (float): shear modulus in GPa
        dens (float): bulk density in g/cm3
    Returns:
        P wave velocity (float) in km/s
    """
    k = np.array(k)
    mu = np.array(mu)
    dens = np.array(dens)
    p = ((k + (4*mu)/3)/dens) ** 0.5
    return p

def s_vel(mu, dens):
    """
    Calculate S velocity from moduli and density

    Inputs:
        mu (float): shear modulus in GPa
        dens (float): bulk density in g/cm3
    Returns:
        S wave velocity (float) in km/s
    """
    mu = np.array(mu)
    dens = np.array(dens)
    s = (mu/dens) ** 0.5
    return s


def poisson_vel(vp, vs):
    """
    Calculate Poisson's ratio from Vp and Vs
    [Naming is inconsistent with bulk and shear modulus functions]

    Inputs:
        vp (float): p wave velocity in m/s or km/s
        vs (float): p wave velocity in m/s or km/s
    Returns:
        Poisson's ratio (float)
    """
    vp = np.array(vp)
    vs = np.array(vs)
    v = 0.5 * (((vp/vs)**2)-2) / (((vp/vs)**2)-1)
    return v


def poisson_mod(k, mu):
    """
    Calculate Poisson's ration from moduli
    [Naming is inconsistent with bulk and shear modulus functions]

    Inputs:
        k (float): bulk modulus in GPa
        mu (float): shear modulus in GPa
    Returns:
        Poisson's ratio (float)
    """
    k = np.array(k)
    mu = np.array(mu)
    v = (3*k - 2*mu) / (2*(3*k + mu))
    return v
    



def lame_mod(k, mu):
    """
    Calculate Lame's Constant from moduli

    Inputs:
        k (float): bulk modulus in GPa
        mu (float): shear modulus in GPa
    Returns:
        Lame's constant (float)
    """
    k = np.array(k)
    mu = np.array(mu)
    lame = k - (2*mu)/3
    return lame

