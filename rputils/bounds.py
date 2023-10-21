"""
Bounds and mixing laws
"""

import numpy as np
import math

def voight_average(volume_fracts, moduli):
    """
    Calculate Voight average given volume fractions and moduli

    Inputs:
        volume_fracts (list or array): volume fractions of constituents
        moduli (list or array): moduli of constituents

    Returns:
        Voight average (float)
    """
    assert len(volume_fracts) == len(moduli), "Check inputs to Voight"
    assert math.isclose(sum(volume_fracts), 1), "Check volume fractions for Voight"
    
    holder = []
    for f, m in zip(volume_fracts, moduli):
        holder.append(f*m)
    v = sum(holder)
    return v


def modified_voight(mineral_moduli, suspension_moduli, porosity, critical_porosity):
    """
    Calculate modified Voight average between a mineral modulus
    and it's suspension at critical porosity

    Inputs:
        mineral_moduli (float): modulus of solid grain material
        suspension_moduli (float): effective modulus of material at critical porosity
        porosity (float 0-1): porosity at which to calculate the modified Voight average 
        critical_porosity (float 0-1): critical porosity of media

    Returns:
        Modified Voight average (float)
    """
    phi_prime = porosity / critical_porosity
    mv = (1-phi_prime)*mineral_moduli + phi_prime*suspension_moduli
    
    return mv


def reuss_average(volume_fracts, moduli):
    """
    Calculate Reuss average given volume fractions and moduli

    Inputs:
        volume_fracts (list or array): volume fractions of constituents
        moduli (list or array): moduli of constituents

    Returns:
        Reuss average (float)
    """
    assert len(volume_fracts) == len(moduli), "Check inputs to Reuss"
    assert math.isclose(sum(volume_fracts), 1), "Check volume fractions for Reuss"
    
    holder = []
    for f, m in zip(volume_fracts, moduli):
        if m == 0:
            continue
        holder.append(f/m)
    r = (sum(holder))**-1
    return r


def modified_reuss(mineral_moduli, frame_moduli, porosity, critical_porosity):
    """
    Calculate the modified reuss average.
    Taken from Zimmer (2007, Part 2) but seems wonky

    Inputs:
        mineral_moduli (float): effective grain modulus
        frame_moduli (float): frame modulus at pressure of interest
        porosity (float): porosity (0-1)
        critical_porosity (float):  NOTE, AUTHOR STATES THIS ISN'T CRITICAL POROSITY

    Returns:
        Modified Reuss average (float)
    """
    phi_prime = porosity / critical_porosity

    r = ((phi_prime/frame_moduli) + ((1-phi_prime)/mineral_moduli))**-1

    return r


def hill_average(volume_fracts, moduli):
    """
    Hill average between Voight and Reuss averages

    Inputs:
        volume_fracts (list or array): volume fractions of constituents
        moduli (list or array): moduli of constituents

    Returns:
        Hill average (float)
    """
    assert len(volume_fracts) == len(moduli), "Check inputs to Hill"
    assert math.isclose(sum(volume_fracts), 1), "Check volume fractions for Hill"
    
    h = 0.5 * (voight_average(volume_fracts, moduli) + reuss_average(volume_fracts, moduli))
    return h


def modified_hill_average(volume_fracts, moduli, porosity, critical_porosity):
    """
    BROKEN: modified Voight assumes it is given the single average
    moduli value, calling min and max on moduli likely a bas idea.
    Likely don't need this function, come back to it later if required.
    
    Hill average between Modified Voight and Reuss averages.
    
    Assumes higher modulus is mineral, lower is suspension
    """
    assert len(volume_fracts) == len(moduli), "Check inputs to Modified Hill"
    assert math.isclose(sum(volume_fracts), 1), "Check volume fractions for Modified Hill"
    
    mineral_moduli = max(moduli)
    suspension_moduli = min(moduli)
    
    h = 0.5 * (modified_voight(mineral_moduli, suspension_moduli, porosity, critical_porosity) \
               + reuss_average(volume_fracts, moduli))
    return h


def woods_vel(volume_fracts, moduli, densities):
    """
    Compute P-velocity of a suspension based on Wood's formula

    Inputs:
        volume_fracts (list or array): volume fractions of constituents
        moduli (list or array): moduli of constituents
        densities (list or array): densities of constituents

    Returns:
        Velocity (float)
    """
    mix_density = sum([f*d for f, d in zip(volume_fracts, densities)])
    
    vel = np.sqrt(reuss_average(volume_fracts, moduli) / mix_density)
    
    return vel



def hs(bound, volume_fracts, bulk_mods, shear_mods, porosity=0):
    """
    Compute Hashin-Shtrikman bounds for multi-component mixture
    
    If using a pore fluid, set volume fraction to 0 and use porosity value.
    Assumes shear modulus of fluid is 0
    
    Inputs
        bound (string): 'upper' or 'lower'
        volume_fracts (list or array): component fractions
        bulk_mods (list or array): components bulk moduli
        shear_mods (list or array): components shear moduli
        porosity (float 0-1): porosity value if including fluids
        
    Returns
        Bulk and Shear bounds in GPa
        
    """
    if not isinstance(volume_fracts, np.ndarray):
        volume_fracts = np.array(volume_fracts)
    if not isinstance(bulk_mods, np.ndarray):
        bulk_mods = np.array(bulk_mods)
    if not isinstance(shear_mods, np.ndarray):
        shear_mods = np.array(shear_mods)
    
    assert volume_fracts.size == bulk_mods.size == shear_mods.size, "Check inputs to HS-Bound"
    assert sum(volume_fracts) == 1, "Check volume fractions for HS-Bound"
    assert bound.lower() == "upper" or bound.lower() == "lower", "Input correct Bound keyword (upper/lower)"
    
    fluid_loc = np.where(shear_mods==0)[0]
    
    if bound.lower() == "upper":
        try:
            km = max(bulk_mods)
            um = max(shear_mods)
        except:
            km = max([bulk_mods])
            um = max([shear_mods])
        
    if bound.lower() == "lower":
        try:
            km = min(bulk_mods)
            um = min(shear_mods)
        except:
            km = min([bulk_mods])
            um = min([shear_mods])
        
    zeta = (um/6) * ((9*km+8*um)/(km+2*um))
    
    k_holder = []
    u_holder = []

    try:
        for k, u, v in zip(bulk_mods, shear_mods, volume_fracts):
            if fluid_loc.size>0 and u == shear_mods[fluid_loc]:
                k_val = porosity / (k + (4/3)*um)
                if bound.lower() == "upper":
                    u_val = porosity / zeta
                else:
                    u_val = 0
            else:
                k_val = ((1-porosity)*(v)) / (k + (4/3)*um)
                if bound.lower() == "upper":
                    u_val = ((1-porosity)*(v)) / (u + zeta)
                else:
                    if fluid_loc.size>0:
                        u_val = 0
                    else:
                        u_val = ((1-porosity)*(v)) / (u + zeta)
            k_holder.append(k_val)
            u_holder.append(u_val)
    except:
        for k, u, v in zip([bulk_mods], [shear_mods], [volume_fracts]):
            if fluid_loc.size>0 and u == shear_mods[fluid_loc]:
                k_val = porosity / (k + (4/3)*um)
                if bound.lower() == "upper":
                    u_val = porosity / zeta
                else:
                    u_val = 0
            else:
                k_val = ((1-porosity)*(v)) / (k + (4/3)*um)
                if bound.lower() == "upper":
                    u_val = ((1-porosity)*(v)) / (u + zeta)
                else:
                    if fluid_loc.size>0:
                        u_val = 0
                    else:
                        u_val = ((1-porosity)*(v)) / (u + zeta)
            k_holder.append(k_val)
            u_holder.append(u_val)
        
        
    hs_k = (sum(k_holder)**-1) - (4/3) * um
    
    try:
        hs_u = (sum(u_holder)**-1) - zeta
    except ZeroDivisionError:
        hs_u = 0
    
    
    return hs_k, hs_u


def modified_hs_upper(k1, k2, u1, u2, porosity, critical_porosity):
    """
    Calculate the modified upper Hashin-Shtrikman bound that connects
    the mineral point with the effective moduli at critical porosity
    
    Make assumption that k1 is mineral, k2 is suspension

    Inputs:
        k1 (float): bulk modulus of mineral in GPa
        k2 (float): bulk modulus of effective media at critical porosity in GPa
        u1 (float): shear modulus of mineral in GPa
        u2 (float): shear modulus of effective media at critical porosity in GPa
        porosity (float 0-1): porosity at which to calculate boundary value
        critical_porosity (float 0-1): critical porosity of media

    Returns:
        Bulk and Shear bounds in GPa
    """
    
    km = k1
    um = u1
    phi_prime = porosity / critical_porosity
    
    k = k1 + (phi_prime/((k2-k1)**-1 + ((1-phi_prime)*(k1+(4/3)*um)**-1)))
    u = u1 + (phi_prime/((u2-u1)**-1 + ((1-phi_prime)*(u1+(1/6)*um*((9*km+8*um)/(km+2*um)))**-1)))
    
    return k, u

