import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
from sympy import symbols, Eq, solve, nsolve, solveset, S
import sys


## Functions

##----------------------------------------------------
## Elastic properties

def lower_murphy(por):
    """
    Lower Murphy bound for estimating coordination
    number from porosity (Murphy, 1982, Effects of 
    microstructure and pore fluids on the acoustic 
    properties of granular sedimentary materials)
    
    Inputs:
        por (float): porosity (0-1)
    Returns:
        coordination number (float)
    """
    n = 17.34 - 34*por + 14*(por**2)
    return n


def upper_murphy(por):
    """
    Upper Murphy bound for estimating coordination
    number from porosity (Murphy, 1982, Effects of 
    microstructure and pore fluids on the acoustic 
    properties of granular sedimentary materials)
    
    Inputs:
        por (float): porosity (0-1)
    Returns:
        coordination number (float)
    """
    n = 20 - 34*por + 14*(por**2)
    return n


def poisson_vel(vp, vs):
    """
    Calculate Poisson's ratio from Vp and Vs

    Inputs:
        vp (float): p wave velocity in m/s or km/s
        vs (float): p wave velocity in m/s or km/s
    Returns:
        Poisson's ratio (float)
    """
    v = 0.5 * (((vp/vs)**2)-2) / (((vp/vs)**2)-1)
    return v


def poisson_mod(k, mu):
    """
    Calculate Poisson's ration from moduli

    Inputs:
        k (float): bulk modulus in GPa
        mu (float): shear modulus in GPa
    Returns:
        Poisson's ratio (float)
    """
    v = (3*k - 2*mu) / (2*(3*k + mu))
    return v
    

def shear_mod(vs, dens):
    """
    Calculate shear modulus
    [Yes this naming convention is opposite of other functions]
    
    Inputs:
        vs (float): s wave velocity in km/s
        dens (float): bulk density in g/cm3
    Returns:
        Shear Modulus (float) in GPa
    """
    u = dens * vs**2
    return u


def bulk_mod(vp, vs, dens):
    """
    Calculate bulk modulus
    [Yes this naming convention is opposite of other functions]

    Inputs:
        vp (float): p wave velocity in km/s
        vs (float): s wave velocity in km/s
        dens (float): bulk density in g/cm3
    Returns:
        Bulk Modulus (float) in GPa
    """
    k = dens*(vp**2 - ((4/3)*(vs**2)))
    return k


def p_vel_mod(k, mu, dens):
    """
    Calculate P velocity from moduli and density

    Inputs:
        k (float): bulk modulus in GPa
        mu (float): shear modulus in GPa
        dens (float): bulk density in g/cm3
    Returns:
        P wave velocity (float) in km/s
    """
    p = ((k + (4*mu)/3)/dens) ** 0.5
    return p

def s_vel_mod(mu, dens):
    """
    Calculate S velocity from moduli and density

    Inputs:
        mu (float): shear modulus in GPa
        dens (float): bulk density in g/cm3
    Returns:
        S wave velocity (float) in km/s
    """
    s = (mu/dens) ** 0.5
    return s

def lame_mod(k, mu):
    """
    Calculate Lame's Constant from moduli

    Inputs:
        k (float): bulk modulus in GPa
        mu (float): shear modulus in GPa
    Returns:
        Lame's constant (float)
    """
    lame = k - (2*mu)/3
    return lame


##----------------------------------------------------
## Bounds and mixing

def voight_average(volume_fracts, moduli):
    """
    Calculate Voight average given volume fractions and moduli
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
    """
    phi_prime = porosity / critical_porosity
    mv = (1-phi_prime)*mineral_moduli + phi_prime*suspension_moduli
    
    return mv


def reuss_average(volume_fracts, moduli):
    """
    Calculate Reuss average given volume fractions and moduli
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
    """
    phi_prime = porosity / critical_porosity

    r = (phi_prime/frame_moduli) + ((1-phi_prime)/mineral_moduli)

    return r**-1


def hill_average(volume_fracts, moduli):
    """
    Hill average between Voight and Reuss averages
    """
    assert len(volume_fracts) == len(moduli), "Check inputs to Hill"
    assert math.isclose(sum(volume_fracts), 1), "Check volume fractions for Hill"
    
    h = 0.5 * (voight_average(volume_fracts, moduli) + reuss_average(volume_fracts, moduli))
    return h


def modified_hill_average(volume_fracts, moduli, porosity, critical_porosity):
    """
    Hill average between Modified Voight and Reuss averages
    
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
    """
    mix_density = sum([f*d for f, d in zip(volume_fracts, densities)])
    
    vel = np.sqrt(reuss_average(volume_fracts, moduli) / mix_density)
    
    return vel



def hs_bound(bound, volume_fracts, bulk_mods, shear_mods, porosity=0):
    """
    Compute Hashin-Shtrikman bounds for multi-component mixture
    
    If using a pore fluid, set volume fraction to 0 and use porosity value.
    Assumes shear modulus of fluid is 0
    
    Inputs
        bound: string, 'upper' or 'lower'
        volume_fracts: list or array of component fractions
        bulk_mods: list or array of component bulk moduli
        shear_mods: list of array of component shear moduli
        porosity
        
    Returns
        Bulk and Shear bounds in GPa
        
    """
    assert len(volume_fracts) == len(bulk_mods) == len(shear_mods), "Check inputs to HS-Bound"
    assert sum(volume_fracts) == 1, "Check volume fractions for HS-Bound"
    assert bound.lower() == "upper" or bound.lower() == "lower", "Input correct Bound keyword (upper/lower)"
    
    volume_fracts = np.array(volume_fracts)
    bulk_mods = np.array(bulk_mods)
    shear_mods = np.array(shear_mods)
    
    fluid_loc = np.where(shear_mods==0)[0]
    
    if bound.lower() == "upper":
        km = max(bulk_mods)
        um = max(shear_mods)
        
    if bound.lower() == "lower":
        km = min(bulk_mods)
        um = min(shear_mods)
        
    zeta = (um/6) * ((9*km+8*um)/(km+2*um))
    
    k_holder = []
    u_holder = []
        
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
        
        
    hs_k = (sum(k_holder)**-1) - (4/3) * um
    
    try:
        hs_u = (sum(u_holder)**-1) - zeta
    except ZeroDivisionError:
        hs_u = 0
    
    
    return hs_k, hs_u


def modified_hs_upper(k1, k2, u1, u2, porosity, critical_porosity):
    """
    Make assumption that k1 is mineral, k2 is suspension
    """
    
    km = k1
    um = u1
    phi_prime = porosity / critical_porosity
    
    k = k1 + (phi_prime/((k2-k1)**-1 + ((1-phi_prime)*(k1+(4/3)*um)**-1)))
    u = u1 + (phi_prime/((u2-u1)**-1 + ((1-phi_prime)*(u1+(1/6)*um*((9*km+8*um)/(km+2*um)))**-1)))
    
    return k, u


    
##----------------------------------------------------
## Models

def hertz_mindlin(k_grain, mu_grain, crit_por, C, pressure, f=1):
    """
    Effective moduli, Hertz-Mindlin
    RP Handbook (2nd) pg.258
    
    Inputs
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        crit_por: critical porosity (0-1)
        C: coordination number
        pressure: effective pressure in GPa
        f: friction coefficient, fraction of grains with perfect cohesion (0-1)
            Note: f=1 == standard Hertz Mindlin
        
    Returns
        Effective bulk modulus in GPa
        Effective shear modulus in GPa
    """
    poisson_grain = poisson_mod(k_grain, mu_grain)
    
    k = ((((C**2)*((1-crit_por)**2)*(mu_grain**2)) / ((18*(np.pi**2))*((1-poisson_grain)**2))) * pressure)**(1/3)
    
    u = ((2+3*f-poisson_grain*(1+3*f))/(5*(2-poisson_grain))) * \
        ((((3*C**2)*((1-crit_por)**2)*(mu_grain**2)) \
         / ((2*(np.pi**2))*((1-poisson_grain)**2))) * pressure)**(1/3)
    
    return k, u
    
    

def bachrach_angular_old(k_grain, mu_grain, por, C, pressure, Rc_ratio, 
                          cohesionless_percent=0, Rg=1):
    """
    Method from Bachran et al. 2000 to control radii of curvature between
    grains to match measured observations from beach sand. Roughly
    corresponds with RP Handbook page 246-249. Returns the same values
    as other HM function when Rc_ratio and cohensionless_percent are
    both set to 1
    
    Inputs
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        por: porosity (0-1)
        C: coordination number
        pressure: effective pressure in GPa
        Rc_ratio: describes angularity of grains, 1=spherical
        cohesionless_percent: percent of grains with frictionless contacts.
            This may be a fitting parameter for shallow loose sands.
            HS- will be used to mix moduli
        Rg: grain radius (seems to be negligible)
        
    Returns
        Effective bulk modulus in GPa
        Effective shear modulus in GPa
    """
    assert Rc_ratio <= 1, "Rc_ratio should be value (0-1)"
    assert cohesionless_percent <= 1, "Rc_ratio should be value (0-1)"
    
    poisson_grain = poisson_mod(k_grain, mu_grain)
    
    F = (4 * np.pi * (Rg**2) * pressure) / (C * (1-por))
    
    a = ((3 * F * Rc_ratio * (1-poisson_grain)) / (8 * mu_grain))**(1/3)
    
    Sn = (4*a*mu_grain)/(1-poisson_grain)
    St = (8*a*mu_grain)/(2-poisson_grain)
    
    khm = ((C*(1-por))/(12*np.pi*Rg))*Sn
    uhm = ((C*(1-por))/(20*np.pi*Rg))*(Sn + 1.5*St)
    
    if cohesionless_percent != 0:
        
        uhm_co = ((C*(1-por))/(20*np.pi*Rg))*(Sn)
        volume_fracts = [1-cohesionless_percent, cohesionless_percent]
        bulk_mods = [khm, khm]
        shear_mods = [uhm, uhm_co]
        
        khm, uhm = hs_bound('lower', volume_fracts, bulk_mods, shear_mods)
    
    return khm, uhm

def bachrach_angular(k_grain, mu_grain, por, C, pressure, c_ratio, 
                          slip_percent=0, Rg=1):
    """
    Method from Bachran and Avseth 2008 
    
    Inputs
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        por: porosity (0-1)
        C: coordination number
        pressure: effective pressure in GPa
        c_ratio: describes contact angularity of grains, 1=spherical
        slip_percent: percent of slipping grain contacts
        Rg: grain radius (seems to be negligible)
        
    Returns
        Effective bulk modulus in GPa
        Effective shear modulus in GPa
    """
    assert c_ratio <= 1, "c_ratio should be value (0-1)"
    assert slip_percent <= 1, "Rc_ratio should be value (0-1)"

    ## Fixing "slip percent" since it should actually be "no-slip percent"
    slip_percent = 1-slip_percent
    
    poisson_grain = poisson_mod(k_grain, mu_grain)
    
    keff = (((1-por)**2*mu_grain**2)/(18*np.pi**2*(1-poisson_grain)**2))**(1/3)*\
            (C**2*c_ratio)**(1/3)*pressure**(1/3)

    ueff = ((1/10)*((12*(1-por)**2*mu_grain**2)/(np.pi**2*(1-poisson_grain)**2))**(1/3)*\
           (C**2*c_ratio)**(1/3)*pressure**(1/3)) + ((3/10)*((12*(1-por)**2*mu_grain**2*\
            (1-poisson_grain))/(np.pi**2*(2-poisson_grain)**3))**(1/3) * (C**2*c_ratio)**(1/3)*\
            pressure**(1/3)) * slip_percent

                              
    return keff, ueff


def digby(k_grain, mu_grain, por, C, pressure, bond_ratio=0.01):
    """
    Compute effective moduli using the Digby model
    RP Handbook (2nd) pg.249

    Inputs
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        por: porosity (0-1)
        C: coordination number
        pressure: effective pressure in GPa
        bond_ratio: ratio of bonded radius to grain radius

    Returns
        Effective bulk modulus in GPa
        Effective shear modulus in GPa
    """

    poisson_grain = poisson_mod(k_grain, mu_grain)

    dvar = symbols("dvar")
    eq = Eq((dvar**3 + (3/2)*(bond_ratio**2)*dvar - (3*np.pi*(1-poisson_grain)*pressure)
             /(2*C*(1-por)*mu_grain)), 0)
    d = float(solveset(eq, "dvar", domain=S.Reals).args[0])

    bR = np.sqrt(d**2 + bond_ratio**2)
    SnR = (4*mu_grain*bR)/(1-poisson_grain)
    StR = (8*mu_grain*bond_ratio)/(2-poisson_grain)
    
    keff = ((C*(1-por))/(12*np.pi))*SnR
    ueff = ((C*(1-por))/(20*np.pi))*(SnR + 1.5*StR)

    return keff, ueff


def walton(k_grain, mu_grain, por, C, pressure, mode):
    """
    Compute effective moduli using the Walton model
    RP Handbook (2nd) pg.248

    Inputs
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        por: porosity (0-1)
        C: coordination number
        pressure: effective pressure in GPa
        mode: can be "rough" or "smooth"

    Returns
        Effective bulk modulus in GPa
        Effective shear modulus in GPa
    """
    assert mode.lower()=="rough" or mode.lower()=="smooth", "Input proper mode argument"
    
    lame_grain = lame_mod(k_grain, mu_grain)
    A = (1/(4*np.pi))*((1/mu_grain)-(1/(mu_grain+lame_grain)))
    B = (1/(4*np.pi))*((1/mu_grain)+(1/(mu_grain+lame_grain)))

    if mode.lower()=="rough":
        keff = (1/6)*((3*(1-por)**2*C**2*pressure)/(np.pi**4*B**2))**(1/3)
        ueff = (3/5)*keff*((5*B+A)/(2*B+A))

    elif mode.lower()=="smooth":
        ueff = (1/10)*((3*(1-por)**2*C**2*pressure)/(np.pi**4*B**2))**(1/3)
        keff = (5/3)*ueff

    return keff, ueff


def uncemented_model(por, crit_por, Khm, Uhm, k_grain, mu_grain):
    """
    Effective moduli, uncemented sand model. Uses a modified
    HS lower bound to evaluate effective moduli at a porosity
    (por) compared to the starting framework (crit_por).
    RP Handbook (2nd) pg.258
    
    Inputs
        por: Porosity for which solving for Effective Bulk Modulus
        crit_por: critical porosity 
        Khm: Hertz-Mindlin bulk modulus
        Uhm: Hertz-Mindlin shear modulus
        k_grain: grain bulk modulus in GPa
        
    Returns
        Effective bulk modulus in GPa 
        Effective shear modulus in GPa 
    """
    
    k = ((((por/crit_por)/(Khm+((4/3)*Uhm)))+((1-(por/crit_por))/(k_grain+((4/3)*Uhm))))**(-1)) - ((4/3)*Uhm)
    
    third_term = ((9*Khm+8*Uhm)/(Khm+2*Uhm))
    first_num = por / crit_por
    first_denom = Uhm + (Uhm/6)*third_term
    second_num = 1 - (por / crit_por)
    second_denom = mu_grain + (Uhm/6)*third_term
    u = (((first_num/first_denom)+(second_num/second_denom))**-1) - (Uhm/6)*third_term
    
    return k, u
    


def stiff_model(por, crit_por, Khm, Uhm, k_grain, mu_grain):
    """
    Effective moduli, stiff sand model. Uses a modified
    HS uper bound to evaluate effective moduli at a porosity
    (por) compared to the starting framework (crit_por).
    RP Handbook (2nd) pg.260
    
    Inputs
        por: Porosity for which solving for Effective Bulk Modulus
        crit_por: critical porosity 
        Khm: Hertz-Mindlin bulk modulus
        Uhm: Hertz-Mindlin shear modulus
        k_grain: grain bulk modulus in GPa
        
    Returns
        Effective bulk modulus in GPa 
        Effective shear modulus in GPa 
    """
    
    k = ((((por/crit_por)/(Khm+((4/3)*mu_grain)))+((1-(por/crit_por))/(k_grain+((4/3)*mu_grain))))**(-1)) \
        - ((4/3)*mu_grain)
    
    third_term = ((9*k_grain+8*mu_grain)/(k_grain+2*mu_grain))
    first_num = por / crit_por
    first_denom = Uhm + (mu_grain/6)*third_term
    second_num = 1 - (por / crit_por)
    second_denom = mu_grain + (mu_grain/6)*third_term
    u = (((first_num/first_denom)+(second_num/second_denom))**-1) - (mu_grain/6)*third_term
    
    return k, u



def cement_radius_ratio(scheme, uncemented_porosity, cemented_porosity, C):
    """
    Scheme can be "contact" or "uniform" (see RP Handbook [2nd] pg. 257)
    """
    assert scheme.lower()=='contact' or scheme.lower()=='uniform', "Check 'Scheme' input for cement radius"
#     assert cemented_porosity<=uncemented_porosity, "Check porosity inputs for cement radius"
#     if cemented_porosity > uncemented_porosity:
#         cemented_porosity = uncemented_porosity
#         warnings.warn("Provided invalid porosity values to cement radius", stacklevel=2)
    
    if scheme.lower()=='contact':
        a = 2*((uncemented_porosity-cemented_porosity)/(3*C*(1-uncemented_porosity)))**(0.25)
    
    if scheme.lower()=='uniform':
        a = ((2*(uncemented_porosity-cemented_porosity))/(3*(1-uncemented_porosity)))**0.5
        
    return a



def cemented_model(k_cement, mu_cement, k_grain, mu_grain, 
                  por, crit_por, C, scheme):
    """
    Calculate the Effective Moduli from the cemented sand model
    RP Handbook (2nd) pg.256
    
    Inputs
        k_cement: bulk modulus of cement in GPa
        mu_cement: shear modulus of cement in GPa
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        por: porosity after cementation
        crit_por: critical porosity of uncemented matrix
        C: coordination number
        scheme: cementation scheme ('contact' or 'uniform')
        
    Returns
        Effective bulk modulus in GPa for cemented sand model
        Effective shear modulus in GPa for cemented sand model    
    """
    
    poisson_cement = poisson_mod(k_cement, mu_cement)
    poisson_grain = poisson_mod(k_grain, mu_grain)
    alpha = cement_radius_ratio(scheme, crit_por, por, C)
    
    lambda_n = ((2*mu_cement)/(np.pi*mu_grain))*(((1-poisson_grain)*(1-poisson_cement))/(1-2*poisson_cement))
    Cn = 0.00024649*lambda_n**-1.9864
    Bn = 0.20405*lambda_n**-0.89008
    An = -0.024153*lambda_n**-1.3646
    Sn = An*alpha**2 + Bn*alpha + Cn
    Mc = (k_cement+(4/3)*mu_cement)
    k = (1/6)*C*(1-crit_por)*Mc*Sn
    
    lambda_t = mu_cement/(np.pi*mu_grain)
    Ct = (10**-4)*(9.654*poisson_grain**2+4.945*poisson_grain+3.1) \
        *lambda_t**(0.01867*poisson_grain**2+0.4011*poisson_grain-1.8186)
    Bt = (0.0573*poisson_grain**2+0.0937*poisson_grain+0.202) \
        *lambda_t**(0.0274*poisson_grain**2+0.0529*poisson_grain-0.8765)
    At = (-10**-2)*(2.26*poisson_grain**2+2.07*poisson_grain+2.3) \
        *lambda_t**(0.079*poisson_grain**2+0.1754*poisson_grain-1.342)
    St = At*alpha**2 + Bt*alpha + Ct
    u = (3/5)*k + (3/20)*C*(1-crit_por)*mu_cement*St
    
    return k, u



def self_consistent_contact_cement(k_cement, mu_cement, k_grain, mu_grain, 
                  por, crit_por, C, scheme, inclusion_shape='spheres',
                  crack_aspect_ratio=10e-2):
    """
    Combine self-consistent effective medium modeling with Contact Cement
    model to calculate effective moduli over full porosity range.
    Reference: Dvorkin et al. 1999
    
    Currently written for matrix, cement, and void phases. Only handles one 
    inclusion shape for both cement inclusion and voids
    
    !! UPDATE THIS TO TAKE ASPECT RATIO FOR VOIDS SEPERATE FROM CEMENT
    
    Inputs
        k_cement: bulk modulus of cement in GPa
        mu_cement: shear modulus of cement in GPa
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        por: porosity transition from cementing to pore-filling
        crit_por: critical porosity of uncemented matrix
        C: coordination number
        scheme: cementation scheme ('contact' or 'uniform')
        inclusion_shape: shape of included cement and voids
        crack_aspect_ratio: shape parameter for cement and voids
        
    Returns
        Array of effective bulk moduli in GPa for cemented sand model
        Array of effective shear moduli in GPa for cemented sand model 
        Porosity values corresponding to moduli
    """
    
    allowable_inclusions = ['spheres', 'disks', 'penny cracks']
    assert inclusion_shape.lower() in allowable_inclusions, "Inclusion shape not supported"


    ## Helper functions for P and Q coefficients
    ## RP handbook, page 187
    def z_var(k, u):
        z = (u*(9*k + 8*u)) / (6*(k + 2*u))
        return z
    
    def b_var(k,u):
        b = u * ((3*k + u)/(3*k + 4*u))
        return b

    def q_var(ki, ui, km, um, shape):
        
        if shape.lower()=='spheres':
            q = (um + z_var(km, um)) / (ui + z_var(km, um))
            
        if shape.lower()=='disks':
            q = (um + z_var(ki, ui)) / (ui + z_var(ki, ui))
            
        if shape.lower()=='penny cracks':
            q = (1/5) * (1 + (8*um / (4*ui + np.pi*crack_aspect_ratio*(um + 2*b_var(km, um)))) + \
                         2*((ki + 2/3*(ui + um))/(ki + 4/3*ui + np.pi*crack_aspect_ratio*b_var(km, um))))
            
        return q

    def p_var(ki, ui, km, um, shape):
        
        if shape.lower()=='spheres':
            p = (km + (4/3)*um) / (ki + (4/3)*um)
            
        if shape.lower()=='disks':
            p = (km + (4/3)*ui) / (ki + (4/3)*ui)
            
        if shape.lower()=='penny cracks':
            p = (km + (4/3)*ui) / (ki + (4/3)*ui + np.pi*crack_aspect_ratio*b_var(km, um))
            
        return p
    
    
    ## Get values from contact cement model
    k_contact, u_contact = [],[]
    cemented_por_range = np.arange(por,crit_por,0.01)
    for por_val in cemented_por_range:
        k_cem, u_cem = cemented_model(k_cement, mu_cement, k_grain, mu_grain, 
                  por_val, crit_por, C, scheme)
        k_contact.append(k_cem)
        u_contact.append(u_cem)
    
    
    ## Use self-consistent model to to calculate moduli between
    ## zero porosity and cemented fitting porosity
    k_cem, u_cem = cemented_model(k_cement, mu_cement, k_grain, mu_grain, 
                  por, crit_por, C, scheme)
    zs = z_var(k_cem, u_cem)
    ks = (((1/(k_cem + (4/3)*u_cem)) - (por / ((4/3)*u_cem)))**-1) * (1 - por) - (4/3)*u_cem
    us = (((1 / (u_cem + zs)) - (por / zs))**-1) * (1 - por) - zs

    k_fill, u_fill = hs_bound(bound='lower', volume_fracts=[crit_por, 1-crit_por], 
                              bulk_mods = [k_cement, k_grain], shear_mods = [mu_cement, mu_grain], porosity=0)

    error_tolerance = 0.005

    while True:
        target_k = (por * (k_cement - k_fill) * p_var(k_cement, mu_cement, k_fill, u_fill, inclusion_shape)) / \
                    ((1 - por) * p_var(ks, us, k_fill, u_fill, inclusion_shape)) + ks

        if target_k > voight_average([crit_por, 1-crit_por], [k_grain, k_cement]):
            target_k = voight_average([crit_por, 1-crit_por], [k_grain, k_cement])
        if target_k < reuss_average([crit_por, 1-crit_por], [k_grain, k_cement]):
            target_k = reuss_average([crit_por, 1-crit_por], [k_grain, k_cement])

        target_u = (por * (mu_cement - u_fill) * q_var(k_cement, mu_cement, k_fill, u_fill, inclusion_shape)) / \
                ((1 - por) * q_var(ks, us, k_fill, u_fill, inclusion_shape)) + us

        if target_u > voight_average([crit_por, 1-crit_por], [mu_grain, mu_cement]):
            target_u = voight_average([crit_por, 1-crit_por], [mu_grain, mu_cement])
        if target_u < reuss_average([crit_por, 1-crit_por], [mu_grain, mu_cement]):
            target_u = reuss_average([crit_por, 1-crit_por], [mu_grain, mu_cement])

        if (np.abs(target_k - k_fill) > error_tolerance) and (np.abs(target_u - u_fill) > error_tolerance):
            k_fill = target_k
            u_fill = target_u
            continue
        else:
            k_fill = target_k
            u_fill = target_u
            break



    keff_dvorkin, ueff_dvorkin = [],[]

    empty_porosity = np.arange(por, 0.0, -0.01)
    keff_tmp, ueff_tmp = k_cem, u_cem
    previous_keff, previous_ueff = keff_tmp, ueff_tmp

    for pore in empty_porosity:
        pore = round(pore, 2)
        iterations = 0
        error_tolerance = 0.01
        buffer=1.004   #limits how much value can change per iteration

        while True:
            if iterations % 300 == 0 and iterations != 0:
                error_tolerance += 0.05

            target_keff = ((1 - por) * (ks - keff_tmp) * p_var(ks, us, keff_tmp, ueff_tmp, inclusion_shape) + \
                      (por - pore) * (k_cement - keff_tmp) * \
                      p_var(k_cement, mu_cement, keff_tmp, ueff_tmp, inclusion_shape)) / \
                      (pore *  p_var(0, 0, keff_tmp, ueff_tmp, inclusion_shape))

            if target_keff > keff_tmp * buffer:
                target_keff = keff_tmp * buffer
            if target_keff < keff_tmp / buffer:
                target_keff = keff_tmp / buffer
            if target_keff > ks * buffer:
                target_keff = ks * buffer
            if target_keff < k_cem / buffer:
                target_keff = k_cem / buffer

            try:
                if target_keff < previous_keff:
                    target_keff = previous_keff
            except:
                pass


            target_ueff = ((1 - por) * (us - ueff_tmp) * q_var(ks, us, keff_tmp, ueff_tmp, inclusion_shape) + \
                          (por - pore) * (mu_cement - ueff_tmp) * \
                           q_var(k_cement, mu_cement, keff_tmp, ueff_tmp, inclusion_shape)) / \
                          (pore *  q_var(0, 0, keff_tmp, ueff_tmp, inclusion_shape))

            if target_ueff > ueff_tmp * buffer:
                target_ueff = ueff_tmp * buffer
            if target_ueff < ueff_tmp / buffer:
                target_ueff = ueff_tmp / buffer
            if target_ueff > us * buffer:
                target_ueff = us * buffer
            if target_ueff < u_cem / buffer:
                target_ueff = u_cem / buffer

            try:
                if target_ueff < previous_ueff:
                    target_ueff = previous_ueff
            except:
                pass

            if (np.abs(target_keff - keff_tmp) > error_tolerance) and np.abs(target_ueff - ueff_tmp) > error_tolerance:
                keff_tmp = target_keff
                ueff_tmp = target_ueff
                iterations += 1
                continue
            else:
                keff_dvorkin.append(target_keff)
                ueff_dvorkin.append(target_ueff)
                previous_keff = target_keff
                previous_ueff = target_ueff
                break
                
    ## Gather up moduli and porosity values
    porosity = np.hstack((np.flip(empty_porosity), cemented_por_range))
    
    keff_dvorkin.reverse()
    k = np.hstack((np.array(keff_dvorkin), np.array(k_contact)))
    
    ueff_dvorkin.reverse()
    u = np.hstack((np.array(ueff_dvorkin), np.array(u_contact)))
    
    return k, u, porosity



def self_consistent_contact_cement_SYM(k_cement, mu_cement, k_grain, mu_grain, 
                  por, crit_por, C, scheme, inclusion_shape='spheres',
                  crack_aspect_ratio=10e-2):
    """
    Combine self-consistent effective medium modeling with Contact Cement
    model to calculate effective moduli over full porosity range.
    Reference: Dvorkin et al. 1999
    
    Currently written for matrix, cement, and void phases. Only handles one 
    inclusion shape for both cement inclusion and voids
    
    !! UPDATE THIS TO TAKE ASPECT RATIO FOR VOIDS SEPERATE FROM CEMENT
    
    Inputs
        k_cement: bulk modulus of cement in GPa
        mu_cement: shear modulus of cement in GPa
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        por: porosity transition from cementing to pore-filling
        crit_por: critical porosity of uncemented matrix
        C: coordination number
        scheme: cementation scheme ('contact' or 'uniform')
        inclusion_shape: shape of included cement and voids
        crack_aspect_ratio: shape parameter for cement and voids
        
    Returns
        Array of effective bulk moduli in GPa for cemented sand model
        Array of effective shear moduli in GPa for cemented sand model 
        Porosity values corresponding to moduli
    """
    
    allowable_inclusions = ['spheres', 'disks', 'penny cracks']
    assert inclusion_shape.lower() in allowable_inclusions, "Inclusion shape not supported"


    ## Helper functions for P and Q coefficients
    ## RP handbook, page 187
    def z_var(k, u):
        z = (u*(9*k + 8*u)) / (6*(k + 2*u))
        return z
    
    def b_var(k,u):
        b = u * ((3*k + u)/(3*k + 4*u))
        return b

    def q_var(ki, ui, km, um, shape):
        
        if shape.lower()=='spheres':
            q = (um + z_var(km, um)) / (ui + z_var(km, um))
            
        if shape.lower()=='disks':
            q = (um + z_var(ki, ui)) / (ui + z_var(ki, ui))
            
        if shape.lower()=='penny cracks':
            q = (1/5) * (1 + (8*um / (4*ui + np.pi*crack_aspect_ratio*(um + 2*b_var(km, um)))) + \
                         2*((ki + 2/3*(ui + um))/(ki + 4/3*ui + np.pi*crack_aspect_ratio*b_var(km, um))))
            
        return q

    def p_var(ki, ui, km, um, shape):
        
        if shape.lower()=='spheres':
            p = (km + (4/3)*um) / (ki + (4/3)*um)
            
        if shape.lower()=='disks':
            p = (km + (4/3)*ui) / (ki + (4/3)*ui)
            
        if shape.lower()=='penny cracks':
            p = (km + (4/3)*ui) / (ki + (4/3)*ui + np.pi*crack_aspect_ratio*b_var(km, um))
            
        return p
    
    
    ## Get values from contact cement model
    k_contact, u_contact = [],[]
    cemented_por_range = np.arange(por,crit_por,0.01)
    for por_val in cemented_por_range:
        
        #~~~~~ try using murphy
        C = upper_murphy(por_val)
        #~~~~~~~~~~~
        
        k_cem, u_cem = cemented_model(k_cement, mu_cement, k_grain, mu_grain, 
                  por_val, crit_por, C, scheme)
        k_contact.append(k_cem)
        u_contact.append(u_cem)
    
    
    ## Use self-consistent model to to calculate moduli between
    ## zero porosity and cemented fitting porosity
    #~~~~~ try using murphy
    C = upper_murphy(por)
    #~~~~~~~~~~~
    
    k_cem, u_cem = cemented_model(k_cement, mu_cement, k_grain, mu_grain, 
                  por, crit_por, C, scheme)
    zs = z_var(k_cem, u_cem)
    ks = (((1/(k_cem + (4/3)*u_cem)) - (por / ((4/3)*u_cem)))**-1) * (1 - por) - (4/3)*u_cem
    us = (((1 / (u_cem + zs)) - (por / zs))**-1) * (1 - por) - zs

    k_fill, u_fill = hs_bound(bound='lower', volume_fracts=[crit_por, 1-crit_por], 
                              bulk_mods = [k_cement, k_grain], shear_mods = [mu_cement, mu_grain], porosity=0)

    
    ##!! THIS COULD BE REPLACED WITH SYMPY
    error_tolerance = 0.005

    while True:
        target_k = (por * (k_cement - k_fill) * p_var(k_cement, mu_cement, k_fill, u_fill, inclusion_shape)) / \
                    ((1 - por) * p_var(ks, us, k_fill, u_fill, inclusion_shape)) + ks

        if target_k > voight_average([crit_por, 1-crit_por], [k_grain, k_cement]):
            target_k = voight_average([crit_por, 1-crit_por], [k_grain, k_cement])
        if target_k < reuss_average([crit_por, 1-crit_por], [k_grain, k_cement]):
            target_k = reuss_average([crit_por, 1-crit_por], [k_grain, k_cement])

        target_u = (por * (mu_cement - u_fill) * q_var(k_cement, mu_cement, k_fill, u_fill, inclusion_shape)) / \
                ((1 - por) * q_var(ks, us, k_fill, u_fill, inclusion_shape)) + us

        if target_u > voight_average([crit_por, 1-crit_por], [mu_grain, mu_cement]):
            target_u = voight_average([crit_por, 1-crit_por], [mu_grain, mu_cement])
        if target_u < reuss_average([crit_por, 1-crit_por], [mu_grain, mu_cement]):
            target_u = reuss_average([crit_por, 1-crit_por], [mu_grain, mu_cement])

        if (np.abs(target_k - k_fill) > error_tolerance) and (np.abs(target_u - u_fill) > error_tolerance):
            k_fill = target_k
            u_fill = target_u
            continue
        else:
            k_fill = target_k
            u_fill = target_u
            break



    keff_dvorkin, ueff_dvorkin = [],[]

    empty_porosity = np.arange(por, 0.0, -0.01)
    
    k_start, u_start = k_cem, u_cem

    iter_counter = 1
    for pore in empty_porosity:
        sys.stdout.write("\rRunning self-consistent iteration {}/{}".format(iter_counter,len(empty_porosity)))
        sys.stdout.flush()
        
        pore = round(pore, 2)
        
        keff_tmp, ueff_tmp = symbols('keff_tmp ueff_tmp', positive=True, real=True)
        
        eqk = Eq(((1-por)*(ks-keff_tmp)*p_var(ks, us, keff_tmp, ueff_tmp, inclusion_shape) + 
         (por - pore)*(k_cement - keff_tmp)*p_var(k_cement, mu_cement, keff_tmp, ueff_tmp, inclusion_shape) - 
         pore *keff_tmp*  p_var(0, 0, keff_tmp, ueff_tmp, inclusion_shape)),0)
    
        equ = Eq(((1-por)*(us-ueff_tmp)*q_var(ks, us, keff_tmp, ueff_tmp, inclusion_shape) + 
                 (por-pore)*(mu_cement-ueff_tmp)*q_var(k_cement, mu_cement, keff_tmp, ueff_tmp, inclusion_shape) - 
                 pore *ueff_tmp*  q_var(0, 0, keff_tmp, ueff_tmp, inclusion_shape)),0)
        
#         iter_solve = solve((eqk, equ), (keff_tmp, ueff_tmp))
#         if len(iter_solve)==0:
        iter_solve = nsolve((eqk, equ), (keff_tmp, ueff_tmp), (k_start, u_start))
    
        try:
            target_keff = iter_solve[0][0]
            target_ueff = iter_solve[0][1]
#             target_keff = iter_solve[0]
#             target_ueff = iter_solve[1]
            
        except:
            target_keff = iter_solve[0]
            target_ueff = iter_solve[1]
        
        keff_dvorkin.append(target_keff)
        ueff_dvorkin.append(target_ueff)
        k_start, u_start = target_keff, target_ueff
        
        iter_counter+=1
                
    ## Gather up moduli and porosity values
    porosity = np.hstack((np.flip(empty_porosity), cemented_por_range))
    
    keff_dvorkin.reverse()
    k = np.hstack((np.array(keff_dvorkin), np.array(k_contact)))
    
    ueff_dvorkin.reverse()
    u = np.hstack((np.array(ueff_dvorkin), np.array(u_contact)))
    
    return k, u, porosity



