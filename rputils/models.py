"""
Assorted rock physics models
"""


import numpy as np
import warnings
from scipy.optimize import curve_fit
import sys

from . import elastic
from . import bounds

try:
    from sympy import symbols, Eq, solve, nsolve, solveset, S
    has_sympy = True
except ImportError:
    has_sympy = False




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
    por = np.array(por)
    n = 17.34 - 34*por + 14*(por**2)
    return n

def lower_murphy_bounded(por):
    """
    Calculate polynomial fit to lower Murphy (1982) equation
    that has been clipped to Cmin=3. Only fit up to 0.7 porosity
    
    Inputs:
        por (float): porosity (0-1)
    Returns:
        coordination number (float)
    """
    por = np.array([por])
    por_range = np.arange(0., 0.71, 0.05)
    
    min_C = 3
    murphy_vals = []
    for p in por_range:
        murphy_vals.append(lower_murphy(p))
    murphy_vals = [min_C if i < min_C else i for i in murphy_vals]
    murphy_fit = np.poly1d(np.polyfit(por_range, murphy_vals, deg=7))

    C = np.array([murphy_fit(p) for p in por])
    C = np.squeeze(np.clip(C, a_min=min_C, a_max=None))
    return C

def modified_murphy(por):
    """
    Modify the Murphy (1982) coordination number fit to produce 
    lower values above 0.3 porosity and smoothly approach a
    minimum value of 3. Only fit up to 0.7 porosity
    
    Inputs:
        por (float): porosity (0-1)
    Returns:
        coordination number (float)
    """
    por = np.array([por])
    por_range = np.array([0., 0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.70])
    c_vals = np.array([upper_murphy(0), upper_murphy(0.05), upper_murphy(0.15),
                      upper_murphy(0.25), upper_murphy(0.35), 6., 3.2, 3])
    
    my_fit = np.poly1d(np.polyfit(por_range, c_vals, deg=3))

    C = np.array([my_fit(p) for p in por])
    C = np.squeeze(C)
    return C


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
    por = np.array(por)
    # n = 20 - 34*por + 14*(por**2)  #linear formulation
    # n = 24*np.exp(-2.547*por) - 0.3731  #provided in Zimmer 2007
    n = 23.63*np.exp(-3.199*por) + 1.602  #based on least squares fit to data
    return n



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
    poisson_grain = elastic.poisson_mod(k_grain, mu_grain)
    
    k = ((((C**2)*((1-crit_por)**2)*(mu_grain**2)) / ((18*(np.pi**2))*((1-poisson_grain)**2))) * pressure)**(1/3)
    
    u = ((2+3*f-poisson_grain*(1+3*f))/(5*(2-poisson_grain))) * \
        ((((3*C**2)*((1-crit_por)**2)*(mu_grain**2)) \
         / ((2*(np.pi**2))*((1-poisson_grain)**2))) * pressure)**(1/3)
    
    return k, u
    
    

# def bachrach_angular_old(k_grain, mu_grain, por, C, pressure, Rc_ratio, 
#                           cohesionless_percent=0, Rg=1):
#     """
#     Method from Bachran et al. 2000 to control radii of curvature between
#     grains to match measured observations from beach sand. Roughly
#     corresponds with RP Handbook page 246-249. Returns the same values
#     as other HM function when Rc_ratio and cohensionless_percent are
#     both set to 1
    
#     Inputs
#         k_grain: grain bulk modulus in GPa
#         mu_grain: grain shear bulk modulus in GPa
#         por: porosity (0-1)
#         C: coordination number
#         pressure: effective pressure in GPa
#         Rc_ratio: describes angularity of grains, 1=spherical
#         cohesionless_percent: percent of grains with frictionless contacts.
#             This may be a fitting parameter for shallow loose sands.
#             HS- will be used to mix moduli
#         Rg: grain radius (seems to be negligible)
        
#     Returns
#         Effective bulk modulus in GPa
#         Effective shear modulus in GPa
#     """
#     assert Rc_ratio <= 1, "Rc_ratio should be value (0-1)"
#     assert cohesionless_percent <= 1, "Rc_ratio should be value (0-1)"
    
#     poisson_grain = elastic.poisson_mod(k_grain, mu_grain)
    
#     F = (4 * np.pi * (Rg**2) * pressure) / (C * (1-por))
    
#     a = ((3 * F * Rc_ratio * (1-poisson_grain)) / (8 * mu_grain))**(1/3)
    
#     Sn = (4*a*mu_grain)/(1-poisson_grain)
#     St = (8*a*mu_grain)/(2-poisson_grain)
    
#     khm = ((C*(1-por))/(12*np.pi*Rg))*Sn
#     uhm = ((C*(1-por))/(20*np.pi*Rg))*(Sn + 1.5*St)
    
#     if cohesionless_percent != 0:
        
#         uhm_co = ((C*(1-por))/(20*np.pi*Rg))*(Sn)
#         volume_fracts = [1-cohesionless_percent, cohesionless_percent]
#         bulk_mods = [khm, khm]
#         shear_mods = [uhm, uhm_co]
        
#         khm, uhm = bounds.hs('lower', volume_fracts, bulk_mods, shear_mods)
    
#     return khm, uhm


def bachrach_angular(k_grain, mu_grain, por, C, pressure, c_ratio, slip_percent=0):
    """
    Method from Bachrach and Avseth 2008 
    
    Inputs
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        por: porosity (0-1)
        C: coordination number
        pressure: effective pressure in GPa
        c_ratio: describes contact angularity of grains, 1=spherical
        slip_percent: percent of slipping grain contacts
        
    Returns
        Effective bulk modulus in GPa
        Effective shear modulus in GPa
    """
    assert c_ratio <= 1, "c_ratio should be value (0-1)"
    assert slip_percent <= 1, "Slip should be value (0-1)"

    ## Fixing "slip percent" since it should actually be "no-slip percent"
    slip_percent = 1-slip_percent
    
    poisson_grain = elastic.poisson_mod(k_grain, mu_grain)
    
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
    assert has_sympy, "Digby model requires SymPy module"

    poisson_grain = elastic.poisson_mod(k_grain, mu_grain)

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
    
    lame_grain = elastic.lame_mod(k_grain, mu_grain)
    A = (1/(4*np.pi))*((1/mu_grain)-(1/(mu_grain+lame_grain)))
    B = (1/(4*np.pi))*((1/mu_grain)+(1/(mu_grain+lame_grain)))

    if mode.lower()=="rough":
        keff = (1/6)*((3*(1-por)**2*C**2*pressure)/(np.pi**4*B**2))**(1/3)
        ueff = (3/5)*keff*((5*B+A)/(2*B+A))

    elif mode.lower()=="smooth":
        ueff = (1/10)*((3*(1-por)**2*C**2*pressure)/(np.pi**4*B**2))**(1/3)
        keff = (5/3)*ueff

    return keff, ueff


# def amos_model(por, crit_por, keff, ueff, k_grain, mu_grain, fit_por, max_poisson):
#     """
#     Use HS lower bound for K similar to Uncemented Model. Calculate
#     Mu based on trend between observed Poisson's ratio and the effective
#     mineral Poisson's ratio. Assumes PR of grain is smaller than PR of
#     granular mixture
    
#     Inputs
#         por: Porosity for which solving for Effective Bulk Modulus
#         crit_por: critical porosity 
#         keff: effective bulk modulus at critical porosity
#         ueff: effective shear modulus at critical porosity
#         k_grain: grain bulk modulus in GPa
#         mu_grain: grain shear bulk modulus in GPa
#         fit_por: fitting porosity for max_poisson
#         max_poisson: maximum poisson ratio to fit (high porosity end)
        
#     Returns
#         Effective bulk modulus in GPa 
#         Effective shear modulus in GPa 
#     """

#     k = ((((por/crit_por)/(keff+((4/3)*ueff)))+((1-(por/crit_por))/(k_grain+((4/3)*ueff))))\
#          **(-1)) - ((4/3)*ueff)

#     # ## Trying an inverse distance weighting for PR
#     # ## NOTE: inverse weighting appears to have worse results 
#     # ## than linear weighting below
#     # poisson_grain = poisson_mod(k_grain, mu_grain)
#     # dist_factor = (por / crit_por) * 100
#     # if dist_factor == 0:
#     #     target_poisson = poisson_grain
#     # else:
#     #     inv_factor = 1 / dist_factor
#     #     target_poisson = inv_factor*poisson_grain + (1-inv_factor)*max_poisson

#     ## Try using a root function to fit PR
#     def fit_func_bounded(x, vert_stretch, vert_translate):
#         horz_translate = 0.25
#         curve_factor = 0.25  # was using 0.2 previously
#         return vert_stretch * ((x-horz_translate)/(curve_factor + np.abs(x-horz_translate)))\
#                 + vert_translate
        
#     poisson_grain = elastic.poisson_mod(k_grain, mu_grain)
#     fit_x = [0, fit_por]
#     fit_y = [poisson_grain, max_poisson]
#     initial_guess = [0.2, 0.3]
#     fit_parms, _ = curve_fit(fit_func_bounded, fit_x, fit_y, p0=initial_guess)
#     target_poisson = fit_func_bounded(por, *fit_parms)
        
#     ## Do linear interpolation of PR between grain PR and highest
#     ## observed PR in the data
#     # poisson_grain = poisson_mod(k_grain, mu_grain)
#     # por_scale_factor = por / fit_por
#     # poisson_diff = max_poisson - poisson_grain
#     # target_poisson = poisson_grain + por_scale_factor*poisson_diff

#     ## make sure PR values aren't above 0.5 
#     if target_poisson > 0.495:
#         target_poisson = 0.495

#     u = (3*k - 6*k*target_poisson)/(2*target_poisson + 2)
    
#     return k, u


def amos_isoframe_model(por, crit_por, keff, ueff, k_grain, mu_grain, fit_por, max_poisson, trans_por):

    ## USING ISOFRAME MODEL MIX BELOW TRANSITION POROSITY
    k1, k2 = k_grain, keff
    u1, u2 = mu_grain, ueff
    k_upper, _ = bounds.modified_hs_upper(k1, k2, u1, u2, por, crit_por)

    effective_por = por/crit_por
    bulk_mods = [keff, k_grain]
    shear_mods = [ueff, mu_grain]
    f1, f2 = effective_por, 1-effective_por
    vol_fracts = [f1, f2]
    k_lower, _ = bounds.hs('lower', vol_fracts, bulk_mods, shear_mods)
    # k_lower = ((((por/crit_por)/(keff+((4/3)*ueff)))+((1-(por/crit_por))/(k_grain+((4/3)*ueff))))\
    #      **(-1)) - ((4/3)*ueff)

    ## create an exponential scaler to apply to Upper/Lower mixing
    ## since linear mixing produced harsh kink in moduli
    x_scaling = np.arange(0,1., 0.002)
    y_scaling = x_scaling**(1/50)
    y_scaling = [(i-np.amin(y_scaling))/(np.amax(y_scaling)-np.amin(y_scaling)) for i in y_scaling]

    if por >= trans_por:
        k = k_lower
    else:
        trans_por_frac = por/trans_por
        scale_idx = np.abs(x_scaling-trans_por_frac).argmin()
        scale_val = y_scaling[scale_idx]
        low_mix = scale_val * k_lower
        upper_mix = (1-scale_val) * k_upper
        k = low_mix + upper_mix

    ## Try using a root function to fit PR
    ## NOTE: the curve_fit call below produces a warning that covariance
    ## cannot be estimated, but that does not halt generation of the 
    ## necessary fit parameters
    def fit_func_bounded(x, vert_stretch, vert_translate):
        horz_translate = 0.25
        curve_factor = 0.25  # was using 0.2 previously
        return vert_stretch * ((x-horz_translate)/(curve_factor + np.abs(x-horz_translate)))\
                + vert_translate
        
    poisson_grain = elastic.poisson_mod(k_grain, mu_grain)
    fit_x = [0, fit_por]
    fit_y = [poisson_grain, max_poisson]
    initial_guess = [0.2, 0.3]
    fit_parms, _ = curve_fit(fit_func_bounded, fit_x, fit_y, p0=initial_guess)
    target_poisson = fit_func_bounded(por, *fit_parms)

    ## make sure PR values aren't above 0.5 
    if target_poisson > 0.47:
        target_poisson = 0.47

    u = (3*k - 6*k*target_poisson)/(2*target_poisson + 2)
    
    return k, u



# def amos_model_reuss(por, crit_por, keff, ueff, k_grain, mu_grain, fit_por, max_poisson):
#     """
#     Use modified Reuss bound for K. Calculate
#     Mu based on trend between observed Poisson's ratio and the effective
#     mineral Poisson's ratio. Assumes PR of grain is smaller than PR of
#     granular mixture
    
#     Inputs
#         por: Porosity for which solving for Effective Bulk Modulus
#         crit_por: critical porosity 
#         keff: effective bulk modulus at critical porosity
#         ueff: effective shear modulus at critical porosity
#         k_grain: grain bulk modulus in GPa
#         mu_grain: grain shear bulk modulus in GPa
#         fit_por: fitting porosity for max_poisson
#         max_poisson: maximum poisson ratio to fit (high porosity end)
        
#     Returns
#         Effective bulk modulus in GPa 
#         Effective shear modulus in GPa 
#     """
#     high_por = 0.6  ## testing, author states this differs from crit por
#     k = bounds.modified_reuss(k_grain, keff, por, high_por)


#     ## Try using a root function to fit PR
#     def fit_func_bounded(x, vert_stretch, vert_translate):
#         horz_translate = 0.3
#         curve_factor = 0.2
#         return vert_stretch * ((x-horz_translate)/(curve_factor + np.abs(x-horz_translate)))\
#                 + vert_translate
        
#     poisson_grain = elastic.poisson_mod(k_grain, mu_grain)
#     fit_x = [0, fit_por]
#     fit_y = [poisson_grain, max_poisson]
#     initial_guess = [0.2, 0.3]
#     fit_parms, _ = curve_fit(fit_func_bounded, fit_x, fit_y, p0=initial_guess)
#     target_poisson = fit_func_bounded(por, *fit_parms)

#     u = (3*k - 6*k*target_poisson)/(2*target_poisson + 2)
    
#     return k, u



def patchy_ice_model(por, crit_por, C, 
                     keff, ueff, 
                     k_grain, mu_grain, dens_grain,
                     k_ice, u_ice, dens_ice,
                     fit_por, trans_por, max_poisson,
                     contact_scheme, max_cement,
                     patchy_scheme, mix_amount):
    """
    Use Avseth's et al. (2016) Patchy Cement Model concept to create patchy 
    mix between Contact Cement model and Amos Isoframe model 

    Inputs
        por (float 0-1): porosity value(s) at which to calculate moduli
        crit_por (float 0-1): critical posority
        C (float): coordination number
        keff (float): effective bulk modulus at critical porosity in GPa
        ueff (float): effective shear modulus at critical porosity in GPa
        k_grain (float): grain bulk modulus in GPa
        mu_grain (float): grain shear bulk modulus in GPa
        dens_grain (float): average grain density in g/cm3
        k_ice (float): ice bulk modulus in GPa
        u_ice (float): ice shear modulus in GPa
        dens_ice (float): ice density in g/cm3
        fit_por (float 0-1): fitting porosity for max_poisson
        trans_por (float 0-1): transition porosity below which increase stiffness
        max_poisson (float): maximum poisson ratio to fit in Amos model
        contact_scheme (str): cement scheme for Dvorkin model, "contact" or "uniform"
        max_cement (float 0-1): maximum amount of cement, suggest less than 0.15
        patchy_scheme (str): mixing scheme for Patchy Cement, "soft" or "stiff"
        mix_amount (float 0-1): ratio between minimum and maximum using patchy_scheme

    Returns
        k_patchy (array): bulk moduli of patchy mix
        u_patchy (array): shear moduli of patchy mix
        dens_patchy (array): densities of patchy mix
    """
    ##-------------------------------
    ## CHECKS 
    
    ## Make sure porosity is numpy array for easier handling
    if type(por) != np.ndarray:
        por = np.array(por, ndmin=1)

    ## Warn if coordination number is below logical low value
    if C < 3:
        warnings.warn("Coordination number is below minimum value of 3")

    assert np.all((por >= 0) & (por <= 1)), "Porosity values out of bounds (0-1)"
    assert 0 <= fit_por <= 1, "Fitting porosity out of bounds (0-1)" 
    assert 0 <= trans_por <= 1, "Transition porosity out of bounds (0-1)"
    # assert 0 <= max_poisson <= 0.5, "Max-Poisson out of bounds (0-0.5)"
    assert 0 <= max_cement <= 1, "max_cement out of bounds (0-1), suggest below 0.15" 
    assert 0 <= mix_amount <= 1, "mix_amount out of bounds (0-1)" 

    contact_schemes_allowed = ["contact", "uniform"]
    patchy_schemes_allowed = ["soft", "stiff"]
    assert contact_scheme in contact_schemes_allowed, \
        "Contact scheme must be 'contact' or 'uniform'"
    assert patchy_scheme in patchy_schemes_allowed, \
        "Patchy scheme must be 'soft' or 'stiff'"

    ## make sure PR values aren't above 0.5 
    if max_poisson > 0.47:
        max_poisson = 0.47

    ##-------------------------------
    ## MODELING
    
    ## Calculate CCT model for mixing
    kcem, ucem = cemented_model(k_ice, u_ice, k_grain, mu_grain, 
                  crit_por-max_cement, crit_por, C, scheme=contact_scheme)

    ## Create soft mix
    tmp_k_soft, tmp_u_soft = bounds.hs("lower", [mix_amount, 1-mix_amount], [kcem, keff], [ucem, ueff])

    ## Create stiff mix
    tmp_k_stiff, tmp_u_stiff = bounds.hs("upper", [mix_amount, 1-mix_amount], [kcem, keff], [ucem, ueff])

    ## Create effective icy mineral
    norm_val = k_grain - k_ice
    tmp_k_mix = norm_val - (norm_val * (mix_amount * max_cement)) + k_ice
    norm_val = mu_grain - u_ice
    tmp_u_mix = norm_val - (norm_val * (mix_amount * max_cement)) + u_ice

    k_patchy, u_patchy, dens_patchy = [], [], []
    mix_por = crit_por-(mix_amount*max_cement)  # Max uncemented porosity
    for p in por:

        ## Only calculate up to max uncemented porosity
        ## THIS CAUSES NANS IN LUNAR MODEL
        # if p > mix_por:
        #     continue


        ##!!!!!!!!!!!!!!!!!!
        ## HOW SHOULD I BE TREATING DENSITY AND MODULI OF MIXES?
        ## REMOVE THE CEMENT AMOUNT THEN MIX GRAIN AND ICE??
        
        ## Calculate density
        # dens = eff_min_density*(1-p)
        # dens_patchy.append(float(dens))
  #-      # if I let bulk density decrease as below, velocity oddly increases
  #-      # maybe if I also change out the effective mineral moduli this would fix

        tmp_ice_vol = mix_amount * max_cement
        tmp_grain_dens = sum([f*d for f, d in zip([1-tmp_ice_vol, tmp_ice_vol], [dens_grain, dens_ice])])
        dens =  (tmp_grain_dens) * (1-p)
        dens_patchy.append(dens)

        ## Use uncemented model for Stiff moduli
        if patchy_scheme.lower() == "stiff":
            if mix_amount == 0:
                k_tmp, u_tmp = amos_isoframe_model(p, mix_por, tmp_k_soft, tmp_u_soft,
                                                tmp_k_mix, tmp_u_mix, fit_por = fit_por, 
                                                  max_poisson=max_poisson, trans_por=trans_por)
            else:
                k_tmp, u_tmp = modified_uncemented_model(p, mix_por, tmp_k_stiff, tmp_u_stiff, 
                                                 tmp_k_mix, tmp_u_mix, trans_por=trans_por)
            k_patchy.append(k_tmp)
            u_patchy.append(u_tmp)

        ## Use Amos model for Soft moduli
        elif patchy_scheme.lower() == "soft":
            k_tmp, u_tmp = amos_isoframe_model(p, mix_por, tmp_k_soft, tmp_u_soft,
                                                tmp_k_mix, tmp_u_mix, fit_por = fit_por, 
                                                  max_poisson=max_poisson, trans_por=trans_por)
            k_patchy.append(k_tmp)
            u_patchy.append(u_tmp)

    k_patchy, u_patchy, dens_patchy = np.array(k_patchy), np.array(u_patchy), np.array(dens_patchy)

    return k_patchy, u_patchy, dens_patchy




def modified_uncemented_model(por, crit_por, k_eff, u_eff, k_grain, mu_grain, trans_por):
    """
    Uncemented model modified with moduli increase below transition
    porosity as in my isoframe model

    Inputs
        por (float 0-1):  porosity at which to calculate effective moduli
        crit_por (float 0-1): critical porosity of media 
        k_eff: effective bulk modulus at critical porosity in GPa
        u_eff: effective shear modulus at critical porosity in GPa
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        trans_por (float 0-1): transition porosity above which to 
            begin mixing stiffer material
        
    Returns
        Effective bulk modulus in GPa 
        Effective shear modulus in GPa 
    """

    ## USING ISOFRAME MODEL MIX BELOW TRANSITION POROSITY
    k1, k2 = k_grain, k_eff
    u1, u2 = mu_grain, u_eff
    k_upper, u_upper = bounds.modified_hs_upper(k1, k2, u1, u2, por, crit_por)

    effective_por = por/crit_por
    bulk_mods = [k_eff, k_grain]
    shear_mods = [u_eff, mu_grain]
    f1, f2 = effective_por, 1-effective_por
    vol_fracts = [f1, f2]
    k_lower, u_lower = bounds.hs('lower', vol_fracts, bulk_mods, shear_mods)

    ## create an exponential scaler to apply to Upper/Lower mixing
    ## since linear mixing produced harsh kink in moduli
    x_scaling = np.arange(0,1., 0.002)
    y_scaling = x_scaling**(1/50)
    y_scaling = [(i-np.amin(y_scaling))/(np.amax(y_scaling)-np.amin(y_scaling)) for i in y_scaling]

    if por >= trans_por:
        k = k_lower
        u = u_lower

    else:
        trans_por_frac = por/trans_por
        scale_idx = np.abs(x_scaling-trans_por_frac).argmin()
        scale_val = y_scaling[scale_idx]
        low_mix_k = scale_val * k_lower
        low_mix_u = scale_val * u_lower
        upper_mix_k = (1-scale_val) * k_upper
        upper_mix_u = (1-scale_val) * u_upper
        k = low_mix_k + upper_mix_k
        u = low_mix_u + upper_mix_u
    
    return k, u




def uncemented_model(por, crit_por, k_eff, u_eff, k_grain, mu_grain):
    """
    Effective moduli, uncemented sand model. Uses a modified
    HS lower bound to evaluate effective moduli at a porosity
    (por) compared to the starting framework (crit_por).
    RP Handbook (2nd) pg.258
    
    Inputs
        por (float 0-1):  porosity at which to calculate effective moduli
        crit_por (float 0-1): critical porosity of media 
        k_eff: effective bulk modulus at critical porosity in GPa
        u_eff: effective shear modulus at critical porosity in GPa
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        
    Returns
        Effective bulk modulus in GPa 
        Effective shear modulus in GPa  
    """
    
    k = ((((por/crit_por)/(k_eff+((4/3)*u_eff)))+((1-(por/crit_por))/(k_grain+((4/3)*u_eff))))**(-1)) - ((4/3)*u_eff)
    
    third_term = ((9*k_eff+8*u_eff)/(k_eff+2*u_eff))
    first_num = por / crit_por
    first_denom = u_eff + (u_eff/6)*third_term
    second_num = 1 - (por / crit_por)
    second_denom = mu_grain + (u_eff/6)*third_term
    u = (((first_num/first_denom)+(second_num/second_denom))**-1) - (u_eff/6)*third_term
    
    return k, u
    


def stiff_model(por, crit_por, k_eff, u_eff, k_grain, mu_grain):
    """
    Effective moduli, stiff sand model. Uses a modified
    HS uper bound to evaluate effective moduli at a porosity
    (por) compared to the starting framework (crit_por).
    RP Handbook (2nd) pg.260
    
    Inputs
        por: Porosity for which solving for Effective Bulk Modulus
        crit_por: critical porosity 
        k_eff: effective bulk modulus at critical porosity in GPa
        u_eff: effective shear modulus at critical porosity in GPa
        k_grain: grain bulk modulus in GPa
        mu_grain: grain shear bulk modulus in GPa
        
    Returns
        Effective bulk modulus in GPa 
        Effective shear modulus in GPa 
    """
    
    k = ((((por/crit_por)/(k_eff+((4/3)*mu_grain)))+((1-(por/crit_por))/(k_grain+((4/3)*mu_grain))))**(-1)) \
        - ((4/3)*mu_grain)
    
    third_term = ((9*k_grain+8*mu_grain)/(k_grain+2*mu_grain))
    first_num = por / crit_por
    first_denom = u_eff + (mu_grain/6)*third_term
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
    
    poisson_cement = elastic.poisson_mod(k_cement, mu_cement)
    poisson_grain = elastic.poisson_mod(k_grain, mu_grain)
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



def scc(k_cement, mu_cement, k_grain, mu_grain, 
                  por, crit_por, C, scheme, inclusion_shape='spheres',
                  crack_aspect_ratio=10e-2):
    """
    NEEDS TESTING!!!
    
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

    # k_fill, u_fill = hs_bound(bound='lower', volume_fracts=[crit_por, 1-crit_por], 
    #                           bulk_mods = [k_cement, k_grain], shear_mods = [mu_cement, mu_grain], porosity=0)

    # error_tolerance = 0.005

    # while True:
    #     target_k = (por * (k_cement - k_fill) * p_var(k_cement, mu_cement, k_fill, u_fill, inclusion_shape)) / \
    #                 ((1 - por) * p_var(ks, us, k_fill, u_fill, inclusion_shape)) + ks

    #     if target_k > voight_average([crit_por, 1-crit_por], [k_grain, k_cement]):
    #         target_k = voight_average([crit_por, 1-crit_por], [k_grain, k_cement])
    #     if target_k < reuss_average([crit_por, 1-crit_por], [k_grain, k_cement]):
    #         target_k = reuss_average([crit_por, 1-crit_por], [k_grain, k_cement])

    #     target_u = (por * (mu_cement - u_fill) * q_var(k_cement, mu_cement, k_fill, u_fill, inclusion_shape)) / \
    #             ((1 - por) * q_var(ks, us, k_fill, u_fill, inclusion_shape)) + us

    #     if target_u > voight_average([crit_por, 1-crit_por], [mu_grain, mu_cement]):
    #         target_u = voight_average([crit_por, 1-crit_por], [mu_grain, mu_cement])
    #     if target_u < reuss_average([crit_por, 1-crit_por], [mu_grain, mu_cement]):
    #         target_u = reuss_average([crit_por, 1-crit_por], [mu_grain, mu_cement])

    #     if (np.abs(target_k - k_fill) > error_tolerance) and (np.abs(target_u - u_fill) > error_tolerance):
    #         k_fill = target_k
    #         u_fill = target_u
    #         continue
    #     else:
    #         k_fill = target_k
    #         u_fill = target_u
    #         break



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



def scc_SYM(k_cement, mu_cement, k_grain, mu_grain, 
                  por, crit_por, C, scheme, inclusion_shape='spheres',
                  crack_aspect_ratio=10e-2):
    """
    NEEDS TESTING!!!
    
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

    assert has_sympy, "scc_SYM requires SymPy module"
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

    k_fill, u_fill = bounds.hs(bound='lower', volume_fracts=[crit_por, 1-crit_por], 
                              bulk_mods = [k_cement, k_grain], shear_mods = [mu_cement, mu_grain], porosity=0)

    
    ##!! THIS COULD BE REPLACED WITH SYMPY
    error_tolerance = 0.005

    while True:
        target_k = (por * (k_cement - k_fill) * p_var(k_cement, mu_cement, k_fill, u_fill, inclusion_shape)) / \
                    ((1 - por) * p_var(ks, us, k_fill, u_fill, inclusion_shape)) + ks

        if target_k > bounds.voight_average([crit_por, 1-crit_por], [k_grain, k_cement]):
            target_k = bounds.voight_average([crit_por, 1-crit_por], [k_grain, k_cement])
        if target_k < bounds.reuss_average([crit_por, 1-crit_por], [k_grain, k_cement]):
            target_k = bounds.reuss_average([crit_por, 1-crit_por], [k_grain, k_cement])

        target_u = (por * (mu_cement - u_fill) * q_var(k_cement, mu_cement, k_fill, u_fill, inclusion_shape)) / \
                ((1 - por) * q_var(ks, us, k_fill, u_fill, inclusion_shape)) + us

        if target_u > bounds.voight_average([crit_por, 1-crit_por], [mu_grain, mu_cement]):
            target_u = bounds.voight_average([crit_por, 1-crit_por], [mu_grain, mu_cement])
        if target_u < bounds.reuss_average([crit_por, 1-crit_por], [mu_grain, mu_cement]):
            target_u = bounds.reuss_average([crit_por, 1-crit_por], [mu_grain, mu_cement])

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
    
    # k_start, u_start = k_cem, u_cem
    k_start, u_start = k_grain*por, mu_grain*por

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



