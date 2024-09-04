import pybamm
import numpy as np
import pandas as pd
from pybamm import exp

# anode_raw = pd.read_csv(r'params_data/anode_ocp.csv')
# cathode_raw = pd.read_csv(r'params_data/cathode_ocp.csv')

# anode_sto_data = anode_raw.to_numpy()[:,0]
# anode_ocp_data = anode_raw.to_numpy()[:,1]
# cathode_sto_data = cathode_raw.to_numpy()[:,0]
# cathode_ocp_data = cathode_raw.to_numpy()[:,1]

# def anode_ocp(sto):
#     return pybamm.Interpolant(anode_sto_data, anode_ocp_data, sto, extrapolate=True)

# def cathode_ocp(sto):
#     return pybamm.Interpolant(cathode_sto_data, cathode_ocp_data, sto, extrapolate=True)

csmax_pos = 49034   # unit is mol/m^3, from FN P2D
csmax_neg = 31085   # unit is mol/m^3, from FN P2D

def anode_ocp(sto): 
    a1 =    7.19612503590668e-13
    a2 =    26.7952832958785
    a3 =    1.12442394245328
    b1 =    -12.4818534264270
    b2 =    27.5881779446520
    b3 =    0.0202512953236839
    c1 =    0.334265137924962
    c2 =    5.23141514297590
    c3 =    -0.314623755034365
    d1 =   1.00198718626221
    d2 =    32.6082106238054
    d3 =    -0.0998002406587664
    e1 =    22.5621817311624
    e2 =   -1.07295202705289
    e3 =   -1.43653644997359
    f1 =   -9.08915655677233
    f2 =  -1.81434757012079
    f3 =  -0.945892776009528
    g1 =  0.929674728826629
    g2 =  -35.8575505904909
    g3 =  -0.100609473717635
    u_eq = (
    a1*pybamm.exp(a2*sto)+a3
    +b1*pybamm.tanh(b2*(sto+b3))
    +c1*pybamm.tanh(c2*(sto+c3))
    +d1*pybamm.tanh(d2*(sto+d3))
    +e1*pybamm.tanh(e2*(sto+e3))
    +f1*pybamm.tanh(f2*(sto+f3))
    +g1*pybamm.tanh(g2*(sto+g3))
    )
    return u_eq
def cathode_ocp(sto): 
    a1 =    -4.40737340724912
    a2 =    6.53838349960042
    b1 =    31.2312677694666
    b2 =    -4.09290721223931
    b3 =    -0.435350225473633
    c1 =    13.3747083258844
    c2 =    4.96694452209909
    c3 =    -0.399119193405277
    d1 =    0.564048457189585
    d2 =    11.4574682413520
    d3 =    -0.231944342715458
    e1 =    -14.6483766967181
    e2 =    3.97369563476902
    e3 =    -0.617604325918297
    f1 =    0.478423940696707
    f2 =    -59.3012876881322
    f3 =    -0.955441113267442
    g1 =   33.3440808754568
    g2 =    3.63587543392702
    g3 =   -0.549939275644489

    u_eq = (
    a1*sto+a2
    +b1*pybamm.tanh(b2*(sto+b3))
    +c1*pybamm.tanh(c2*(sto+c3))
    +d1*pybamm.tanh(d2*(sto+d3))
    +e1*pybamm.tanh(e2*(sto+e3))
    +f1*pybamm.tanh(f2*(sto+f3))
    +g1*pybamm.tanh(g2*(sto+g3))
    )
    return u_eq

def anode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    m_ref = 3e-6 # (0.96~19.3 e-6 [(A/m2)(mol/m3)**1.5])
    E_r = 56000 # (50~70 [kJ/mol])
    # E_r = 0
    arrhenius = exp(E_r / 8.314 * (1 / 293.15 - 1 / T))
    return (m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5)

def cathode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    m_ref = 3e-6 #3.42e-6  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 17800
    # E_r = 0
    arrhenius = exp(E_r / 8.314 * (1 / 293.15 - 1 / T))
    return (m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5)

def anode_diffusivity(sto, T):
    m_ref = 0.75e-14 #1.6e-14
    E_r = 35000
    # E_r = 0
    arrhenius = exp(E_r / 8.314 * (1 / 293.15 - 1 / T))
    coeff = (1.5 - sto) ** 1.5   # need to change
    return (m_ref * arrhenius * 1)

def cathode_diffusivity(sto,T):
    m_ref = 0.75e-14 #3e-14
    E_r = 30000
    # E_r = 0   # need to change
    arrhenius = exp(E_r / 8.314 * (1 / 293.15 - 1 / T))
    return (m_ref * arrhenius)

def electrolyte_conductivity(c_e, T):
    sigma_e = 0.29*(0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000))
    E_r = 14000
    # E_r = 0
    arrhenius = exp(E_r / 8.314 * (1 / 293.15 - 1 / T))
    return sigma_e * arrhenius

def electrolyte_diffusivity(c_e, T):
    D_c_e = 0.75*(8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10)
    E_r = 20000
    arrhenius = exp(E_r / 8.314 * (1 / 293.15 - 1 / T))
    return D_c_e * arrhenius

params = {
    # cell spec.
    'Nominal cell capacity [A.h]': 76,
    'Electrode height [m]':0.2575, #length of positive electrode
    'Electrode width [m]': 0.0984, # width of positive electrode
    'Number of electrodes connected in parallel to make a cell': 39*2.03,  #p76 has 39 layers, multiply 2 as each layer has two sides, 2.xx  is a simple electrode balancing, should be tuned based on different initial capacity due to cell-to-cell variance.
    'Number of cells connected in series to make a battery': 1.0,

    # design parameter
     ## negative
    'Negative electrode thickness [m]': 80.12e-6, #from FN p2d
    'Negative electrode porosity': 0.25893, #from FN p2d
    'Negative electrode active material volume fraction': 0.69307,  #from FN p2d
    'Negative particle radius [m]': 6.2e-6, #from FN p2d
    'Negative electrode Bruggeman coefficient (electrode)': 1.5, # assumed
    'Negative electrode Bruggeman coefficient (electrolyte)': 1.5, # assumed  
     ## separator
    'Separator thickness [m]': 12e-6, #from FN p2d
    'Separator porosity': 0.38, #from FN p2d
    'Separator Bruggeman coefficient (electrolyte)': 1.5, # assumed
     ## positive
    'Positive electrode thickness [m]': 57.955e-06, #from FN p2d
    'Positive electrode porosity': 0.26205, #from FN p2d
    'Positive electrode active material volume fraction': 0.69295, # from FN p2d
    'Positive particle radius [m]': 3e-6, # from FN p2d
    'Positive electrode Bruggeman coefficient (electrode)': 1.5, # assumed
    'Positive electrode Bruggeman coefficient (electrolyte)': 1.5, # assumed

    # material properties - electrode
    'Negative electrode exchange-current density [A.m-2]': anode_exchange_current_density,
    'Negative electrode OCP [V]': anode_ocp,
    'Negative electrode OCP entropic change [V.K-1]': 0, # useless but necessary,should be used when not under 298K
    'Maximum concentration in negative electrode [mol.m-3]': csmax_neg,
    'Negative electrode diffusivity [m2.s-1]': anode_diffusivity,
    'Negative electrode conductivity [S.m-1]': 27.73, #from fn p2d
    'Negative electrode charge transfer coefficient': 0.5, 
    'Positive electrode exchange-current density [A.m-2]': cathode_exchange_current_density,
    'Positive electrode OCP [V]': cathode_ocp,
    'Positive electrode OCP entropic change [V.K-1]': 0,# useless but necessary
    'Maximum concentration in positive electrode [mol.m-3]': csmax_pos,
    'Positive electrode diffusivity [m2.s-1]': cathode_diffusivity,
    'Positive electrode conductivity [S.m-1]': 0.134, #from fn p2d
    'Positive electrode charge transfer coefficient': 0.5, # assumed

    # material properties - electrolyte
    'Electrolyte conductivity [S.m-1]': 0.7, # from fn p2d
    'Electrolyte diffusivity [m2.s-1]': electrolyte_diffusivity,
    'Cation transference number': 0.363, # privided by Farasis, which is similar to the WangCY-Nature
    'Thermodynamic factor': 1, # assumed

    ### Initial parameters 
    'Initial concentration in electrolyte [mol.m-3]': 1000.0, # assumed
    'Bulk solvent concentration [mol.m-3]': 1000.0, # assumed
    'Initial concentration in negative electrode [mol.m-3]': 1554.3,
    'Initial concentration in positive electrode [mol.m-3]': 45421,
    # 'Initial concentration in negative electrode [mol.m-3]': Cs_max_n*(theta_n1),
    # 'Initial concentration in positive electrode [mol.m-3]': Cs_max_p*(theta_p0),    
    ### Operating conditions 
    'Current function [A]': -76, # will be overwirte in 'battery_simulator'
    #'Typical current [A]': 76, # useless but necessary
    'Lower voltage cut-off [V]': 2.75,
    'Upper voltage cut-off [V]': 4.2,
    'Ambient temperature [K]': 298.15, # useless but necessary
    'Initial temperature [K]': 298.15, # useless but necessary
    'Reference temperature [K]': 298.15, # useless but necessary
    ### Currrent collector
    'Negative current collector conductivity [S.m-1]': 58411000.0, # useless but necessary
    'Positive current collector conductivity [S.m-1]': 36914000.0, # useless but necessary
    ### Other parameters
    'Negative electrode electrons in reaction': 1, # useless but necessary
    'Positive electrode electrons in reaction': 1, # useless but necessary
  


  ## custom parameters
    'External pressure [Pa]': 1000000.0,
}

def get_parameter_values():
    return params