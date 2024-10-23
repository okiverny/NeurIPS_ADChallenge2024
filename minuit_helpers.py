import numpy as np
from iminuit import Minuit

def polynomial(x, c0, c1, c2, c3):
    pol = np.sum(p * (x)**(i) for i, p in enumerate([c0, c1, c2, c3]))
    return pol

def step_function(x, center, width, tan):
        a1 = -center + width
        a2 = center + width
        return  1/(1+np.exp(-tan*(x+a1)) )- 1/(1+np.exp(-tan*(x-a2)))
    
def eclipse(x, c0, c1, c2, c3, center, width, tan, frac):
    pol = polynomial(x, c0, c1, c2, c3)
    step = step_function(x, center, width, tan)
    return pol-frac*step

def fit_transit(y_data, new_params:dict=None, fix_params:dict=None, limit_params:dict=None, log=False):
    x_data = np.linspace(0, 1, 187)
    y_errors = np.sqrt(y_data)  
    y_errors = y_errors/y_data.sum() 
    y_data = y_data/y_data.sum()
    
    def cost_function(c0, c1, c2, c3, center, width, tan, frac):
        return np.sum(((y_data - eclipse(x_data, c0, c1, c2, c3, center, width, tan, frac)) / y_errors) ** 2)

    initial_params = {'c0': 5.0e-3, 'c1':-5.0e-6, 'c2':-5.0e-6, 'c3':3.e-6, 'center':0.5, 'width':0.17, 'tan':250, 'frac':1.4e-5}
    if new_params:
        initial_params.update(new_params)
    m = Minuit(cost_function,  **initial_params)
    if fix_params:
        for fix_par in fix_params:
            m.fixto(fix_par, initial_params[fix_par])
    #m.limits['c1'] = (1e-6, 1e-2)
    m.limits['frac'] = (1e-6, 1e-2)
    m.limits['tan'] = (50, 2000)
    m.limits['center'] = (0.4, 0.6)
    m.limits['width'] = (0.05, 0.45)
    m.migrad()
    m.hesse()
    if log:
        print(m)

    result = {
        'values': m.values,
        'errors': m.errors,
    }

    return result
