import zfit
import tensorflow
import tqdm
import numpy as np
import matplotlib.pyplot as plt

class PlanetTimePDF(zfit.pdf.ZPDF):
    _N_OBS = 1  # dimension, can be omitted
    _PARAMS = ['center', 'width', 'tan'
              ]  # the name of the parameters
    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)  # returns a list with the columns: do x, y, z = z.unstack_x(x) for 3D
        center = self.params['center']
        width = self.params['width']
        tan = self.params['tan']
        a1 = -center + width
        a2 = center + width
        #print(center, width, tan)
        #x = zfit.models.polynomials.rescale_minus_plus_one(x, limits=self.space)
        return  1/(1+tensorflow.exp(-tan*(x+a1)) )- 1/(1+tensorflow.exp(-tan*(x-a2)))
    
def get_pdf(params=None, i=0, floating=True, obs=None):
    if params is None:
        params = {
            f'c0_{i}': {'value':-0.0001},
            f'c1_{i}': {'value':0.0},
            f'c2_{i}': {'value':0.0},
            f'tan_{i}': {'value':50},
            f'center_{i}': {'value':0.35},
            f'width_{i}': {'value':0.2},

        }
    #print(params)
    c0 = zfit.Parameter(f"c0_{i}", params[f"c0_{i}"]['value'], -0.002, 0.002#, floating=floating
                       )
    c1 = zfit.Parameter(f"c1_{i}", params[f"c1_{i}"]['value'], -0.0015, 0.0015#, floating=floating
                       )
    c2 = zfit.Parameter(f"c2_{i}", params[f"c2_{i}"]['value'], -0.001, 0.001#, floating=floating
                       )
    
    bkg_pdf =  zfit.pdf.Chebyshev2(obs=obs, coeffs=[c0,c1,c2])
    
    tan = zfit.Parameter(f"tan_{i}", params[f"tan_{i}"]['value'], 50, 2000, floating=floating)
    center = zfit.Parameter(f"center_{i}",params[f"center_{i}"]['value'], 0.35, 0.65, floating=floating)
    width = zfit.Parameter(f"width_{i}", params[f"width_{i}"]['value'], 0.05, 0.25, floating=floating)
    
    sig_pdf = PlanetTimePDF(obs=obs, center=center, width=width, tan=tan)
    frac = zfit.Parameter(f"frac_{i}", -0.0002,  -0.01, -0.000005)
    
    #center = zfit.Parameter(f"center_{i}", 0.2, 0.1, 0.9, floating=floating)
    #width = zfit.Parameter(f"width_{i}", 0.2, 0.05, 0.5, floating=floating)
    
    #gauss_pdf = zfit.pdf.Gauss(obs=obs, sigma=width, mu=center)
    
    pdf = zfit.pdf.BinnedSumPDF([sig_pdf, bkg_pdf], fracs = frac)#.to_binned(obs)
    
    return pdf

def fit_planet(data_train, planet_id=0, lambdas=(0,283), params=None, floating=True):
    binning = np.linspace(0,1,data_train.shape[1])
    obs = zfit.Space('Time', limits=(0,1), binning=data_train.shape[1])
    data = zfit.data.BinnedData.from_tensor(obs, data_train[planet_id][:,lambdas[0]:lambdas[1],:].sum(axis=2).mean(axis=1))

    pdf = get_pdf(params=params, floating=floating, obs=obs)

    # Stage 1: create a binned likelihood with the given PDF and dataset
    nll = zfit.loss.BinnedNLL(model=pdf, data=data)

    # Stage 2: instantiate a minimiser (in this case a basic minuit minimizer)
    minimizer = zfit.minimize.Minuit()

    # Stage 3: minimise the given negative likelihood
    result = minimizer.minimize(nll)

    param_hesse = result.hesse()
    del data
    return obs, pdf, result