import matplotlib.pyplot as plt
from helpers import quadratic_func, chi2_interpol_func

def plot_chi2_scan(mu_values, chi2_values, params, ndf, planet_id, lambdas):
    fig, ax = plt.subplots(figsize=(16, 5))
    min_chi2 = params['a']
    ax.plot(mu_values, chi2_values, '.', alpha=0.3, label='chi2/ndf')
    ax.plot(mu_values, chi2_interpol_func(mu_values, *params), '-', label='parabola')
    ax.set_title(f'Relative stellar flux darkening (planet_id = {planet_id}), lambda: {lambdas[0]}-{lambdas[1]}', fontsize=13)
    ax.axhline(min_chi2, color="r", linestyle="--")
    ax.axhline(min_chi2+1/ndf, color="r", linestyle="--")
    ax.set_xlabel('mu', fontsize=13)
    ax.set_ylabel('chi2/NDF', fontsize=13)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=13, loc='upper left')
    plt.ylim(top=min_chi2+30/ndf)
    plt.ylim(bottom=min_chi2-10/ndf)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_bkg_hypothesis(x, y_template, y_template_errors, best_bkg_pars):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.errorbar(x, y_template, y_template_errors, fmt="ok", label="data + template w/ best mu")
    ax.plot(x, quadratic_func(x, *best_bkg_pars), label="BKG-only hypothesis (pol2)")
    ax.set_xlabel('time units', fontsize=13)
    ax.set_ylabel('Events', fontsize=13)
    ax.legend(fontsize=13, loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()