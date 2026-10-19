#%%
from numpy import average
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import importlib
import helper_functions 
importlib.reload(helper_functions)
from helper_functions import linear_correction_for_g2
#%%
class TTCProcessor:
    """Class to handle Time-Time Correlation (TTC) data processing and binning."""

    @staticmethod
    def exp_fun(x, beta, A, tau, kww):
        """Exponential function for fitting the g2."""
        return A + beta * np.exp(-2 * (x / tau)**kww)

    @staticmethod
    def kww_S(t, tau, alpha):
        return np.exp(-2.0 * (t / tau) ** alpha)

    @staticmethod
    def ttc_vertical(ttc, time_delays_in_second):
        """Extracts horizontal cuts of TTC and removes self-correlation (time=0)."""
        shape = np.shape(ttc)[0]
        ttc_v = np.ones([shape, shape]) * np.nan
        for i in range(shape):
            m = shape - i
            ttc_v[i, 0:m] = np.diagonal(ttc, offset=i)
        # Self correlation is removed here, i.e., t_delay = 0 is removed
        return ttc_v[1:, :-1], time_delays_in_second[1:]

    @staticmethod
    def logbin_tdelay(t, n):
        """Generates log binning for a given time delay vector."""
        epsilon = 1e-12 * t[0]
        bins = np.logspace(np.log10(t[0] - epsilon), np.log10(t[-1]), n)

        t_mean = binned_statistic(t, t, statistic='mean', bins=bins)
        t_binned = t_mean.statistic

        t_std = binned_statistic(t, t, statistic='std', bins=bins)
        dt_binned = t_std.statistic
        
        binnumber = t_mean.binnumber
        _, counts = np.unique(binnumber, return_counts=True)

        C = 0
        log_binned = [0]
        for count in counts:
            C += count
            log_binned.append(C)
            
        mask = ~np.isnan(t_binned)
        return np.array(log_binned), t_binned[mask], dt_binned[mask]

    @staticmethod
    def weighted_nanaverage(A, weights, axis=None):
        """Calculates weighted average and error handling NaNs."""
        valid_mask = ~np.isnan(A)
        mean = np.nansum(A * weights, axis=axis) / np.nansum(valid_mask * weights, axis=axis)
        err = 1 / np.sqrt(np.nansum(valid_mask * weights, axis=axis))
        return mean, err

    def g2_section_area(self, ttc, std, time_delays_in_second, num, log_avg_n):
        """Calculates horizontal g2 sections with log-averaging."""
        # Self correlation is excluded
        ttc_v, time = self.ttc_vertical(ttc, time_delays_in_second)
        std_ttc_v, time = self.ttc_vertical(std, time_delays_in_second)
        
        no_ttc_lines_per_cut = int(ttc_v.shape[1] / num)
        section_edges = np.arange(0, ttc_v.shape[1], no_ttc_lines_per_cut).astype(int)
        section_edges = section_edges[:num + 1]
        
        if num == 1:
            section_edges = np.array([0, ttc_v.shape[1]])

        g2_cuts, g2_cuts_err = None, None
        tdelay_section, dtdelay_section = None, None

        for i in range(len(section_edges) - 1):
            start, end = section_edges[i], section_edges[i+1]
            g_len = np.count_nonzero(~np.isnan(ttc_v[:, start]))
            
            log_indices, t_cut, dt_cut = self.logbin_tdelay(time[0:g_len], log_avg_n)

            g2_p, dg2_p = [], []
            for j in range(len(log_indices) - 1):
                area = ttc_v[log_indices[j]:log_indices[j+1], start:end]
                std_area = std_ttc_v[log_indices[j]:log_indices[j+1], start:end]
                
                weights = 1.0 / (std_area + 1e-15)**2
                val, err = self.weighted_nanaverage(area, weights)
                g2_p.append(val)
                dg2_p.append(err)

            g2_p, dg2_p = np.array(g2_p), np.array(dg2_p)

            if i == 0:
                g2_cuts = np.full((num, len(g2_p)), np.nan)
                g2_cuts_err = np.full((num, len(g2_p)), np.nan)
                tdelay_section = np.full((num, len(g2_p)), np.nan)
                dtdelay_section = np.full((num, len(g2_p)), np.nan)

            length = len(g2_p)
            tdelay_section[i, :length] = t_cut
            dtdelay_section[i, :length] = dt_cut
            g2_cuts[i, :length] = g2_p
            g2_cuts_err[i, :length] = dg2_p

        # waiting time calculation
        t_delay_unit = time_delays_in_second[1] - time_delays_in_second[0]
        t_w = t_delay_unit * (section_edges[:-1] + section_edges[1:]) / 2
        return tdelay_section, dtdelay_section, g2_cuts, g2_cuts_err, t_w

    def fit_S_kww(self, t, S, S_err, alpha_bounds=(0.7, 1.2), tau_bounds=(1e-9, 1e2)):
    # initial guesses
        tau0 = np.median(t)
        alpha0 = 1.0

        popt, pcov = curve_fit(
            self.kww_S, t, S,
            p0=(tau0, alpha0),
            bounds=([tau_bounds[0], alpha_bounds[0]],
                    [tau_bounds[1], alpha_bounds[1]]),
            sigma=S_err,
            absolute_sigma=True,
            maxfev=20000
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr


    def fit_g2(self, t, g2, g2_err, initial_beta):
        """
        Fits a single g2 curve using the fixed-contrast logic.
        """
        # Remove NaNs
        mask = ~np.isnan(t) & ~np.isnan(g2)
        t_fit = t[mask]
        g2_fit = g2[mask]
        err_fit = g2_err[mask]

        # Boundaries and Initial Guess from your snippet
        p0 = [initial_beta, 0, 1, 1.0]
        bounds = (
            [initial_beta - 0.001*initial_beta, -2, 1e-9, 0.5], 
            [initial_beta + 0.001*initial_beta, 2, 1e9, 1.9]
        )

        try:
            popt, pcov = curve_fit(
                self.exp_fun, t_fit, g2_fit, sigma=err_fit/100, 
                p0=p0, bounds=bounds, maxfev=100000, absolute_sigma=False
            )
            return popt, pcov
        except Exception as e:
            print(f"Fit failed: {e}")
            return np.full(4, np.nan), np.full((4,4), np.nan)

    def propagate_g2_error(self, t, popt, pcov):
        """Propagates error from parameters to the fitted function."""
        if np.any(np.isnan(popt)) or np.any(np.isnan(pcov)):
            return np.full_like(t, np.nan)

        beta, A, tau, kww = popt
        
        # Avoid division by zero or log of zero/negative
        # Assuming t > 0 and tau > 0
        
        arg = (t / tau)
        term_pow = arg ** kww
        E = np.exp(-2 * term_pow)
        
        # Partial derivatives
        d_beta = E
        d_A = np.ones_like(t)
        d_tau = beta * E * (2 * kww / tau * term_pow)
        d_kww = beta * E * (-2 * term_pow * np.log(arg))
        
        # Jacobian matrix (N_points, N_params)
        jac = np.stack([d_beta, d_A, d_tau, d_kww], axis=1)
        
        # Error propagation: sigma^2 = J * Cov * J.T (diagonal elements only)
        # var = sum_ij (J_ni * Cov_ij * J_nj)
        var = np.einsum('ni,ij,nj->n', jac, pcov, jac)
        return np.sqrt(var)

    def process_and_fit_all_q(self, ttc_stack, se_stack, time_vector, betas_to_use, num_cuts, log_avg_n, correct_g2=True):
        """
        Replicates the loop over all Q-values.
        """
        n_q = len(betas_to_use)
        results = {
            'betas': np.zeros((n_q, num_cuts)), 'taus': np.zeros((n_q, num_cuts)), 
            'As': np.zeros((n_q, num_cuts)), 'kwws': np.zeros((n_q, num_cuts)),
            'dbetas': np.zeros((n_q, num_cuts)), 'dtaus': np.zeros((n_q, num_cuts)), 
            'dAs': np.zeros((n_q, num_cuts)), 'dkwws': np.zeros((n_q, num_cuts)), 
            'g2_list': [],
            'g2_list_corrected': [],
            'tau_fixed': np.zeros((n_q, num_cuts)),
            'dtau_fixed': np.zeros((n_q, num_cuts)),
            'alpha_fixed': np.zeros((n_q, num_cuts)),
            'dalpha_fixed': np.zeros((n_q, num_cuts)),
            'kww_fixed': np.zeros((n_q, num_cuts)),
            'dkww_fixed': np.zeros((n_q, num_cuts)),
            'beta_fixed': np.zeros((n_q, num_cuts)),
            'dbeta_fixed': np.zeros((n_q, num_cuts)),
            'A_fixed': np.zeros((n_q, num_cuts)),
            'dA_fixed': np.zeros((n_q, num_cuts)),
            'fitted_g2_list': [],
            'fitted_g2_err_list': [],
            'err_list': [],
            't_list': [], 
            't_list_fitted': []
        }

        for i in range(n_q):
            # 1. Extract g2
            t_list,dtdelay_section, g2_list, err_list,t_w= self.g2_section_area(
                ttc_stack[i, :100, :100], se_stack[i, :100, :100], 
                time_delays_in_second=time_vector, num=num_cuts, log_avg_n=log_avg_n
            )
            g2_fitted_list = []
            g2_fitted_err_list = []
            g2_corrected_list = []
            t_list_fitted = []
            for n_c in range(num_cuts):
                g_2 =  g2_list[n_c]
                g_2_err = err_list[n_c]
                t_ = t_list[n_c]
                #remove nan values
                mask = ~np.isnan(t_) & ~np.isnan(g_2) & ~np.isnan(g_2_err)
                t_ = t_[mask]
                g_2 = g_2[mask]
                g_2_err = g_2_err[mask]
                
                popt, pcov = self.fit_g2(t_, g_2, g_2_err, betas_to_use[i])
                perr = np.sqrt(np.diag(pcov))

                g_2_fitted = self.exp_fun(t_, *popt)
                g_2_fitted_err = self.propagate_g2_error(t_, popt, pcov)
                
                g2_fitted_list.append(g_2_fitted)
                g2_fitted_err_list.append(g_2_fitted_err)
                t_list_fitted.append(t_)
                
                if correct_g2:
                    g_2_corrected, kww_fixed = linear_correction_for_g2(g_2, popt[1],  popt[0], t_, fit_points=5)
                    def kww_only_func(x, kww):
                        return  popt[1] + popt[0] * np.exp(-2 * (x / popt[2])**kww)
                    try:
                        popt_fixed, perr_fixed_cov = curve_fit(kww_only_func, t_, g_2_corrected, p0=[1], bounds=([0.6], [1.2]))
                        perr_fixed = np.sqrt(np.diag(perr_fixed_cov))
                    except:
                        popt_fixed, perr_fixed = [np.nan], [np.nan]
                        
                    g2_corrected_list.append(g_2_corrected)
                    alpha_fixed = popt_fixed[0]
                    dalpha_fixed = perr_fixed[0]
                    results['betas'][i,n_c], results['As'][i,n_c], results['taus'][i,n_c], results['kwws'][i,n_c] = popt
                    results['dbetas'][i,n_c], results['dAs'][i,n_c], results['dtaus'][i,n_c], results['dkwws'][i,n_c] = perr
                    results['kww_fixed'][i,n_c] = kww_fixed
                    results['dkww_fixed'][i,n_c] = dalpha_fixed
                    results['alpha_fixed'][i,n_c] = alpha_fixed
                    results['dalpha_fixed'][i,n_c] = dalpha_fixed

            # 3. Save parameters
            results['g2_list'].append(g2_list)
            results['fitted_g2_list'].append(g2_fitted_list)
            results['fitted_g2_err_list'].append(g2_fitted_err_list)
            results['err_list'].append(err_list)
            results['t_list'].append(t_list) 
            results['g2_list_corrected'].append(g2_corrected_list)
            results['t_list_fitted'].append(t_list_fitted)
           
            
            
        return results


class TTCPlotter:
    """Class to handle plotting of TTC and g2 data."""
    
    def __init__(self, sample_name, run_numbers):
        self.sample = sample_name
        self.run_nos = run_numbers

    def configure_ax(self, ax, q_val, num_cuts):
        """Standard axis configuration for g2 plots."""
        ax.set_xscale('log')
        ax.set_xlabel(r'$t$ (s)', fontsize=14)
        ax.set_ylabel(r'$g_{2}-1 (q,t)$', fontsize=14)
        ax.legend(title=f'q = {10*q_val:.4f} $(nm^{{-1}})$')
        ax.grid(c='gray', ls=':', alpha=0.7)
        ax.set_title(f'{self.sample}, cuts={num_cuts}\nRuns: {self.run_nos}')
        return ax




if __name__ == "__main__":
    # Parameters from the notebook
    BETAS = np.array([0.05188205, 0.05192311, 0.05136611, 0.05171926, 0.05108087,
                      0.05104867, 0.05020785, 0.04957793, 0.0491428, 0.04859964,
                      0.04808463, 0.04736483, 0.04679808, 0.04616146, 0.04531015,
                      0.04457282, 0.04359917, 0.04177554, 0.04105258, 0.0395628])
    
    Q_VALS = np.array([0.0095, 0.0105, 0.0115, 0.0125, 0.0135,
                       0.0145, 0.0155, 0.0165, 0.0175, 0.0185,
                       0.0195, 0.0205, 0.0215, 0.0225, 0.0235,
                       0.0245, 0.0255, 0.0265, 0.0275, 0.02885438])



    number_of_q_values_eggyolk = 20

    q_values_eggyolk = np.array([0.095     , 0.105     , 0.115     , 0.125     , 0.135     ,
        0.145     , 0.155     , 0.165     , 0.175     , 0.185     ,
        0.195     , 0.205     , 0.215     , 0.225     , 0.235     ,
        0.245     , 0.255     , 0.265     , 0.275     , 0.28854377])

    contrast_eggyolk =np.array([0.05188205, 0.05192311, 0.05136611, 0.05171926, 0.05108087,
        0.05104867, 0.05020785, 0.04957793, 0.0491428 , 0.04859964,
        0.04808463, 0.04736483, 0.04679808, 0.04616146, 0.04531015,
        0.04457282, 0.04359917, 0.04177554, 0.04105258, 0.0395628 ])


    ave_eggyolk = [5,25,50,100,200,400,600,800,1000,1200,1300]

    ttc_eggyolk = np.zeros((len(ave_eggyolk),number_of_q_values_eggyolk, 100,100))

    SE_ttc_eggyolk = np.zeros((len(ave_eggyolk),number_of_q_values_eggyolk, 100, 100))


    for idx, ab in enumerate(ave_eggyolk):
        s_ttc = 1
        print(f'Loading data for {ab} ttc averaging...')
        file_name =f'./yolkplasma/{ab}ttc_averaged/Avg_TTC_{s_ttc}_{ab+s_ttc}.npy'
        file_name_se =f'./yolkplasma/{ab}ttc_averaged/SE_TTC_{s_ttc}_{ab+s_ttc}.npy'
        ttc_eggyolk_ = np.load(file_name)[: number_of_q_values_eggyolk]
        se_ttc_eggyolk_ = np.load(file_name_se)[0:number_of_q_values_eggyolk]
        for id, ttc in enumerate(ttc_eggyolk_):
            np.fill_diagonal(ttc, contrast_eggyolk[id])
            ttc_eggyolk[idx, id] = ttc
        SE_ttc_eggyolk[idx, 0:number_of_q_values_eggyolk] = se_ttc_eggyolk_
    

    ttc_stack_denoised = np.zeros((number_of_q_values_eggyolk, len(ave_eggyolk), 100, 100))
    se_stack_denoised = np.zeros((number_of_q_values_eggyolk, len(ave_eggyolk), 100, 100))
    for q_value_index in range(number_of_q_values_eggyolk):
        ttc_deoinsed = np.load(f'./denoised_ttc_v2/eggyolk/1step_denoising/ttc_eggyolk_denoised_q{q_value_index+1}.npy')
        se_deoinsed = np.load(f'./denoised_ttc_v2/eggyolk/1step_denoising/se_eggyolk_denoised_q{q_value_index+1}.npy')
        ttc_stack_denoised[q_value_index,:,:] = ttc_deoinsed
        se_stack_denoised[q_value_index,:,:] = se_deoinsed




    t_delay = 0.44e-6
    delays = np.arange(0, 200) * t_delay
    time_binned = delays.reshape(100, 2).mean(axis=1) # 100 points

    RUN_NOS = [244]
    SAMPLE = 'yp_pure_T=9.73e-5'

    # Load data (Assuming ttc_stack and se_stack are loaded 2D arrays)
    # ttc_stack shape: (n_q, 100, 100)
    # se_stack shape: (n_q, 100, 100)
    
    # Example: Load your data here
    # ttc_stack = np.load('your_ttc_data.npy')
    # se_stack = np.load('your_se_data.npy')
    processor = TTCProcessor()
    # Run the processing
    average_index = 5
    results_raw = processor.process_and_fit_all_q(
        ttc_eggyolk[average_index,:,:], SE_ttc_eggyolk[average_index,:,:], time_binned, BETAS, num_cuts=1, log_avg_n=30
    )

    results_denoised = processor.process_and_fit_all_q(
        ttc_stack_denoised[:,0,:], se_stack_denoised[:,0,:], time_binned, BETAS, num_cuts=1, log_avg_n=30
    )

    # Plotting
    plotter = TTCPlotter(SAMPLE, RUN_NOS)
    
    print(f"Processor and Plotter initialized for sample: {SAMPLE}")