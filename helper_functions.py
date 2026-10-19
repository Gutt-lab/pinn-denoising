import matplotlib.pyplot as plt
import sys
import numpy as np
# NumPy 2.0+ compatibility: np.trapz was renamed to np.trapezoid
if not hasattr(np, 'trapz'):
    np.trapz = getattr(np, 'trapezoid', None)
if not hasattr(np, 'trapezoid'):
    np.trapezoid = getattr(np, 'trapz', None)
from scipy.optimize import nnls
from scipy.stats import norm, linregress
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.gridspec as gridspec
import math
from scipy.ndimage import rotate, geometric_transform
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import chi2


def rotate_ttc_45(ttc_matrix, angle=-45):
    """
    Rotates the TTC matrix by 45 degrees so the Main Diagonal becomes Horizontal.
    
    Args:
        ttc_matrix: Input 2D square array.
        angle: -45 usually makes the diagonal horizontal.
    """
    # 1. Rotate
    # reshape=True ensures we don't crop the corners (makes the image larger)
    # order=1 (Bilinear) or 0 (Nearest) preserves data best. 3 (Cubic) looks smoother.
    # cval=np.nan fills the empty corners with NaNs instead of zeros.
    ttc_rotated = rotate(ttc_matrix, angle, reshape=True, order=1, cval=np.nan)
    
    # 2. Crop (Optional)
    # The rotation creates a large diamond with lots of empty space.
    # We can crop it to the center to zoom in on the data.
    non_empty_rows = np.where(~np.isnan(ttc_rotated).all(axis=1))[0]
    non_empty_cols = np.where(~np.isnan(ttc_rotated).all(axis=0))[0]
    
    if len(non_empty_rows) > 0 and len(non_empty_cols) > 0:
        crop_r_start, crop_r_end = non_empty_rows[0], non_empty_rows[-1]
        crop_c_start, crop_c_end = non_empty_cols[0], non_empty_cols[-1]
        ttc_rotated = ttc_rotated[crop_r_start:crop_r_end, crop_c_start:crop_c_end]

    # 3. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    ax1.imshow(ttc_matrix, origin='lower', cmap='viridis')
    ax1.set_title("Original TTC\n(Diagonal is tilted)")
    ax1.plot([0, ttc_matrix.shape[1]], [0, ttc_matrix.shape[0]], 'w--', alpha=0.5)
    
    # Rotated
    ax2.imshow(ttc_rotated, origin='lower', cmap='viridis')
    ax2.set_title("Rotated TTC\n(Diagonal is Horizontal)")
    # Draw a line through the center
    mid_y = ttc_rotated.shape[0] // 2
    ax2.axhline(mid_y, color='w', linestyle='--', alpha=0.5, label='Main Diagonal')
    
    ax2.set_xlabel("Age (Experimental Time)")
    ax2.set_ylabel("Lag (tau)")
    
    plt.tight_layout()
    plt.show()
    
    return ttc_rotated

def apply_lorentz_transformation(ttc_matrix):
    """
    Applies a coordinate transformation (Lorentz-like rotation) to the TTC matrix.
    Converts (t1, t2) -> (Age, Lag).
    
    The result is a matrix where:
    - X-axis = System Age (t1 + t2)/2
    - Y-axis = Lag Time (t2 - t1)
    """
    rows, cols = ttc_matrix.shape
    center = (rows // 2, cols // 2)
    
    # 1. Define the Mapping Function (Inverse Map)
    # The transform function asks: "For pixel (r, c) in the OUTPUT, 
    # where should I look in the INPUT?"
    # Output coords: (lag, age) -> Input coords: (t1, t2)
    def mapping(coords):
        lag, age = coords # r, c in output
        
        # Shift origin to center for rotation, then shift back is standard,
        # but here we map directly to t1, t2 physical constraints.
        # We need to adjust indices because 'age' runs 0..N and 'lag' runs 0..N
        # But physically, age is roughly t1+t2.
        
        # Simple 45-degree rotation logic:
        # t1 = age - lag/2
        # t2 = age + lag/2
        
        # We adjust scale to keep the image centered
        # Center of output (N/2, N/2) maps to Center of input (N/2, N/2)
        
        r_out, c_out = coords
        
        # Center-relative coordinates
        y = r_out - center[0] # vertical (lag)
        x = c_out - center[1] # horizontal (age)
        
        # Inverse Rotation Matrix for 45 degrees
        # [ t1 ] = [  cos  sin ] [ x ]
        # [ t2 ] = [ -sin  cos ] [ y ]
        # sin(45) = cos(45) = 1/sqrt(2)
        
        inv_sqrt2 = 1.0 / np.sqrt(2)
        
        t1 = (x * inv_sqrt2) - (y * inv_sqrt2) + center[0]
        t2 = (x * inv_sqrt2) + (y * inv_sqrt2) + center[1]
        
        return (t2, t1) # scipy uses (row, col) convention

    # 2. Apply the Geometric Transform
    # order=1 (Linear) is usually best tradeoff. order=0 (Nearest) is aliased.
    ttc_transformed = geometric_transform(ttc_matrix, mapping, order=1, cval=np.nan)
    
    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Input
    ax1.imshow(ttc_matrix, origin='lower', cmap='viridis')
    ax1.set_title("Original TTC\n$(t_1, t_2)$ Coordinates")
    ax1.set_xlabel("Time 1 ($t_1$)")
    ax1.set_ylabel("Time 2 ($t_2$)")
    # Draw diagonal
    ax1.plot([0, cols], [0, rows], 'w--', alpha=0.5)

    # Output
    ax2.imshow(ttc_transformed, origin='lower', cmap='viridis')
    ax2.set_title("Lorentz Transformed\n(Age vs. Lag) Coordinates")
    ax2.set_xlabel("System Age ($t_{age}$)")
    ax2.set_ylabel("Lag Time ($\\tau$)")
    # Draw horizontal center line
    ax2.axhline(rows//2, color='w', linestyle='--', alpha=0.5, label='Zero Lag')
    
    plt.tight_layout()
    plt.show()
    
    return ttc_transformed


def linear_correction_for_g2(g2, B, beta, t, fit_points=30):
    
    g1 = (g2 - B) / (beta+ 1e-15)
    mask = (g1 > 0) & (g1 < 1)
    if np.sum(mask) < fit_points:
        print("Not enough points to fit")
        return g2, 0
    g1_in_log_space = np.log(-0.5 * np.log(g1[mask]))
    t_in_log_space = np.log(t[mask])
    length = len(t_in_log_space)
    coeffs = np.polyfit(t_in_log_space, g1_in_log_space, 1)

    # 5. Reconstruct the full curve using the slope and intercept
    # Note: We apply the fit to the full original 't' vector
    full_t_log = np.log(t)
    g1_loglog_fit = coeffs[0] * full_t_log + coeffs[1]


    g1_corrected = np.exp(-2 * np.exp(g1_loglog_fit))
    g2_corrected = (beta * g1_corrected) + B
    g2[:2] = g2_corrected[:2]
    g2[-2:] = g2_corrected[-2:]

    return g2_corrected, coeffs[0]
def analyze_residuals(y_noisy, y_denoised, dt=1.0, qvalue=0.0, plot=True):
    """
    Analyzes residuals and prints a PASS/FAIL verdict directly on the plot.
    """
    
    # --- 1. PREPARE DATA ---
    # Flatten to ensure 1D array regardless of input shape
    residuals = (y_noisy - y_denoised).flatten()
    
    # --- 2. CALCULATE METRICS ---
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    
    x_min, x_max = -5*res_std, 5*res_std
    x = np.linspace(x_min, x_max, 100)
    p = norm.pdf(x, res_mean, res_std)
    # Autocorrelation (Structure Check)
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    else:
        autocorr = 0

    # --- 3. DETERMINE PASS/FAIL ---
    # Thresholds:
    # 1. Bias: Mean must be less than 10% of the standard deviation
    # 2. Structure: Autocorrelation must be low (< 0.2 is strict, < 0.5 is loose)
    
    is_unbiased = abs(res_mean) < (0.2 * res_std)
    is_random = abs(autocorr) < 0.25  # You can adjust this threshold
    
    if is_unbiased and is_random:
        verdict = "PASSED: Physics Preserved"
        title_color = "green"
    elif is_unbiased:
        verdict = "WARNING: Some Structure Left (Autocorr High)"
        title_color = "orange"
    else:
        verdict = "FAILED: Significant Bias or Structure"
        title_color = "red"

    # --- 4. FFT COMPUTATION ---
    n = len(residuals)
    freq = np.fft.rfftfreq(n, d=dt)
    spectrum = np.abs(np.fft.rfft(residuals))**2 / n

    
    if plot:
    # --- 5. PLOTTING ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        # Plot A: Residuals (Time/Index)
        axes[0].plot(residuals, color='#6a0dad', alpha=0.5, linewidth=0.7)
        axes[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        #axes[0].set_title('Residuals (Flattened)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, linestyle=':', alpha=0.3)

        # Plot B: Histogram
        axes[1].hist(residuals, bins=50, density=True, color='gray', alpha=0.6, label='Data')
        
        # Overlay Ideal Gaussian
        axes[1].plot(x, p, 'r', linewidth=2, label='Ideal Gaussian')
        #axes[1].set_title(f'Residual distribution')
        axes[1].legend(loc='upper right')

        # Plot C: Power Spectrum
        #axes[2].semilogy(freq, spectrum, color='teal', alpha=0.8)
        #axes[2].set_title(f'Power Spectrum (FFT)')
        #axes[2].set_xlabel('Frequency')
        
        # --- SUPER TITLE WITH VERDICT ---
        #plt.suptitle(f"q = {qvalue} \n {verdict}\n(Bias Check: {'OK' if is_unbiased else 'FAIL'} | "
                    #f"Structure Check: {'OK' if is_random else 'FAIL'} | Autocorr: {autocorr:.2f})", 
                    #fontsize=16, color=title_color, weight='bold')
        
        plt.tight_layout()
        plt.savefig(f'./plotPINN/ttc_residuals_q{qvalue}.png', dpi=300)
        #plt.show()

    return residuals, p, x, freq, spectrum

def plot_ttcs_time_freq(ttc_array, 
                        dt=1.0, 
                        main_title="", 
                        title_prefix="TTC Data", 
                        label_prefix=[], 
                        qvalue=0.05, 
                        rebin_factor=1, 
                        y_lim_min=None, 
                        y_lim_max=None, 
                        x_lim_max=None, 
                        x_lim_min=None):
    """
    Plots TTC curves in Time and Frequency domains.
    
    UPDATES:
    1. Removes NaNs from the input arrays (critical for Rectified data).
    2. Trims all curves to the minimum valid length to ensure array consistency.
    3. Sorts by amplitude for visibility.
    """

    cleaned_list = []
    for ttc in ttc_array:
        # Extract only valid pixels (ignore the NaNs from the triangle)
        valid_data = ttc[~np.isnan(ttc)]
        cleaned_list.append(valid_data)
    
    # Ensure all curves have the same length for the numpy array
    min_len = min(len(c) for c in cleaned_list)
    ttc_array_flat = np.array([c[:min_len] for c in cleaned_list])
    
    # --- 2. SORT BY AMPLITUDE ---
    amplitudes = np.max(ttc_array_flat, axis=1)
    sorted_indices = np.argsort(amplitudes)[::-1]
    
    # --- 3. PREPARE TIME AXIS & REBINNING ---
    num_curves, num_points = ttc_array_flat.shape
    new_len = num_points - (num_points % rebin_factor)
    trimmed_array = ttc_array_flat[:, :new_len]
    
    # Rebinning
    rebinned_array = trimmed_array.reshape(num_curves, new_len // rebin_factor, rebin_factor).mean(axis=2)
    
    new_dt = dt * rebin_factor
    num_points_new = rebinned_array.shape[1]
    t_rebinned = np.arange(num_points_new) * new_dt

    # --- 4. PREPARE FREQUENCY AXIS ---
    freqs = np.fft.rfftfreq(num_points_new, d=new_dt)
    
    # --- 5. SETUP LAYOUT (GridSpec) ---
    # We need a layout with 1 Row and 3 "Areas".
    # Area 1: Time Plot
    # Area 2: Freq Plot
    # Area 3: A nested grid for the images
    
    fig = plt.figure(figsize=(18, 6))
    plt.suptitle(f"{main_title} \n q = {qvalue}", fontsize=14)
    
    # Define Outer Grid (1 row, 3 cols)
    outer_grid = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.2)
    
    # Create the main axes
    ax_time = fig.add_subplot(outer_grid[0])
    ax_freq = fig.add_subplot(outer_grid[1])
    
    # Define Inner Grid for Images (Area 3)
    # Calculate how many rows/cols we need for the thumbnails
    n_imgs = len(ttc_array)
    n_cols_img = int(math.ceil(math.sqrt(n_imgs)))
    n_rows_img = int(math.ceil(n_imgs / n_cols_img))
    
    inner_grid = gridspec.GridSpecFromSubplotSpec(n_rows_img, n_cols_img, 
                                                  subplot_spec=outer_grid[2], 
                                                  wspace=0.1, hspace=0.2)
    
    colors = cm.jet(np.linspace(0, 1, num_curves))
    
    # --- 6. MAIN LOOP ---
    for i, idx in enumerate(sorted_indices):
        curve = rebinned_array[idx]
        
        # Handle Labels
        if isinstance(label_prefix, list) and idx < len(label_prefix):
            base_label = label_prefix[idx]
        else:
            base_label = f"{title_prefix} {idx}"

        # --- A. Frequency Domain (FFT & Fit) ---
        fft_mag = np.abs(np.fft.rfft(curve))
        label_with_stats = base_label

        # Plot FFT
        ax_freq.loglog(freqs, fft_mag, color=colors[i], alpha=0.7, linewidth=1, label=label_with_stats)

        # --- B. Time Domain ---
        ax_time.plot(t_rebinned, curve, color=colors[i], alpha=0.7, linewidth=1.5, label=base_label)
        
        # --- C. 2D Matrix Grid ---
        # Instead of overwriting, we add a subplot to the specific slot in the inner grid
        if i < (n_rows_img * n_cols_img):
            ax_img = fig.add_subplot(inner_grid[i])
            ax_img.imshow(ttc_array[idx], aspect='auto', cmap='viridis', origin='lower')
            
            # Label with the same color as the curve for easy matching
            ax_img.set_title(base_label, fontsize=9, color='black') # or color=colors[i]
            
            # Add a colored border or text to match the curve
            for spine in ax_img.spines.values():
                spine.set_edgecolor(colors[i])
                spine.set_linewidth(2)
                
            ax_img.set_xticks([])
            ax_img.set_yticks([])

    # --- 7. FORMATTING ---
    
    # Time Domain
    ax_time.set_title("Flattened Signal (Time Domain)")
    ax_time.set_xlabel("Time Step")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True, alpha=0.3)
    if y_lim_min is not None and y_lim_max is not None:
        ax_time.set_ylim(y_lim_min, y_lim_max)
    if x_lim_max is not None:
        ax_time.set_xlim(0, x_lim_max)
    ax_time.legend(fontsize='small', loc='upper right')

    # Freq Domain
    ax_freq.set_title("FFT Spectrum")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude")
    ax_freq.grid(True, which="both", alpha=0.3)
    ax_freq.legend(fontsize='small', loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_diagonal_fft(ttc_matrix, 
                      dt=1.0, 
                      title="Diagonal TTC Analysis", 
                      cmap='inferno'):
    """
    Performs 2D FFT on a Diagonal TTC matrix (Time vs Time).
    Visualizes the characteristic 'Anti-Diagonal Streak' in the frequency domain.
    
    Args:
        ttc_matrix: Square 2D numpy array (Time vs Time).
        dt: Sampling time (used for axis scaling).
        title: Plot title.
    """
    N = ttc_matrix.shape[0]
    
    # --- 1. COMPUTE 2D FFT ---
    # fft2 computes the 2-dimensional transform
    fft_result = np.fft.fft2(ttc_matrix)
    
    # fftshift moves the Zero Frequency (DC) component to the center of the image
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Calculate Magnitude (Power Spectrum)
    magnitude = np.abs(fft_shifted)
    
    # --- 2. SETUP AXES (Physical Units) ---
    # Frequency axis runs from -Nyquist to +Nyquist
    freq_axis = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    extent_freq = [freq_axis[0], freq_axis[-1], freq_axis[0], freq_axis[-1]]
    
    # Time axis
    extent_time = [0, N*dt, 0, N*dt]

    # --- 3. VISUALIZATION ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot A: Time Domain (Original)
    im1 = axes[0].imshow(ttc_matrix, origin='lower', cmap='viridis', 
                         extent=extent_time)
    axes[0].set_title(f"{title}\nTime Domain (Input)")
    axes[0].set_xlabel("Time ($t_1$) [s]")
    axes[0].set_ylabel("Time ($t_2$) [s]")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot B: Frequency Domain (2D FFT)
    # We use LogNorm because the DC component is massive compared to high freq
    im2 = axes[1].imshow(magnitude, origin='lower', cmap=cmap, norm=LogNorm(),
                         extent=extent_freq)
    
    axes[1].set_title(f"{title}\nFrequency Domain (2D FFT)")
    axes[1].set_xlabel("Frequency ($f_1$) [Hz]")
    axes[1].set_ylabel("Frequency ($f_2$) [Hz]")
    
    # Add crosshairs at DC (0,0) for reference
    axes[1].axvline(0, color='white', alpha=0.3, linewidth=0.5, linestyle='--')
    axes[1].axhline(0, color='white', alpha=0.3, linewidth=0.5, linestyle='--')
    
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Magnitude (Log Scale)')
    
    plt.tight_layout()
    plt.show()

def analyze_diagonal_fft_slice(
    ttc_matrix,
    dt=1.0,
    title="TTC",
    f_min=None,
    f_max=None,
    eps=1e-12
):
    """
    2D FFT of TTC(t1,t2), extract anti-diagonal ridge (ω1 = -ω2),
    and analyze its frequency spectrum.
    """

    # -------------------------
    # 0) PREPARE DATA
    # -------------------------
    x = np.asarray(ttc_matrix, float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    rows, cols = x.shape
    N = min(rows, cols)

    # Center crop to NxN (important!)
    r0 = (rows - N) // 2
    c0 = (cols - N) // 2
    x = x[r0:r0+N, c0:c0+N]

    # Remove DC
    x -= x.mean()
    w = np.hanning(N)
    win = np.outer(w, w)
    # -------------------------
    # 1) WINDOWING
    # -------------------------
    xw = x * win

    # -------------------------
    # 2) 2D FFT
    # -------------------------
    F = np.fft.fftshift(np.fft.fft2(xw))
    P = np.abs(F)**2   # POWER spectrum

    # -------------------------
    # 3) RIDGE EXTRACTION (YOUR REQUEST)
    # -------------------------
    ridge = np.fliplr(P).diagonal()


    # -------------------------
    # 4) FREQUENCY AXIS
    # -------------------------
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    center = N // 2

    # positive frequencies only (skip DC)
    freqs_1d = freqs[center+1:]
    ridge_1d = ridge[center+1:]

    # -------------------------
    # 5) FIT RANGE
    # -------------------------
    if f_min is None:
        f_min = freqs_1d[0]
    if f_max is None:
        f_max = freqs_1d[-1]

    mask = (
        (freqs_1d >= f_min) &
        (freqs_1d <= f_max) &
        (ridge_1d > eps)
    )

    x_fit = freqs_1d[mask]
    y_fit = ridge_1d[mask]

    if len(x_fit) >= 3:
        lx = np.log10(x_fit)
        ly = np.log10(y_fit)
        res = linregress(lx, ly)
        slope = res.slope
        intercept = res.intercept
        fit_y = 10**(intercept + slope * lx)
    else:
        slope = np.nan
        fit_y = None

    # -------------------------
    # 6) PLOTTING
    # -------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    # ---- A) 2D FFT MAP ----
    im = ax1.imshow(np.log10(P + eps), origin="lower", cmap="inferno")
    ax1.set_title(f"{title}\n2D FFT |TTC|²")
    ax1.plot([0, N-1], [N-1, 0], 'c--', lw=1.5, label="ω₁ = −ω₂ ridge")
    ax1.legend()
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="log₁₀ Power")

    # ---- B) RIDGE SPECTRUM ----
    ax2.loglog(freqs_1d, ridge_1d, 'o-', color='gray',
               markersize=3, alpha=0.6, label="Ridge spectrum")

    if fit_y is not None:
        ax2.loglog(x_fit, fit_y, 'k--', lw=2,
                   label=f"Fit slope α = {slope:.2f}")

        # Reference slope -2
        mid = len(x_fit) // 2
        ref_x = x_fit[mid]
        ref_y = fit_y[mid]
        ref_line = ref_y * (freqs_1d / ref_x)**(-2)
        ax2.loglog(freqs_1d, ref_line, 'b:',
                   alpha=0.6, label="Ref slope −2")

    ax2.axvline(f_min, color='red', ls='--', alpha=0.3)
    ax2.axvline(f_max, color='red', ls='--', alpha=0.3)

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.set_title("Anti-diagonal ridge spectrum")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    ax3.plot(ridge)
    ax3.set_title("Raw Ridge")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()

    return slope

def analyze_averaged_fft(rectified_ttc, dt=1.0):
    """
    Computes the FFT of each vertical column INDIVIDUALLY, 
    then averages the Power Spectra.
    
    ELIMINATES the "Stitching/Sawtooth" artifact.
    """
    rows, cols = rectified_ttc.shape
    psd_list = []
    
    # 1. Loop through every column (every single decay curve)
    for i in range(cols):
        # Get one curve
        curve = rectified_ttc[:, i]
        
        # Clean NaNs
        valid_curve = curve[~np.isnan(curve)]
        
        if len(valid_curve) > 10:
            # Compute Power Spectrum (Magnitude Squared)
            fft_mag = np.abs(np.fft.rfft(valid_curve))**2
            
            # Normalize length differences (simplification: interp to common grid)
            # For quick check, we just append if length matches max_lag
            # (assuming most columns are full length)
            if len(fft_mag) == (rows // 2) + 1: 
                psd_list.append(fft_mag)
            elif len(valid_curve) == rows: # If full length
                psd_list.append(fft_mag)

    # 2. Average the Spectra
    if len(psd_list) > 0:
        avg_psd = np.mean(psd_list, axis=0)
        freqs = np.fft.rfftfreq(rows, d=dt)
        
        # 3. Plot
        plt.figure(figsize=(8, 6))
        plt.loglog(freqs[1:], avg_psd[1:], 'b-', label='Averaged PSD (Physics)')
        
        # Add Reference Lorentzian (-2 slope)
        mid = len(freqs)//2
        ref = avg_psd[mid] * (freqs/freqs[mid])**(-2)
        plt.loglog(freqs[1:], ref[1:], 'k--', alpha=0.5, label='Lorentzian Ref (-2.0)')
        
        plt.title("True Physics Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.show()
    else:
        print("Not enough data to average.")


def analyze_diagonal_projection_fft(ttc_diagonal, dt=1.0, title="Projected Diagonal Analysis"):
    """
    1. Projects the 2D Diagonal Matrix into a single 1D g2(tau) curve.
    2. Computes the FFT of that clean curve.
    Result: The cleanest possible PSD with NO stitching artifacts.
    """
    N = ttc_diagonal.shape[0]
    
    # --- STEP 1: PROJECTION (Collapse to 1D) ---
    # We calculate the mean of every diagonal offset (lag)
    g2_curve = []
    lags = np.arange(N)
    
    for tau in lags:
        # Extract the pixels for this lag
        diag_vals = np.diagonal(ttc_diagonal, offset=tau)
        # Average them (this kills the random noise)
        g2_curve.append(np.nanmean(diag_vals))
    
    g2_curve = np.array(g2_curve)
    
    # --- STEP 2: PREPARE FOR FFT ---
    # CRITICAL: Remove the baseline (g2 decays to 1.0 or 0.0)
    # If we don't, the DC spike will hide the Lorentzian.
    baseline = np.min(g2_curve[-20:]) # Estimate baseline from the tail
    signal = g2_curve - baseline
    
    # Apply Window to smooth the drop-off at the end (removes spectral leakage)
    window = np.hanning(len(signal))
    signal_windowed = signal * window
    
    # --- STEP 3: COMPUTE PSD ---
    # Real FFT
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    psd = np.abs(np.fft.rfft(signal_windowed))**2
    
    # --- STEP 4: FIT THE TAIL (Physics Check) ---
    # We ignore the first few points (plateau) and fit the slope
    # Heuristic: Fit from index 10 to index N/2
    start_fit = 15
    end_fit = len(freqs) 
    
    if end_fit > start_fit:
        x_fit = np.log10(freqs[start_fit:end_fit])
        y_fit = np.log10(psd[start_fit:end_fit])
        res = linregress(x_fit, y_fit)
        slope = res.slope
        fit_line = 10**(res.intercept + res.slope * x_fit)
    else:
        slope = 0
        x_fit = []
        y_fit = []

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: The Master Curve (Time Domain)
    ax1.plot(lags*dt, g2_curve, 'b-', linewidth=2, label='Projected Mean g2')
    ax1.axhline(baseline, color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax1.set_title("Step 1: Projected Master Curve")
    ax1.set_xlabel("Lag Time (tau) [s]")
    ax1.set_ylabel("Correlation (g2)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The FFT (Frequency Domain)
    ax2.loglog(freqs[1:], psd[1:], 'b-', alpha=0.6, label='PSD')
    
    if slope != 0:
        # Plot Fit
        ax2.loglog(freqs[start_fit:end_fit], fit_line, 'r--', linewidth=2, label=f'Fit Slope $\\alpha={slope:.2f}$')
        
        # Plot Reference (-2.0)
        mid_idx = (start_fit + end_fit) // 2
        ref_y = psd[mid_idx] * (freqs / freqs[mid_idx])**(-2)
        ax2.loglog(freqs[1:], ref_y[1:], 'g:', alpha=0.6, label='Lorentzian Ref (-2.0)')
        
    ax2.set_title(f"Step 2: Power Spectral Density\n(Slope should be -2.0 for liquid)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def compare_spectral_peaks(t, gt_data, den_data, height_thresh=0.01):
    """
    Calculates spectral slope using ONLY NumPy (No SciPy).
    Fits only the harmonic peaks, ignoring the noise floor.
    """
    dt = t[1] - t[0]
    
    # --- HELPER: Pure Numpy Peak Finder ---
    def find_peaks_numpy(signal, height_thresh):
        # 1. Find indices that are larger than their neighbors (Local Maxima)
        # We shift the array left and right to compare
        is_peak = (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
        
        # 2. Get the actual indices (add 1 because we sliced off the start)
        peak_indices = np.where(is_peak)[0] + 1
        
        # 3. Filter by Height
        if height_thresh is not None:
            peak_indices = peak_indices[signal[peak_indices] > height_thresh]
            
        return peak_indices

    # --- HELPER: Pure Numpy Linear Regression with Error ---
    def get_peak_slope_numpy(freqs, mags, peak_indices):
        x = np.log10(freqs[peak_indices])
        y = np.log10(mags[peak_indices])
        
        # Polyfit degree 1 = Line. cov=True returns the covariance matrix
        (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
        
        # Standard error is the square root of the diagonal of covariance
        slope_err = np.sqrt(cov[0, 0])
        
        return slope, slope_err, intercept

    # 1. Compute FFT (Using NumPy's FFT)
    def get_spectrum(signal):
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        mag = np.abs(np.fft.rfft(signal))
        # Remove DC component (index 0)
        return freqs[1:], mag[1:]

    f_gt, mag_gt = get_spectrum(gt_data)
    f_den, mag_den = get_spectrum(den_data)

    print(np.mean(mag_gt))
    print(np.mean(mag_den))
    print(0.05*np.std(mag_gt))
    print(0.05*np.std(mag_den))

    # 2. Find Peaks (Using Helper)
    # Threshold: 1% of max amplitude
    peaks_gt_idx = find_peaks_numpy(mag_gt, height_thresh=np.mean(mag_gt) - 0.05*np.std(mag_gt))
    peaks_den_idx = find_peaks_numpy(mag_den, height_thresh=np.mean(mag_den) - 0.05*np.std(mag_den))
    
    # 3. Fit Line to PEAKS ONLY
    slope_gt, err_gt, int_gt = get_peak_slope_numpy(f_gt, mag_gt, peaks_gt_idx)
    slope_den, err_den, int_den = get_peak_slope_numpy(f_den, mag_den, peaks_den_idx)

    # 4. Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Full Spectra (Faint)
    ax.semilogy(f_gt, mag_gt, color='purple', alpha=0.3, label='GT Raw Spectrum')
    ax.semilogy(f_den, mag_den, color='yellow', alpha=0.3, label='Denoised Raw Spectrum')
    
    # Plot The Peaks Used for Fitting (Dots)
    ax.semilogy(f_gt[peaks_gt_idx], mag_gt[peaks_gt_idx], 'o', color='purple', markersize=4)
    ax.semilogy(f_den[peaks_den_idx], mag_den[peaks_den_idx], 'x', color='orange', markersize=4)
    
    # Plot The Fits (Dashed Lines)
    # Line equation: 10^(intercept) * f^slope
    line_gt = (10**int_gt) * (f_gt**slope_gt)
    line_den = (10**int_den) * (f_den**slope_den)
    
    ax.semilogy(f_gt, line_gt, color='purple', linestyle='--', linewidth=2, 
              label=f'GT Peak Fit ($\\alpha = {slope_gt:.2f} \\pm {err_gt:.2f}$)')
    ax.semilogy(f_den, line_den, color='orange', linestyle='--', linewidth=2, 
              label=f'Den Peak Fit ($\\alpha = {slope_den:.2f} \\pm {err_den:.2f}$)')
    
    ax.set_title("Peak-Only Spectral Analysis (Pure NumPy)\n(Ignoring the Noise Floor)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def g2_goodness_of_fit(y_data, y_fit, *popt):
    """
    Calculates goodness-of-fit metrics for curve_fit results.
    
    Args:
        y_data: The input data arrays.
        y_fit: The fit data.
        popt: The optimized parameters from curve_fit.
    """
    
    # 2. Calculate Residuals
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)  # Sum of Squares Residuals
    ss_tot = np.sum((y_data - np.mean(y_data))**2) # Total Sum of Squares
    
    # 3. Calculate Metrics
    # R-Squared
    r_squared = 1 - (ss_res / ss_tot)
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Reduced Chi-Squared (Requires degrees of freedom)
    # DOF = Number of data points - Number of fitted parameters
    dof = len(y_data) - len(popt) 
    if dof > 0:
        chi_squared_reduced = ss_res / dof
    else:
        chi_squared_reduced = np.nan # Avoid division by zero
    
    # 4. Print Report
    print("-" * 35)
    print("   GOODNESS OF FIT REPORT")
    print("-" * 35)
    print(f"R-Squared (R²):       {r_squared:.4f}  (Target: -> 1.0)")
    print(f"RMSE:                 {rmse:.4e}  (Target: -> 0.0)")
    print(f"Reduced Chi-Squared:  {chi_squared_reduced:.4f}  (Target: ~1.0)")
    print("-" * 35)
    
    return r_squared, rmse, chi_squared_reduced

def compare_before_after_denoising(t, gt_data, noisy_data, noisy_fit, den_data, den_fit, qvalue):
    """
    Comprehensive comparison: Before (Noisy) vs. After (Denoised).
    Displays SNR, RMSE, and R2 metrics on the plot.
    
    Args:
        t: Time axis
        gt_data: The Ground Truth signal (Reference)
        noisy_data: The raw input measurements
        noisy_fit: The curve_fit result on raw data
        den_data: The denoised network output
        den_fit: The curve_fit result on denoised data
        qvalue: The q-value for the current dataset
    """
    
    # --- 1. CALCULATE METRICS ---
    # A. Residuals (Data - Fit) -> Measures goodness of fit
    res_noisy = noisy_data - noisy_fit
    res_den = den_data - den_fit
    
    # B. Errors (Data - GT) -> Measures physical accuracy
    error_noisy = noisy_data - gt_data
    error_den = den_data - gt_data

    # C. SNR Calculation (dB)
    # Signal Power = Variance of GT
    # Noise Power = Variance of Error (Difference from GT)
    sig_power = np.var(gt_data)
    noise_power_in = np.var(error_noisy)
    noise_power_out = np.var(error_den)
    
    snr_in = 10 * np.log10(sig_power / noise_power_in)
    snr_out = 10 * np.log10(sig_power / noise_power_out)
    snr_gain = snr_out - snr_in
    
    # D. RMSE (Root Mean Square Error)
    rmse_in = np.sqrt(mean_squared_error(gt_data, noisy_data))
    rmse_out = np.sqrt(mean_squared_error(gt_data, den_data))
    rmse_reduction = (rmse_in - rmse_out) / rmse_in * 100
    
    # E. R-Squared (Goodness of Fit)
    r2_noisy = r2_score(noisy_data, noisy_fit)
    r2_den = r2_score(den_data, den_fit)
    
    # --- 2. PRINT REPORT ---
    print("="*60)
    print(f"{'METRIC':<20} | {'BEFORE (Noisy)':<18} | {'AFTER (Denoised)':<18} | {'IMPROVEMENT':<12}")
    print("-" * 75)
    print(f"{'SNR (dB)':<20} | {snr_in:.2f} dB           | {snr_out:.2f} dB           | +{snr_gain:.2f} dB")
    print(f"{'RMSE':<20} | {rmse_in:.4f}             | {rmse_out:.4f}             | -{rmse_reduction:.1f}%")
    print(f"{'R-Squared':<20} | {r2_noisy:.4f}             | {r2_den:.4f}             |")
    print("="*60)

    # --- 3. VISUALIZATION ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"q = {qvalue}", fontsize=16, weight='bold')
    # --- LEFT COLUMN: BEFORE DENOISING ---
    # Plot Data & Fits
    axes[0,0].scatter(t, noisy_data, c='gray', alpha=0.5, s=15, label='Noisy Input')
    axes[0,0].plot(t, noisy_fit, 'k-', linewidth=2, label='Fit (Noisy)')
    axes[0,0].plot(t, gt_data, 'g--', linewidth=2, alpha=0.7, label='Ground Truth')
    axes[0,0].set_xscale('log')
    
    # Add Metrics Box (Top Left)
    text_str_1 = '\n'.join((
        r'$\bf{BEFORE}$',
        f'SNR: {snr_in:.1f} dB',
        f'RMSE: {rmse_in:.3f}',
        f'R²: {r2_noisy:.3f}'))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    axes[0,0].text(0.05, 0.95, text_str_1, transform=axes[0,0].transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)
    
    axes[0,0].set_title('BEFORE: Noisy Input Data')
    axes[0,0].legend(loc='upper right')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot Residuals (Noisy)
    axes[1,0].scatter(t, res_noisy, c='gray', s=10, alpha=0.6)
    axes[1,0].axhline(0, c='k', linestyle='--')
    axes[1,0].set_ylabel('Residuals')
    # Set Y-limit based on noisy data to show scale
    limit = 3 * np.std(res_noisy)
    axes[1,0].set_ylim(-limit, limit)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_title(f'Residual Noise ($\sigma$={np.std(res_noisy):.2f})')

    # --- RIGHT COLUMN: AFTER DENOISING ---
    # Plot Data & Fits
    axes[0,1].scatter(t, den_data, c='red', alpha=0.3, s=15, label='Denoised Output')
    axes[0,1].plot(t, den_fit, 'b-', linewidth=2, label='Fit (Denoised)')
    axes[0,1].plot(t, gt_data, 'g--', linewidth=2, alpha=0.7, label='Ground Truth')
    axes[0,1].set_xscale('log')
    
    # Add Metrics Box (Top Right)
    text_str_2 = '\n'.join((
        r'$\bf{AFTER}$',
        f'SNR: {snr_out:.1f} dB (↑{snr_gain:.1f})',
        f'RMSE: {rmse_out:.3f} (↓{rmse_reduction:.0f}%)',
        f'R²: {r2_den:.3f}'))
    props_success = dict(boxstyle='round', facecolor='#d4edda', alpha=0.8) # Light green box
    axes[0,1].text(0.05, 0.95, text_str_2, transform=axes[0,1].transAxes, fontsize=11,
                   verticalalignment='top', bbox=props_success)
    
    axes[0,1].set_title('AFTER: Denoised Output')
    axes[0,1].legend(loc='upper right')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot Residuals (Denoised) - CRITICAL: KEEP SAME SCALE AS LEFT
    axes[1,1].scatter(t, res_den, c='red', s=10, alpha=0.6)
    axes[1,1].axhline(0, c='b', linestyle='--')
    axes[1,1].set_ylim(-limit, limit) # Force same scale to show reduction
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_title(f'Residual Noise ($\sigma$={np.std(res_den):.2f})')
    
    plt.tight_layout()
    plt.show()

def compare_fitting_results(t, gt_data, gt_fit, den_data, den_fit):
    """
    Visualizes and metrics comparison between Ground Truth and Denoised fitting.
    
    Args:
        t: Time axis
        gt_data: The raw noisy measurements
        gt_fit: The curve_fit result on raw data
        den_data: The denoised measurements (network output)
        den_fit: The curve_fit result on denoised data
    """
    
    # --- 1. CALCULATE METRICS ---
    # Residuals
    res_gt = gt_data - gt_fit
    res_den = den_data - den_fit
    
    # R-Squared
    r2_gt = r2_score(gt_data, gt_fit)
    r2_den = r2_score(den_data, den_fit)
    
    # Standard Error of the Regression (Noise Level estimate)
    ser_gt = np.std(res_gt)
    ser_den = np.std(res_den)
    
    # Improvement Factor (How much noise did we remove?)
    improvement = (ser_gt - ser_den) / ser_gt * 100
    
    # --- 2. GENERATE REPORT ---
    print("="*40)
    print("      FITTING COMPARISON REPORT")
    print("="*40)
    print(f"{'METRIC':<20} | {'GROUND TRUTH':<12} | {'DENOISED':<12}")
    print("-" * 50)
    print(f"{'R-Squared':<20} | {r2_gt:.4f}       | {r2_den:.4f}")
    print(f"{'Residual Std (Noise)':<20} | {ser_gt:.4f}       | {ser_den:.4f}")
    print("-" * 50)
    print(f"Noise Reduction: {improvement:.2f}%")
    
    # --- 3. VISUALIZATION ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # Top Left: GT Fit
    axes[0,0].scatter(t, gt_data, c='gray', alpha=0.5, s=10, label='Raw Data')
    axes[0,0].plot(t, gt_fit, 'k--', linewidth=2, label='GT Fit')
    axes[0,0].set_title(f'Ground Truth (R²={r2_gt:.3f})')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Top Right: Denoised Fit
    axes[0,1].scatter(t, den_data, c='red', alpha=0.3, s=10, label='Denoised Data')
    axes[0,1].plot(t, den_fit, 'b-', linewidth=2, label='Denoised Fit')
    axes[0,1].set_title(f'Denoised Result (R²={r2_den:.3f})')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Bottom Left: GT Residuals
    axes[1,0].scatter(t, res_gt, c='gray', s=10)
    axes[1,0].axhline(0, c='k', linestyle='--')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_ylim(-3*ser_gt, 3*ser_gt) # Scale to GT noise
    axes[1,0].set_title(f'Noise Level: $\sigma = {ser_gt:.2f}$')
    
    # Bottom Right: Denoised Residuals (The "Proof")
    axes[1,1].scatter(t, res_den, c='red', s=10)
    axes[1,1].axhline(0, c='b', linestyle='--')
    axes[1,1].set_ylim(-3*ser_gt, 3*ser_gt) # KEEP SAME SCALE AS GT!
    axes[1,1].set_title(f'Noise Level: $\sigma = {ser_den:.2f}$')
    
    plt.tight_layout()
    plt.show()

def visualize_snr_enhancement(t, y_gt, y_noisy, y_noisy_fit, y_den, y_den_fit):
    """
    Visualizes SNR enhancement and fits with comprehensive metrics on the plot.
    
    Args:
        t: Time axis
        y_gt: Ground Truth (The perfect signal)
        y_noisy: Raw input data
        y_noisy_fit: The curve_fit result on raw data
        y_den: Denoised network output
        y_den_fit: The curve_fit result on denoised data
    """
    
    # --- 1. CALCULATE METRICS ---
    # A. Signal Power (Variance of the Ground Truth)
    sig_power = np.var(y_gt)
    
    # B. Noise Power (Variance of the Residuals)
    # Note: We use Data - GT to get the true "Error Noise"
    noise_power_in = np.var(y_noisy - y_gt)
    noise_power_out = np.var(y_den - y_gt)
    
    # C. SNR (Decibels)
    # Formula: 10 * log10( Signal / Noise )
    snr_in = 10 * np.log10(sig_power / noise_power_in)
    snr_out = 10 * np.log10(sig_power / noise_power_out)
    snr_gain = snr_out - snr_in
    
    # D. RMSE (Accuracy relative to GT)
    rmse_in = np.sqrt(mean_squared_error(y_gt, y_noisy))
    rmse_out = np.sqrt(mean_squared_error(y_gt, y_den))
    
    # --- 2. SETUP PLOT ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
    
    # --- PANEL 1: BEFORE DENOISING ---
    # Plot Data
    ax1.scatter(t, y_noisy, color='gray', alpha=0.5, s=15, label='Noisy Input')
    ax1.plot(t, y_noisy_fit, color='black', linestyle='-', linewidth=2, label='Fit (Noisy)')
    ax1.plot(t, y_gt, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Ground Truth')
    
    # Add Metrics Box
    stats_text_1 = (
        f"BEFORE DENOISING\n"
        f"----------------\n"
        f"SNR:  {snr_in:.2f} dB\n"
        f"RMSE: {rmse_in:.4f}"
    )
    ax1.text(0.95, 0.95, stats_text_1, transform=ax1.transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_title(f"Stage 1: Input Data Analysis (SNR = {snr_in:.1f} dB)", fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # --- PANEL 2: AFTER DENOISING ---
    # Plot Data
    ax2.scatter(t, y_den, color='red', alpha=0.3, s=15, label='Denoised Output')
    ax2.plot(t, y_den_fit, color='darkred', linestyle='-', linewidth=2, label='Fit (Denoised)')
    ax2.plot(t, y_gt, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Ground Truth')
    
    # Add Metrics Box (With Improvement Highlight)
    stats_text_2 = (
        f"AFTER DENOISING\n"
        f"---------------\n"
        f"SNR:  {snr_out:.2f} dB  (↑ {snr_gain:.1f} dB)\n"
        f"RMSE: {rmse_out:.4f} (↓ {(rmse_in-rmse_out)/rmse_in*100:.1f}%)"
    )
    # Determine box color based on success
    box_color = '#d4edda' if snr_gain > 0 else '#f8d7da' # Greenish if improved, Reddish if worse
    
    ax2.text(0.95, 0.95, stats_text_2, transform=ax2.transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9))
    
    ax2.set_title(f"Stage 2: Network Output (SNR Gain = +{snr_gain:.2f} dB)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time (t)", fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def interactive_preview(noisy_data, target_data, cmap="viridis"):
    """
    Interactive loop to inspect noisy vs pure TTCF before training.

    Features:
      - Noisy vs pure TTCF images
      - Noise histogram
      - Random row + column intensity profiles
      - Controls: (n)ext, (c)ontinue, (s)top
    """
    print("\n🔍 Interactive TTCF Inspection Mode")
    print("Press:")
    print("  n = next random sample")
    print("  c = continue training")
    print("  s = stop program\n")

    import random

    while True:
        idx = random.randint(0, noisy_data.shape[0] - 1)
        noisy = noisy_data[idx, :, :, 0]
        pure = target_data[idx, :, :, 0]

        # Noise array
        noise = noisy - pure

        # Pick random row + column
        H, W = noisy.shape
        random_row = 50
        random_col = 50

        # ---- FIGURE: TTCF + Histogram + Profiles ----
        fig = plt.figure(figsize=(16, 10))

        # Noisy TTCF
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_title(f"Noisy TTCF (idx={idx})")
        ax1.imshow(noisy, cmap=cmap)
        ax1.axis("off")

        # Pure TTCF
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title("Pure Target TTCF")
        ax2.imshow(pure, cmap=cmap)
        ax2.axis("off")

        # Noise Histogram
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_title("Noise Distribution")
        ax3.hist(noise.flatten(), bins=50, color="steelblue", alpha=0.8)
        ax3.set_xlabel("Noise Value")
        ax3.set_ylabel("Count")
        ax3.grid(alpha=0.3)

        # Random Row Profile
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title(f"Row Intensity Profile (row={random_row})")
        ax4.plot(noisy[random_row, :], label="Noisy", alpha=0.7)
        ax4.plot(pure[random_row, :], label="Pure", alpha=0.7)
        ax4.legend()
        ax4.grid(alpha=0.3)

        # Random Column Profile
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title(f"Column Intensity Profile (col={random_col})")
        ax5.plot(noisy[:, random_col], label="Noisy", alpha=0.7)
        ax5.plot(pure[:, random_col], label="Pure", alpha=0.7)
        ax5.legend()
        ax5.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # ---- USER INPUT ----
        choice = input("Enter n (next), c (continue), s (stop): ").strip().lower()

        if choice == "n":
            continue
        elif choice == "c":
            print("➡️ Continuing to training...\n")
            break
        elif choice == "s":
            print("🛑 Stopping the program.")
            sys.exit(0)
        else:
            print("⚠️ Invalid input. Try again.")
            

def analyze_g2_list_fft_dynamic(g2_list, dt=1.0, title="Averaged FFT (Dynamic Baseline)"):
    """
    Computes the FFT of a list of g2 curves where EACH curve has its own unique baseline.
    
    1. Calculates baseline from the tail of each curve.
    2. Subtracts it to center the signal at 0.
    3. Computes FFT and averages the Power Spectra.
    """
    g2_array = np.array(g2_list)
    num_curves, num_lags = g2_array.shape
    
    psd_accumulator = []
    freqs = np.fft.rfftfreq(num_lags, d=dt)  
    # --- LOOP: Process Each Curve Individually ---
    for i in range(num_curves):
        curve = g2_array[i, :]
        
        # 1. Handle NaNs
        # If the curve has NaNs (from rectification), we valid mask them
        # or just take the valid prefix
        if np.isnan(curve).any():
             # Option: Fill NaNs with the last valid value (simple fix)
             # or skip if too many NaNs
             mask = ~np.isnan(curve)
             if np.sum(mask) < 10: continue
             # Simple interpolation for FFT safety
             curve = np.interp(np.arange(len(curve)), np.arange(len(curve))[mask], curve[mask])

        # 2. CALCULATE INDIVIDUAL BASELINE
        # We assume the last 10% of the curve represents the fully decayed state.
        tail_len = max(5, int(num_lags * 0.1)) 
        individual_baseline = np.mean(curve[-tail_len:])
        
        # 3. SUBTRACT BASELINE
        # This centers the decay around 0.0, removing the DC spike.
        signal = curve - individual_baseline
        
        # 4. WINDOW & FFT
        # We multiply by window to smooth the start/end discontinuity
        fft_result = np.fft.rfft(signal)
        
        # 5. COMPUTE POWER (Magnitude Squared)
        psd = np.abs(fft_result)**2
        psd_accumulator.append(psd)
        #plotting 
        
        plt.loglog(freqs[1:],psd[1:], linewidth=2, label='Averaged Power Spectrum')
        plt.title(f"{title}\n(Dynamic Baseline Subtraction)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
    plt.show()
   
    return 0


def compute_laplace_transform_g2(g2_list, t_lags, s_values=None, labels=[]):
    """
    Approximates the Distribution of Relaxation Rates G(Gamma) using the 
    First-Order Derivative method.
    
    This converts the g2 decay curve into a PEAK at the characteristic rate.
    
    Peak Position = Characteristic Decay Rate (Gamma)
    """
    g2_array = np.array(g2_list)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.jet(np.linspace(0, 1, len(g2_list)))
    smoothing = 0.01
    for i, g2_curve in enumerate(g2_array):
        # 1. Match Lengths & Clean NaNs
        current_len = len(g2_curve)
        if len(t_lags) >= current_len:
            t = t_lags[:current_len]
        else:
            g2_curve = g2_curve[:len(t_lags)]
            t = t_lags
            
        mask = ~np.isnan(g2_curve) & (t > 0)
        t_valid = t[mask]
        g_valid = g2_curve[mask]
        
        if len(g_valid) < 5: continue

        # 2. FIT SPLINE
        # We work in Log-Time because particle decay is exponential
        x_log = np.log(t_valid)
        
        try:
            # k=3 (Cubic spline), s=smoothing factor
            # We assume weights are uniform (w=None)
            spline = UnivariateSpline(x_log, g_valid, k=3, s=smoothing)
            
            # 3. COMPUTE DERIVATIVE
            # The spline object allows us to calculate the derivative directly
            # G(tau) ~ -dg2 / d(ln t)
            g2_deriv = spline.derivative()(x_log)
            
            # Invert for distribution (Decay is negative slope, so we negate)
            distribution = -g2_deriv
            
            # 4. PLOT
            # Rate Gamma = 1/t
            gamma_axis = 1.0 / t_valid
            
            # Plot the Peak
            if labels:
                plt.semilogx(gamma_axis, distribution, color=colors[i], alpha=0.6, linewidth=1.5, label=labels[i])
            else:
                plt.semilogx(gamma_axis, distribution, color=colors[i], alpha=0.6, linewidth=1.5)
            
        except Exception as e:
            print(f"Spline fit failed for curve {i}: {e}")
            continue

    plt.title("Relaxation Rate Distribution (Spline Method)")
    plt.xlabel("Relaxation Rate $\Gamma$ [Hz] (1/s)")
    plt.ylabel("Amplitude $G(\Gamma)$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()




def plot_g2_laplace_s(
    g2_list,
    t_lags,
    s_values=None,
    subtract_baseline=False,
    labels=None,
    log_s=True
):
    """
    Plot Laplace transform of g2(tau) in s-domain.

    Computes:
        L(s) = ∫ f(tau) exp(-s tau) d tau
    where f(tau) = g2(tau) - 1 if subtract_baseline else g2(tau)

    Args:
        g2_list: list of 1D arrays (each g2 vs tau)
        t_lags:  1D array of tau values (seconds)
        s_values: 1D array of s (1/seconds). If None, auto-select.
        subtract_baseline: if True, use g2-1 (recommended)
        labels: list of labels
        log_s: whether to use log scale on s-axis
    Returns:
        s_values, L_all (array shape [n_curves, len(s_values)])
    """
    t = np.asarray(t_lags, float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_lags must be strictly increasing")

    # Auto s-grid (covers approx 1/t_max .. 1/t_min)
    if s_values is None:
        tmin = np.min(t[t > 0])
        tmax = np.max(t)
        s_min = 1.0 / (tmax * 5.0)
        s_max = 1.0 / (tmin / 5.0)
        s_values = np.logspace(np.log10(s_min), np.log10(s_max), 300)

    s_values = np.asarray(s_values, float)

    L_all = []

    for i, g2 in enumerate(g2_list):
        g2 = np.asarray(g2, float)

        # Match lengths
        n = min(len(g2), len(t))
        tt = t[:n]
        yy = g2[:n]

        # Clean & keep tau>0
        m = np.isfinite(yy) & np.isfinite(tt) & (tt > 0)
        tt = tt[m]
        yy = yy[m]

        if subtract_baseline:
            yy = yy - 1.0

        # Numerical Laplace for each s (vectorized)
        # L(s) ≈ ∫ yy(t) exp(-s t) dt
        # -> use trapezoidal integration
        E = np.exp(-np.outer(s_values, tt))          # shape [S, T]
        Ls = np.trapz(yy * E, tt, axis=1)            # shape [S]
        L_all.append(Ls)

    L_all = np.vstack(L_all)

    # -------- Plot ----------
    plt.figure(figsize=(9, 6))
    for i in range(L_all.shape[0]):
        lab = labels[i] if labels is not None else None
        A0 = np.max(L_all[i])
        
        plt.plot(s_values, L_all[i], lw=2, alpha=0.85, label=lab)

    if log_s:
        plt.xscale("log")

    plt.xlabel(r"Laplace variable $s$ (s$^{-1}$)")
    plt.ylabel(r"$\mathcal{L}\{g_2-1\}(s)$" if subtract_baseline else r"$\mathcal{L}\{g_2\}(s)$")
    plt.title("g₂ in Laplace (s) domain")
    plt.grid(True, which="both", alpha=0.3)
    if labels is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()



def residual_whiteness_g2_single(
    
    g2_obs,
    g2_fit,
    t_lags,
    sigma=None,
    fit_range=None,
    max_lag=50,
    ljung_box_lags=100,
    alpha=0.05,
    title="Residual Whiteness Check",
    dpi=200,
):
    """
    Residual whiteness test for a single g2 curve.
    
    Parameters
    ----------
    g2_obs : array
        Measured g2(τ)
    g2_fit : array
        Fitted model curve
    t_lags : array
        Delay times τ
    sigma : array or None
        Optional uncertainty per point
    fit_range : (t_min, t_max) or None
    """

    t = np.asarray(t_lags, float)
    y = np.asarray(g2_obs, float)
    yfit = np.asarray(g2_fit, float)

    # Mask valid values
    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(yfit) & (t > 0)
    t, y, yfit = t[m], y[m], yfit[m]

    if fit_range is not None:
        t_min, t_max = fit_range
        m2 = (t >= t_min) & (t <= t_max)
        t, y, yfit = t[m2], y[m2], yfit[m2]

    if len(t) < 10:
        raise ValueError("Too few points for whiteness test.")

    # Residuals
    resid = y - yfit

    # Standardization
    if sigma is not None:
        sigma = np.asarray(sigma, float)[m]
        if fit_range is not None:
            sigma = sigma[m2]
        r = resid / sigma
    else:
        # Estimate scale from tail
        tail_len = max(5, len(resid)//10)
        scale = np.std(resid[-tail_len:], ddof=1)
        scale = scale if scale > 0 else np.std(resid)
        r = resid / scale

    r = r[np.isfinite(r)]
    n = len(r)

    # Mean & std
    r_mean = np.mean(r)
    r_std = np.std(r, ddof=1)

    # Autocorrelation
    L = min(max_lag, n-1)
    r0 = r - np.mean(r)
    denom = np.dot(r0, r0) + 1e-30
    acf = np.array([
        np.dot(r0[:-k], r0[k:]) / denom for k in range(1, L+1)
    ])

    # Ljung-Box test
    m_lb = min(ljung_box_lags, L)
    rho = acf[:m_lb]
    Q = n*(n+2)*np.sum((rho**2)/(n - np.arange(1, m_lb+1)))
    pval = 1 - chi2.cdf(Q, df=m_lb)

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=dpi)
    fig.suptitle(title)

    # Residual trace
    ax[0].plot(t, resid)
    ax[0].axhline(0, ls='--')
    ax[0].set_xscale("log")
    ax[0].set_title("Standardized Residuals")
    ax[0].set_xlabel("τ")
    ax[0].set_ylabel("r(τ)")

    # ACF
    bound = 1.96 / np.sqrt(n)
    ax[1].stem(range(1, L+1), acf, basefmt=" ")
    ax[1].axhline(bound, ls='--')
    ax[1].axhline(-bound, ls='--')
    ax[1].axhline(0)
    ax[1].set_title("Residual ACF")
    ax[1].set_xlabel("Lag")

    # Histogram
    ax[2].hist(r, bins=30, density=True)
    ax[2].set_title("Residual Distribution")
    ax[2].set_xlabel("r")

    plt.tight_layout()
    plt.show()

    print("Residual mean:", r_mean)
    print("Residual std:", r_std)
    print("Ljung-Box Q:", Q)
    print("p-value:", pval)

    return {
        "residual_mean": r_mean,
        "residual_std": r_std,
        "Q": Q,
        "p_value": pval,
        "white": pval > alpha
    }








    #=================================================================


def _center_crop_square(x: np.ndarray, N: int) -> np.ndarray:
    r, c = x.shape
    r0 = (r - N) // 2
    c0 = (c - N) // 2
    return x[r0:r0+N, c0:c0+N]

def _hann2d(N: int) -> np.ndarray:
    w = np.hanning(N)
    return np.outer(w, w)

def _snr_bootstrap_ratio(ridge_vals, bg_vals, n_boot=2000, seed=0):
    """Bootstrap CI for ratio mean(ridge)/mean(bg)."""
    ridge_vals = np.asarray(ridge_vals, float)
    bg_vals = np.asarray(bg_vals, float)
    ridge_vals = ridge_vals[np.isfinite(ridge_vals)]
    bg_vals = bg_vals[np.isfinite(bg_vals)]
    if ridge_vals.size < 20 or bg_vals.size < 50:
        return np.nan, (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    R = ridge_vals
    B = bg_vals
    nR, nB = R.size, B.size

    ratios = np.empty(n_boot, float)
    for i in range(n_boot):
        r_s = R[rng.integers(0, nR, size=nR)]
        b_s = B[rng.integers(0, nB, size=nB)]
        ratios[i] = (np.mean(r_s) / np.mean(b_s))

    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return np.median(ratios), (lo, hi)

def analyze_ridge_snr_many(
    ttc_list,
    dt,
    main_title="Ridge SNR",
    labels=None,
    f_min=None,
    f_max=None,
    ridge_half_width=2,       # ridge band thickness (pixels)
    bg_exclusion=6,           # exclude pixels within this distance from ridge
    eps=1e-20,
    n_boot=2000,
    seed=0,
    cmap="inferno",
    dpi=200,
    max_maps=6,
    plot=True,
):
    """
    Computes ridge SNR from 2D FFT power:
      SNR = mean(P in ridge band) / mean(P in background band)
    Optionally band-limit by radial frequency magnitude within [f_min, f_max].

    Returns list of dicts, one per TTC, including SNR and bootstrap CI.
    """

    ttc_list = list(ttc_list)
    M = len(ttc_list)
    if M == 0:
        raise ValueError("ttc_list is empty.")

    if labels is None:
        labels = [f"TTC {i+1}" for i in range(M)]
    if len(labels) != M:
        raise ValueError("labels must match ttc_list length.")

    # sanitize & find common crop size
    xs, shapes = [], []
    for x in ttc_list:
        a = np.asarray(x, float)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        xs.append(a)
        shapes.append(a.shape)

    N = min(min(r, c) for r, c in shapes)
    if N < 8:
        raise ValueError(f"Common N={N} too small.")

    # frequency grids (after fftshift, index center corresponds to f=0)
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    Fx, Fy = np.meshgrid(f, f, indexing="xy")
    Fr = np.sqrt(Fx**2 + Fy**2)  # radial frequency magnitude

    # choose default band once
    # (recommended: avoid DC/window region and avoid Nyquist edge)
    if f_min is None:
        # conservative default: > 5*df
        df = 1.0 / (N * dt)
        f_min = 5.0 * df
    if f_max is None:
        # conservative default: 0.2 * Nyquist
        f_nyq = 1.0 / (2.0 * dt)
        f_max = 0.2 * f_nyq

    # masks for ridge geometry
    I, J = np.indices((N, N))
    dist_to_antidiag = np.abs((I + J) - (N - 1))  # 0 on anti-diagonal

    ridge_mask_geom = dist_to_antidiag <= ridge_half_width
    bg_mask_geom = dist_to_antidiag >= bg_exclusion  # far enough away

    # band-limit mask (radial)
    band_mask = (Fr >= f_min) & (Fr <= f_max)

    # combine
    ridge_mask = ridge_mask_geom & band_mask
    bg_mask = bg_mask_geom & band_mask

    win = _hann2d(N)

    results = []
    Ps = []

    for i, (x, lab) in enumerate(zip(xs, labels)):
        x = _center_crop_square(x, N)
        x = x - np.mean(x)
        xw = x * win

        F2 = np.fft.fftshift(np.fft.fft2(xw))
        P = (np.abs(F2)**2)
        Ps.append(P)

        ridge_vals = P[ridge_mask]
        bg_vals = P[bg_mask]

        mu_r = np.mean(ridge_vals) if ridge_vals.size else np.nan
        mu_b = np.mean(bg_vals) if bg_vals.size else np.nan

        snr = (mu_r / (mu_b + eps)) if np.isfinite(mu_r) and np.isfinite(mu_b) else np.nan
        snr_db = 10.0 * np.log10(snr + eps) if np.isfinite(snr) else np.nan

        snr_med, (lo, hi) = _snr_bootstrap_ratio(
            ridge_vals, bg_vals, n_boot=n_boot, seed=seed + i
        )
        snr_db_ci = (10*np.log10(lo+eps), 10*np.log10(hi+eps)) if np.isfinite(snr_med) else (np.nan, np.nan)
        snr_db_med = 10*np.log10(snr_med+eps) if np.isfinite(snr_med) else np.nan

        results.append({
            "label": lab,
            "N": N,
            "dt": dt,
            "f_min": float(f_min),
            "f_max": float(f_max),
            "ridge_half_width": ridge_half_width,
            "bg_exclusion": bg_exclusion,
            "snr": snr,
            "snr_db": snr_db,
            "snr_boot_median": snr_med,
            "snr_boot_ci": (lo, hi),
            "snr_db_boot_median": snr_db_med,
            "snr_db_boot_ci": snr_db_ci
        })
    if not plot:
        return results

    # ---------------- Plot: publication-style summary ----------------
    show_maps = min(M, max_maps)
    if show_maps > 0:
        logPs = [np.log10(Ps[k] + eps) for k in range(show_maps)]
        all_vals = np.concatenate([lp.ravel() for lp in logPs])
        vmin, vmax = np.percentile(all_vals, [2, 99.5])

        map_cols = min(3, show_maps)
        map_rows = (show_maps + map_cols - 1) // map_cols
        
        fig = plt.figure(figsize=(3.5 * map_cols + 1, 3.5 * map_rows))
        gs = fig.add_gridspec(map_rows, map_cols + 1, width_ratios=[1]*map_cols + [0.05], wspace=0.2)
        
        last_im = None
        for k in range(show_maps):
            ax = fig.add_subplot(gs[k // map_cols, k % map_cols])
            last_im = ax.imshow(np.log10(Ps[k] + eps), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.plot([0, N-1], [N-1, 0], ls="--", color="white", lw=1.2, alpha=0.6)
            ax.set_title(labels[k], fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])

        if last_im:
            cax = fig.add_subplot(gs[:, -1])
            fig.colorbar(last_im, cax=cax, label="log10 Power")
        
        plt.tight_layout()
        plt.show()
    # right = outer[:, 1].subgridspec(1,1, height_ratios=[1.0])
    # ax_bar = fig.add_subplot(right[0, 0])

    # bar plot in dB with bootstrap CI
    x = np.arange(len(labels))
    snr_db = np.array([r["snr_db_boot_median"] for r in results], float)
    ci_db = np.array([r["snr_db_boot_ci"] for r in results], float)
    yerr = np.vstack([snr_db - ci_db[:, 0], ci_db[:, 1] - snr_db])

    plt.figure(figsize=(4, 4))
    plt.errorbar(x, snr_db, yerr=yerr, fmt="o", capsize=4, lw=1.8)
    plt.xticks(x, labels, rotation=35, ha="right", fontsize=9)
    plt.ylabel("Ridge SNR (dB)", fontsize=11)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    return results



############# g2 -based functions #############



def inverse_laplace_nnls_g2(
    g2_array,
    t_lags_array,
    gamma_min=1e4,
    gamma_max=3e4,
    n_gamma=200,
    subtract_baseline=True,
    q_values=None,
    temp=293.15,       # New: Temperature in Kelvin (20 C)
    viscosity=0.001,   # New: Viscosity in Pa.s (Water)
    main_title="Inverse Laplace of g2",
    datasets_labels=None,
    plot=True,
    plot_table=False,
):
    """
    Inverse Laplace transform of g2 using NNLS with Hydrodynamic Size Analysis.

    Returns:
        results_dict: {
            "summary_df": DataFrame with Mean Gamma, PDI, Mean Rh, etc.
            "distributions": List of dicts [{'gamma':..., 'G':..., 'rh':..., 'P_rh':...}]
        }
    """
    results_data = []
    distribution_storage = []
    
    # Physics Constants
    kB = 1.380649e-23
    # Check if q is provided for Rh calculation
    calc_physics = q_values is not None
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        title_text = f'{main_title}'
        if calc_physics:
            title_text += f' (q={q_values:.2e} m^-1)'
        fig.suptitle(title_text, fontsize=16)
    
    # --- MAIN LOOP ---
    for i in range(len(g2_array)):
        # 1. Prepare Data
        t_raw = np.asarray(t_lags_array[i], float)
        g2_raw = np.asarray(g2_array[i], float)
        
        # Robust masking
        mask = np.isfinite(t_raw) & np.isfinite(g2_raw) & (t_raw > 0)
        t = t_raw[mask]
        g2 = g2_raw[mask]

        if len(t) == 0: continue

        # Baseline handling
        if subtract_baseline:
            tail_len = max(5, len(g2)//10)
            baseline = np.mean(g2[-tail_len:])
            g = g2 - baseline
        else:
            g = g2.copy()

        # Keep positive part only
        m = g > 0
        if np.sum(m) < 5: continue 
        t = t[m]
        g = g[m]

        # 2. Setup Gamma Grid
        if gamma_min is None:
            g_min_curr = 1.0 / (10 * t.max())
        else:
            g_min_curr = gamma_min
        if gamma_max is None:
            g_max_curr = 10.0 / t.min()
        else:
            g_max_curr = gamma_max

        Gamma = np.logspace(np.log10(g_min_curr), np.log10(g_max_curr), n_gamma)
        
        # 3. Solve NNLS
        K = np.exp(-2.0 * np.outer(t, Gamma))
        G, rnorm = nnls(K, g)
        threshold = 0.05 * np.max(G)
        G[G < threshold] = 0

        # Normalize G(Gamma)
        area = np.trapz(G, Gamma)
        if area > 0: G /= area
        else: continue

        # 4. MOMENT CALCULATION (Gamma Space)
        #G = G
        #Gamma = Gamma
        mask = (Gamma >= 1e2) & (Gamma <= 1e9)
        G = G[mask]
        Gamma = Gamma[mask]
        
        mean_gamma = np.trapz(Gamma * G, Gamma)
        variance = np.trapz((Gamma - mean_gamma)**2 * G, Gamma)
        pdi = variance / (mean_gamma**2) if mean_gamma > 0 else 0
        pdi_error = pdi * (rnorm / np.linalg.norm(g))
        # Peak (Mode)
        max_idx = np.argmax(G)
        peak_gamma_mode = Gamma[max_idx]
        amp_max = G[max_idx]

        label = datasets_labels[i] if datasets_labels else f"Data {i}"
        
        # 5. HYDRODYNAMIC CALCULATIONS (Rh Space)
        mean_rh = np.nan
        peak_rh_mode = np.nan
        amp_max = np.nan
        rh_axis = np.array([])
        P_rh = np.array([])
        
        if calc_physics:
            # A. Constant Term: Rh = C / Gamma
            # C = (kB * T * q^2) / (6 * pi * eta)
            # Result in nm (1e9 factor)
            C_phys = ((kB * temp) * (q_values*1e9**2)) / (6 * np.pi * viscosity) * 1e9
            
            # B. Scalar Mean Rh (from Mean Gamma)
            mean_rh = C_phys / mean_gamma
            
            # C. Transform Distribution G(Gamma) -> P(Rh)
            # Rh axis (inverse of Gamma, so we must sort it)
            rh_raw = C_phys / Gamma
            mask = (rh_raw >= 1) & (rh_raw <= 100000)
            rh_raw = rh_raw[mask]
            G = G[mask]
            Gamma = Gamma[mask]
            sort_idx = np.argsort(rh_raw)
            rh_axis = rh_raw[sort_idx]
            G_sorted = G[sort_idx]
            Gamma_sorted = Gamma[sort_idx]
            
            # Jacobian: P(R) = G(Gamma) * (Gamma / R)
            P_rh = G_sorted * (Gamma_sorted / rh_axis)
            
            # Normalize P(Rh)
            area_rh = np.trapz(P_rh, rh_axis)
            if area_rh > 0: P_rh /= area_rh
            
            P_rh = P_rh[5:]
            rh_axis = rh_axis[5:]
            max_idx = np.argmax(P_rh)
            peak_rh_mode = rh_axis[max_idx]
            amp_max = P_rh[max_idx]
            # Plot P(Rh) on Ax[2]
            if plot:
                ax[2].plot(rh_axis, P_rh, lw=2, label=label)
                ax[2].fill_between(rh_axis, P_rh, alpha=0.3)
            

        # Store Summary Data
        results_data.append({
            "Label": label,
            "Mean Gamma": mean_gamma,
            "Peak Gamma": peak_gamma_mode,
            "PDI": pdi,
            "PDI_err": pdi_error,
            "Mean Rh (nm)": mean_rh,
            "Peak Rh (nm)": peak_rh_mode,
            "Amplitude": amp_max,
        })
        
        # Store Full Distributions
        distribution_storage.append({
            "label": label,
            "gamma": Gamma,
            "G_gamma": G,
            "rh": rh_axis,
            "P_rh": P_rh
        })

        # Plot G(Gamma) on Ax[1]
        if plot:
            ax[1].semilogx(Gamma, G, lw=2, label=label)

    # --- VISUALIZATION SETUP ---
    df_results = pd.DataFrame(results_data)
    
    if not df_results.empty:
        # AX[0]: Mean Gamma & PDI Error Bars
        x_pos = np.arange(len(df_results))
        y_vals = df_results["Mean Gamma"]
        y_err = y_vals * np.sqrt(df_results["PDI"]) # Error bar = Width (Sigma)
        if plot:
            ax[0].errorbar(x_pos, y_vals, yerr=y_err, fmt='o', capsize=5, lw=2, markersize=8)
            ax[0].set_xticks(x_pos)
            ax[0].set_xticklabels(df_results["Label"], rotation=45, ha='right')
            ax[0].set_yscale('log')
            ax[0].set_ylabel(r"Mean Relaxation Rate $\Gamma$ (s$^{-1}$)")
            ax[0].set_title(r"Rate $\Gamma$ & Width ($\sigma$)")
            ax[0].grid(True, which="both", alpha=0.3)
        
        # Table on Ax[0]
        if plot_table and plot:
            cell_text = []
            for _, row in df_results.iterrows():
                rh_text = f"{row['Mean Rh (nm)']:.1f}" if calc_physics else "-"
                cell_text.append([f"{row['Mean Gamma']:.2e}", f"{row['PDI']:.3f}", rh_text])
                
            table = ax[0].table(cellText=cell_text, 
                            colLabels=["Mean $\Gamma$", "PDI", "Rh (nm)"],
                            rowLabels=df_results["Label"],
                            loc='bottom', bbox=[0.0, -0.6, 1.0, 0.3])
            plt.subplots_adjust(bottom=0.3)

    # AX[1]: G(Gamma)
    if plot:
        ax[1].set_xlabel(r"Relaxation Rate $\Gamma$ (s$^{-1}$)")
        ax[1].set_ylabel(r"$G(\Gamma)$ (Probability)")
        ax[1].set_title(r"Rate Distribution $G(\Gamma)$")
        ax[1].grid(True, which="both", alpha=0.3)
        ax[1].legend()
    
    # AX[2]: P(Rh)
    if calc_physics and plot:
        ax[2].set_xlabel(r"Hydrodynamic Radius $R_h$ (nm)")
        ax[2].set_ylabel(r"$P(R_h)$ (Probability)")
        ax[2].set_title(r"Size Distribution $P(R_h)$")
        ax[2].grid(True, which="both", alpha=0.3)
        # Optional: Log scale usually looks better for size if range is large
        # ax[2].set_xscale('log') 
    elif plot:
        ax[2].text(0.5, 0.5, "Provide 'q_values' to calculate Rh", ha='center')
    
    
    if plot:
        plt.tight_layout()
        plt.show()
    
    # Return Dictionary containing DataFrame AND Full Distributions
    return {
        "summary": df_results,
        "distributions": distribution_storage
    }



def residual_whiteness_from_fits_g2(
    g2_denoised_list,
    g2_raw_list,
    t_lags_list,
    labels=None,
    sigma_raw_list=None,
    sigma_denoised_list=None,
    fit_range=None,
    max_lag=30,
    ljung_box_lags=20,
    alpha=0.05,
    main_title="Residual whiteness check (g2)",
    plot=True,
    dpi=200,
):
    """
    Residual whiteness check using *already fitted* g2 curves.

    Residuals:
        r_i = (g2_obs(t_i) - g2_fit(t_i)) / sigma_total_i   if sigma provided
        r_i = (g2_obs(t_i) - g2_fit(t_i)) / std_tail  otherwise (approx)

    Whiteness indicators:
      - mean(r) ~ 0
      - ACF(r) near 0 (within ±1.96/sqrt(n))
      - Ljung–Box p-value > alpha
      - PSD(r) flat (white noise)

    Returns
    -------
    results : list of dict
        Contains residual mean/std, Ljung–Box Q and p-value, and whiteness decision.
    """
    g2_denoised_list = list(g2_denoised_list)
    g2_raw_list = list(g2_raw_list)
    t_lags_list = list(t_lags_list)

    if not (len(g2_denoised_list) == len(g2_raw_list) == len(t_lags_list)):
        raise ValueError("g2_denoised_list, g2_raw_list, t_lags_list must have same length.")

    M = len(g2_denoised_list)
    if labels is None:
        labels = [f"Curve {i}" for i in range(M)]
    if len(labels) != M:
        raise ValueError("labels length must match number of curves.")

    if sigma_raw_list is not None:
        sigma_raw_list = list(sigma_raw_list)
        if len(sigma_raw_list) != M:
            raise ValueError("sigma_raw_list length must match number of curves.")

    if sigma_denoised_list is not None:
        sigma_denoised_list = list(sigma_denoised_list)
        if len(sigma_denoised_list) != M:
            raise ValueError("sigma_denoised_list length must match number of curves.")

    results = []

    if plot:
        # Update into 4 columns: [Residuals, ACF, PSD, Hist]
        fig, axes = plt.subplots(M, 3, figsize=(20, 3.6 * max(1, M)), dpi=dpi)
        if M == 1:
            axes = np.array([axes])
        fig.suptitle(main_title, fontsize=14, y=0.995)

    for i in range(M):
        label = labels[i]
        t_raw = np.asarray(t_lags_list[i], float)
        y_raw = np.asarray(g2_raw_list[i], float)
        yfit_raw = np.asarray(g2_denoised_list[i], float)

        # basic sanity
        if t_raw.shape != y_raw.shape or yfit_raw.shape != y_raw.shape:
            raise ValueError(f"{label}: t, g2_obs, g2_fit must have same shape.")

        # finite mask + t>0
        m = np.isfinite(t_raw) & np.isfinite(y_raw) & np.isfinite(yfit_raw) & (t_raw > 0)
        t = t_raw[m]
        y = y_raw[m]
        yfit = yfit_raw[m]

        if fit_range is not None:
            t_min, t_max = fit_range
            m2 = (t >= t_min) & (t <= t_max)
            t, y, yfit = t[m2], y[m2], yfit[m2]

        n = t.size
        if n < 10:
            results.append({"Label": label, "status": "too_few_points"})
            continue

        # sigma
        sigma_total = None
        if sigma_raw_list[i] is not None:
            sig_raw = np.asarray(sigma_raw_list[i], float)[m]
            if fit_range is not None:
                sig_raw = sig_raw[m2]
            sig = np.where(np.isfinite(sig_raw) & (sig_raw > 0), sig_raw, np.nan)
            if np.all(~np.isfinite(sig)):
                sig = None
        else:
            sig = None

        if sigma_denoised_list[i] is not None:
            sig_denoised = np.asarray(sigma_denoised_list[i], float)[m]
            if fit_range is not None:
                sig_denoised = sig_denoised[m2]
            sig_denoised = np.where(np.isfinite(sig_denoised) & (sig_denoised > 0), sig_denoised, np.nan)
            if np.all(~np.isfinite(sig_denoised)):
                sig_denoised = None
            
        else:
            sig_denoised = None
        
        if sig is not None and sig_denoised is not None:
            sigma_total = np.sqrt(sig**2 + sig_denoised**2)
            print("sigma_total")
        elif sig is not None:
            sigma_total = sig
            print("sigma_raw")
        elif sig_denoised is not None:
            sigma_total = sig_denoised
            print("sigma_denoised")
        # residuals
        resid = (y - yfit)

        # if no sigma, estimate a scale from tail of residuals (approx)
        if sigma_total is None:
            tail_len = max(5, n // 10)
            scale = np.std(resid[-tail_len:], ddof=1) if tail_len >= 3 else np.std(resid, ddof=1)
            scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
            r = resid / scale
        else:
            r = resid / sigma_total
            

        r = r[np.isfinite(r)]
        n_r = r.size
        
        if n_r < 10:
            results.append({"Label": label, "status": "too_few_residuals"})
            continue
        
        # --- PSD Computation (NumPy only) ---
        # Using periodogram estimate: P(f) = |FFT(x)|² / (fs * N)
        # fs = 1.0 (normalized frequency)
        f_psd = np.fft.rfftfreq(n_r, d=1.0)
        fft_vals = np.fft.rfft(r)
        Pxx_den = (np.abs(fft_vals)**2) / n_r  
        # Normalize for one-sided scaling (x2 for non-DC/Nyquist components)
        Pxx_den[1:-1] *= 2

        r_mean = float(np.mean(r))
        r_std = float(np.std(r, ddof=1))

        # ACF up to max_lag
        L = int(min(max_lag, n_r - 1))
        r0 = r - np.mean(r)
        denom = np.dot(r0, r0) + 1e-30
        acf = np.array([np.dot(r0[:-k], r0[k:]) / denom for k in range(1, L + 1)]) if L >= 1 else np.array([])

        # Ljung–Box Q test (approx chi-square)
        m_lb = int(min(ljung_box_lags, L))
        if m_lb >= 1:
            rho = acf[:m_lb]
            # Ljung-Box Q statistic
            Q = float(n_r * (n_r + 2) * np.sum((rho ** 2) / (n_r - np.arange(1, m_lb + 1))))
            pval = float(1.0 - chi2.cdf(Q, df=m_lb))
        else:
            Q, pval = np.nan, np.nan

        # simple whiteness rule-of-thumb
        bound = 1.96 / np.sqrt(n_r)
        acf_ok = True if acf.size == 0 else bool(np.all(np.abs(acf[:m_lb]) < bound))
        is_white = bool((np.isfinite(pval) and pval > alpha) and (abs(r_mean) < 0.2) and (r_std < 1.5) and acf_ok)

        results.append({
            "Label": label,
            "status": "ok",
            "n": int(n_r),
            "t_lags": t,
            "resid": r,
            "resid_mean": r_mean,
            "resid_std": r_std,
            "Q": Q,
            "p_value": pval,
            "white_residuals": is_white,
            "acf_lags": np.arange(1, L + 1),
            "acf": acf,
            "psd": (f_psd, Pxx_den)
        })

        if plot:
            ax_r, ax_acf, ax_psd = axes[i, 0], axes[i, 1], axes[i, 2]#, axes[i, 3]

            # 1. Residuals vs Tau
            ax_r.plot(t, r, lw=1.2)
            ax_r.axhline(0, ls="--", alpha=0.6)
            ax_r.set_xscale("log")
            ax_r.set_title(f"{label} — standardized residuals", fontsize=11)
            ax_r.set_xlabel(r"$\tau$")
            ax_r.set_ylabel(r"$r(\tau)$")

            # 2. ACF
            if L >= 1:
                ax_acf.stem(np.arange(1, L + 1), acf, basefmt=" ")
                ax_acf.axhline(bound, ls="--", alpha=0.6)
                ax_acf.axhline(-bound, ls="--", alpha=0.6)
            ax_acf.axhline(0, lw=1.0, alpha=0.6)
            ax_acf.set_title(f"ACF (≤ {L} lags)", fontsize=11)
            ax_acf.set_xlabel("Lag")
            ax_acf.set_ylabel("ACF")
            
            # 3. PSD Plot (Power Spectral Density)
            ax_psd.semilogy(f_psd, Pxx_den)
            ax_psd.set_title("PSD of Residuals (Whiteness)", fontsize=11)
            ax_psd.set_xlabel("Frequency (1/sample)")
            ax_psd.set_ylabel("PSD")
            ax_psd.grid(True, which="both", alpha=0.3)

            # 4. Histogram + Gaussian Fit
            #ax_h.hist(r, bins=30, density=True, alpha=0.85, label='Residuals')
            
            # Overlay Standard Normal Gaussian N(0,1)
            #x_g = np.linspace(min(r.min(), -4), max(r.max(), 4), 100)
            #ax_h.plot(x_g, norm.pdf(x_g, 0, 1), 'r--', lw=2, label='N(0,1)')
            
            #ax_h.set_title("Residual Distribution vs N(0,1)", fontsize=11)
            #ax_h.set_xlabel("Standardized Residual (r)")
            #ax_h.set_ylabel("Density")
            #ax_h.legend(fontsize=8, loc='upper right')

            #txt = f"mean={r_mean:.2f}, std={r_std:.2f}\nLB p={pval:.3g} (m={m_lb})\nwhite={is_white}"
            #ax_h.text(
            #    0.02, 0.98, txt, transform=ax_h.transAxes, va="top", ha="left", fontsize=9,
            #    bbox=dict(boxstyle="round,pad=0.3", alpha=0.12, linewidth=0.8)
            #)

    if plot:
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    return results


def fit_g2_kww_many(
    g2_list,
    t_list,
    q_list=None,
    labels=None,
    # --- fit control ---
    fit_range=None,                # (t_min, t_max) or None
    subtract_baseline=True,
    fix_baseline=False,            # New: if True, A is fixed (to 0 if subtract_baseline=True, or baseline_val)
    baseline_val=0.0,              # New: value to fix A to if fix_baseline=True
    baseline_tail_frac=0.15,       # fraction of points used to estimate baseline from tail
    # --- robustness ---
    weights="log_density",         # "log_density" or "1/sigma^2" or "linear"
    sigma_list=None,               # New: list of sigma arrays matching g2_list, required if weights="1/sigma^2"
    irls_iters=4,                  # robust reweight iterations
    huber_k=1.5,                   # smaller -> more robust
    # --- parameter bounds ---
    beta_bounds=(0.1, 3.0),        # broadened default to allow stretched/compressed
    gamma_bounds=(1e-2, 1e12),     # broad; tighten if you know scale (1/s)
    B_bounds=(0.0, 1.2),           # slightly above 1 for normalization noise
    A_bounds=None,                 # if None -> A0 ± 0.05 (unless fixed)
    # --- error bars ---
    method="cov",                 # "cov" or "bootstrap"
    n_boot=100,                   # for bootstrap
    seed=0,
    # --- plotting ---
    plot=True,
    main_title="g2(τ) KWW fits across q",
    dpi=200,
):
    """
    Robust KWW fits for g2(τ) across many q curves.

    Model:
        g2(τ) = A + B * exp(-2 * (Gamma * τ)^beta)

    Returns
    -------
    results : list of dict
        Each dict includes A, B, Gamma, beta, errors/CI, and diagnostics.
    """

    g2_list = list(g2_list)
    t_list = list(t_list)
    if len(g2_list) != len(t_list):
        raise ValueError("g2_list and t_list must have the same length.")
    M = len(g2_list)

    if q_list is None:
        q_list = [np.nan] * M
    else:
        q_list = list(q_list)
        if len(q_list) != M:
            raise ValueError("q_list must match length of g2_list.")

    if labels is None:
        labels = [f"q={q_list[i]:.3g}" if np.isfinite(q_list[i]) else f"curve {i}" for i in range(M)]
    if len(labels) != M:
        raise ValueError("labels must match number of curves.")

    if weights == "1/sigma^2" and sigma_list is None:
        raise ValueError("sigma_list must be provided when weights='1/sigma^2'")
    if sigma_list is not None:
         sigma_list = list(sigma_list)

    rng = np.random.default_rng(seed)

    def kww(t, A, B, Gamma, beta):
        # Clip params to safe ranges to avoid overflow/underflow inside exp
        # Gamma > 0, beta > 0 enforced by bounds, but good to be safe
        params_safe = (Gamma * t)
        # Avoid overflow in power
        # mask = params_safe > 0
        # term = np.zeros_like(t)
        # term[mask] = params_safe[mask] ** beta
        # return A + B * np.exp(-2.0 * term)
        return A + B * np.exp(-2.0 * np.power(params_safe, beta))

    def _prep_curve(t_raw, y_raw, sigma_raw=None):
        t = np.asarray(t_raw, float)
        y = np.asarray(y_raw, float)
        sig = np.asarray(sigma_raw, float) if sigma_raw is not None else None

        m = np.isfinite(t) & np.isfinite(y) & (t > 0)
        t, y = t[m], y[m]
        if sig is not None: sig = sig[m]

        if fit_range is not None:
            tmin, tmax = fit_range
            m2 = (t >= tmin) & (t <= tmax)
            t, y = t[m2], y[m2]
            if sig is not None: sig = sig[m2]

        if t.size < 10:
            return None

        # sort by t
        s = np.argsort(t)
        t, y = t[s], y[s]
        if sig is not None: sig = sig[s]

        # Initial Estimates
        # 1. Baseline A0
        if fix_baseline:
            A0 = baseline_val
        elif subtract_baseline:
            # Estimate from tail
            tail_len = max(5, int(np.ceil(baseline_tail_frac * t.size)))
            A0 = float(np.mean(y[-tail_len:]))
        else:
            A0 = float(np.min(y))

        # 2. Amplitude B0
        # Estimate from head (average of first few points to be robust to noise)
        y_head = np.mean(y[:3])
        B0 = float(max(y_head - A0, 1e-4))

        # 3. Linearized fit for Gamma and beta
        # Model: y = A + B * exp(-2*(G*t)^b)
        # (y - A)/B = exp(...)
        # ln((y - A)/B) = -2*(G*t)^b
        # ln[ -0.5 * ln((y - A)/B) ] = b * ln(G) + b * ln(t)
        # Y_lin = intercept + slope * X_lin
        # slope = b, intercept = b*ln(G) => G = exp(intercept / slope)

        # Select valid range for linearization (0.1 < signal < 0.9 of decay)
        # Normalized signal S = (y - A0)/B0
        S = (y - A0) / B0
        mask_lin = (S > 0.05) & (S < 0.95)
        
        Gamma0 = 1.0 # default
        beta0 = 1.0  # default
        
        if np.sum(mask_lin) > 5:
            try:
                t_lin = t[mask_lin]
                S_lin = S[mask_lin]
                
                # Y = ln( -0.5 * ln(S) )
                # Note: ln(S) is negative, so -0.5*ln(S) is positive.
                # If S > 1 (noise), ln(S) > 0 -> -0.5*ln(S) < 0 -> log fails.
                # So we ensure S < 1 implicitly by mask S < 0.95
                
                val_inner = -0.5 * np.log(S_lin)
                # Filter again for safety
                m_log = val_inner > 1e-9
                if np.sum(m_log) > 4:
                    Y_bias = np.log(val_inner[m_log])
                    X_bias = np.log(t_lin[m_log])
                    
                    # Robust line fit (Theil-Sen or simple polyfit)
                    res = stats.linregress(X_bias, Y_bias)
                    beta_est = res.slope
                    intercept = res.intercept
                    
                    if 0.1 < beta_est < 4.0:
                        beta0 = beta_est
                        # intercept = beta * ln(Gamma) -> Gamma = exp(intercept / beta)
                        Gamma0 = np.exp(intercept / beta0)
            except:
                pass # Fallback to default
        else:
            # Fallback Gamma: 1/time at S=0.5 (1/e decay is roughly fit by stretched exp)
            # 1/e corresponds to value approx 0.36
            # Half decay time is easier
            idx = np.argmin(np.abs(S - 0.5))
            Gamma0 = 1.0 / t[idx] if t[idx] > 0 else 1.0

        Gamma0 = np.clip(Gamma0, gamma_bounds[0], gamma_bounds[1])
        beta0 = np.clip(beta0, beta_bounds[0], beta_bounds[1])

        # weighting
        if weights == "1/sigma^2" and sig is not None:
            w = 1.0 / (sig**2 + 1e-12)
            # Normalize w for stability
            w = w / np.median(w)
        elif weights == "log_density":
            # w ~ 1/sqrt(Δlog t)
            lt = np.log(t)
            dlt = np.diff(lt)
            w = np.ones_like(t)
            if dlt.size > 0:
                dens = np.r_[dlt[0], 0.5*(dlt[:-1] + dlt[1:]), dlt[-1]]
                w = np.sqrt(np.maximum(dens, 1e-6))
            w = w / np.median(w)
        else:
            w = np.ones_like(t)

        return t, y, sig, A0, B0, Gamma0, beta0, w

    def _irls_fit(t, y, p0, bounds, w0, fixed_A=None):
        # p0: [A, B, Gamma, beta] if A not fixed
        # p0: [B, Gamma, beta] if A fixed
        
        def func_wrapper(t, *p):
            if fixed_A is not None:
                return kww(t, fixed_A, *p)
            else:
                return kww(t, *p)

        w = w0.copy()
        popt, pcov = None, None
        
        # Prepare bounds for curve_fit
        # bounds_in: ([low_A, low_B...], [high_A, high_B...])
        if fixed_A is not None:
             # Remove A from bounds (index 0)
             # B, Gamma, beta
             b_lower = [bounds[0][1], bounds[0][2], bounds[0][3]]
             b_upper = [bounds[1][1], bounds[1][2], bounds[1][3]]
             curr_bounds = (b_lower, b_upper)
             curr_p0 = p0[1:] # B, G, b
        else:
             curr_bounds = bounds
             curr_p0 = p0

        for _ in range(max(1, irls_iters)):
            sigma = 1.0 / np.sqrt(np.maximum(w, 1e-10))

            try:
                popt, pcov = curve_fit(
                    func_wrapper, t, y, p0=curr_p0, bounds=curr_bounds,
                    sigma=sigma, absolute_sigma=False,
                    maxfev=10000
                )
            except RuntimeError:
                # If fit fails, break loop and return None or last valid
                break

            yfit = func_wrapper(t, *popt)
            r = y - yfit
            
            # Update weights (Huber)
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            u = r / s
            w_h = np.ones_like(u)
            mask_outlier = np.abs(u) > huber_k
            w_h[mask_outlier] = huber_k / np.abs(u[mask_outlier])
            
            w = w0 * w_h
            curr_p0 = popt # Warm start

        # Reconstruct full params if A fixed
        if popt is not None and fixed_A is not None:
            full_popt = np.r_[fixed_A, popt]
            # Adjust pcov to include A row/col (zeros)
            full_pcov = np.zeros((4,4))
            full_pcov[1:, 1:] = pcov
            return full_popt, full_pcov
        
        return popt, pcov

    from scipy import stats # For linregress

    results = []
    
    # plot setup
    if plot:
        nshow = min(M, 12)
        ncols = 4
        nrows = int(np.ceil(nshow / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.2*nrows), dpi=dpi)
        axes = np.array(axes).reshape(-1) # flat array easier to index
        fig.suptitle(main_title, fontsize=14, y=0.995)

    for i in range(M):
        lab = labels[i]
        
        sig_i = sigma_list[i] if sigma_list is not None else None
        prep = _prep_curve(t_list[i], g2_list[i], sig_i)
        
        if prep is None:
            results.append({"label": lab, "q": q_list[i], "status": "too_few_points"})
            continue
            
        t, y, sig, A0, B0, G0, beta0, w0 = prep
        
        # Determine fixed A
        curr_fixed_A = baseline_val if fix_baseline else None
        
        # Bounds
        if A_bounds is None:
            Ab = (A0 - 0.05, A0 + 0.05)
        else:
            Ab = A_bounds
            
        bounds = (
            [Ab[0], B_bounds[0], gamma_bounds[0], beta_bounds[0]],
            [Ab[1], B_bounds[1], gamma_bounds[1], beta_bounds[1]],
        )
        p0 = [A0, B0, G0, beta0]

        # Fit
        status = "ok"
        popt, pcov = _irls_fit(t, y, p0, bounds, w0, fixed_A=curr_fixed_A)
        
        if popt is None:
             results.append({"label": lab, "q": q_list[i], "status": "fit_failed"})
             continue
             
        # Extract Results
        A, B, Gamma, beta = popt
        yfit = kww(t, A, B, Gamma, beta)
        
        # Metrics
        resid = y - yfit
        rmse = float(np.sqrt(np.mean(resid**2)))
        
        # Errors
        perr = np.zeros(4)
        if pcov is not None:
            perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        
        A_err, B_err, Gamma_err, beta_err = perr
        ci = None
        
        # Bootstrap if requested
        if method == "bootstrap":
             boots = []
             boot_p0 = popt if curr_fixed_A is None else popt[1:] # if A fixed, p0 is 3 params
             
             for b in range(n_boot):
                 idx = rng.integers(0, t.size, size=t.size)
                 tb, yb, wb = t[idx], y[idx], w0[idx]
                 # sort
                 s = np.argsort(tb)
                 tb, yb, wb = tb[s], yb[s], wb[s]
                 
                 popt_b, _ = _irls_fit(tb, yb, p0, bounds, wb, fixed_A=curr_fixed_A)
                 if popt_b is not None:
                     boots.append(popt_b)
             
             boots = np.array(boots)
             if len(boots) > 10:
                 # Stdev as error
                 perr_boot = np.std(boots, axis=0)
                 A_err, B_err, Gamma_err, beta_err = perr_boot
                 ci = {
                     "A": tuple(np.percentile(boots[:,0], [2.5, 97.5])),
                     "B": tuple(np.percentile(boots[:,1], [2.5, 97.5])),
                     "Gamma": tuple(np.percentile(boots[:,2], [2.5, 97.5])),
                     "beta": tuple(np.percentile(boots[:,3], [2.5, 97.5])),
                 }

        results.append({
            "label": lab,
            "q": float(q_list[i]) if np.isfinite(q_list[i]) else np.nan,
            "status": status,
            "A": A, "A_err": A_err,
            "B": B, "B_err": B_err,
            "Gamma": Gamma, "Gamma_err": Gamma_err,
            "beta": beta, "beta_err": beta_err,
            "rmse": rmse,
            "g2_fit": yfit, # full fitted curve
            "t_used": t,    # corresponding time points
            "ci95": ci,
        })
        
        # Plot
        if plot and i < len(axes):
            ax = axes[i]
            ax.errorbar(t, y, fmt="o", ms=3, alpha=0.4, color='gray', label="data")
            ax.plot(t, yfit, lw=1.5, color='red', label="KWW")
            ax.set_xscale("log")
            # ax.set_yscale("linear") # g2 is linear y usually
            ax.set_title(lab, fontsize=9)
            
            # Annotation
            txt = f"G={Gamma:.2e}\nb={beta:.2f}"
            ax.text(0.05, 0.05, txt, transform=ax.transAxes, fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7))
            
            if i==0: ax.legend(fontsize=7)

    if plot:
         # Hide unused
         for j in range(M, len(axes)):
             axes[j].axis('off')
         plt.tight_layout()
         plt.show()

    return results
def get_gamma_distribution_robust(
    g2_array,
    t_lags_array,
    gamma_min=1e2,
    gamma_max=1e9,
    n_gamma=200,
    alpha=1e-4,  # Small regularization helps convergence
    subtract_baseline=True
):
    results_list = []
    Gamma = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_gamma)
    
    for i in range(len(g2_array)):
        t = np.asarray(t_lags_array[i], float)
        g2 = np.asarray(g2_array[i], float)
        
        # 1. Basic Cleaning
        mask = np.isfinite(t) & np.isfinite(g2) & (t > 0)
        t, g2 = t[mask], g2[mask]
        
        if len(t) < 10:
            print(f"Data {i}: Too few points after finite mask.")
            continue
        
        # 2. Baseline Handling
        if subtract_baseline:
            # Take the average of the last 5% of the data
            baseline = np.mean(g2[-max(2, len(g2)//20):])
            g = g2 - baseline
        else:
            g = g2.copy()

        # 3. Robust Positivity Mask
        # Extremely noisy data might dip below 0. 
        # We only keep points where g > 0 to allow log-space physics.
        m = g > 0
        if np.sum(m) < 5:
            print(f"Data {i}: No positive points after baseline subtraction (Baseline too high?)")
            continue
            
        t_fit, g_fit = t[m], g[m]

        # 4. NNLS Solver
        K = np.exp(-2.0 * np.outer(t_fit, Gamma))
        
        if alpha > 0:
            K_reg = np.vstack([K, alpha * np.eye(n_gamma)])
            g_reg = np.concatenate([g_fit, np.zeros(n_gamma)])
            G, rnorm = nnls(K_reg, g_reg)
        else:
            G, rnorm = nnls(K, g_fit)

        # 5. Normalization & Filtering
        G[G < 0.01 * np.max(G)] = 0 # Lowered threshold to 1% to avoid killing real signal
        area = np.trapz(G, Gamma)
        
        if area <= 0:
            print(f"Data {i}: NNLS returned all zeros.")
            continue
            
        G /= area

        # 6. Moment Calculation
        mean_gamma = np.trapz(Gamma * G, Gamma)
        variance = np.trapz((Gamma - mean_gamma)**2 * G, Gamma)
        pdi = variance / (mean_gamma**2) if mean_gamma > 0 else 0
        peak_gamma = Gamma[np.argmax(G)]

        results_list.append({
            "mean_gamma": mean_gamma,
            "peak_gamma": peak_gamma,
            "pdi": pdi,
            "Gamma_grid": Gamma,
            "G_dist": G
        })

    return results_list