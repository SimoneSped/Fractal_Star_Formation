import matplotlib.gridspec as gridspec
import numpy as np

import matplotlib.pyplot as plt

def load_results(results):

    # Load data
    perimeters = np.array(results["perimeters"])
    areas = np.array(results["areas"])
    thresholds = np.array(results["thresholds"])

    # Compute logs
    log_perimeters = np.log10(perimeters)
    log_areas = np.log10(areas)

    # --- Define 1.6% relative uncertainty ---
    rel_error = 0.016

    # Uncertainties in P and A
    sigma_P_abs = perimeters * rel_error
    sigma_A_abs = areas * rel_error

    # Convert to uncertainties in log10(P) and log10(A)
    sigma_logP = sigma_P_abs / (perimeters * np.log(10))
    sigma_logA = sigma_A_abs / (areas * np.log(10))

    # Perform weighted linear regression using np.polyfit with weights
    # Weight is 1 / sigma_y
    weights = 1 / sigma_logA

    coefficients, cov_matrix = np.polyfit(
        log_perimeters, log_areas, deg=1, w=weights, cov=True
    )
    slope, intercept = coefficients
    slope_err = np.sqrt(cov_matrix[0,0])

    # Fractal dimension
    D = 2 / slope
    D_err = 2 * slope_err / slope**2

    # Correlation coefficient
    corr_coef = np.corrcoef(log_perimeters, log_areas)[0,1]
    print(f"Correlation coefficient: {corr_coef:.4f}")
    print(f"Slope: {slope:.4f} ± {slope_err:.4f}")
    print(f"Fractal dimension D = {D:.2f} ± {D_err:.2f}")
    return log_perimeters, log_areas, thresholds, sigma_logP, sigma_logA, slope, intercept, D, D_err

def plot_global_fractal_dimension(results, threshold_min, threshold_max, name_region=""):

    log_perimeters, log_areas, thresholds, sigma_logP, sigma_logA, slope, intercept, D, D_err = load_results(results)
    # --- Plotting ---
    norm = plt.Normalize(threshold_min, threshold_max)
    cmap = plt.cm.viridis

    # Compute residuals
    residuals = log_areas - (slope * log_perimeters + intercept)

    print(f"Average residual (very symmentrical): {np.mean(residuals):.4f}")
    print(f"Mean absolute residual: {np.mean(np.abs(residuals)):.4f}")

    # Set up figure with 2 rows: main plot + residuals
    fig = plt.figure(figsize=(9,9))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1], hspace=0.25)

    # Top: scatter + fit
    ax_main = fig.add_subplot(gs[0])

    sc = ax_main.scatter(
        log_perimeters, log_areas,
        c=thresholds, cmap=cmap, norm=norm, alpha=0.6, edgecolor="k"
    )
    x_fit = np.linspace(np.min(log_perimeters), np.max(log_perimeters), 100)
    y_fit = slope * x_fit + intercept
    ax_main.plot(
        x_fit, y_fit,
        color="red", lw=2,
        label=f"Fit: D = {D:.2f} ± {D_err:.2f}"
    )
    ax_main.errorbar(
        log_perimeters, log_areas,
        xerr=sigma_logP, yerr=sigma_logA,
        fmt='none', ecolor='gray', alpha=0.5, capsize=2, label="Uncertainty"
    )

    ax_main.set_ylabel(r"$\log_{10}(A)$", fontsize=18)
    ax_main.set_xlabel(r"$\log_{10}(P)$", fontsize=18)
    ax_main.legend(fontsize=13)
    ax_main.grid(True)
    ax_main.set_title(f"Global Fractal Dimension - {name_region}", fontsize=18)
    ax_main.set_xlim([log_perimeters[-1]-0.03, log_perimeters[0]+0.03])
    cbar = fig.colorbar(sc, ax=ax_main)
    cbar.set_label("Column Density Threshold [$\mathrm{cm}^{-2}$]", fontsize=15)

    # Bottom: residuals
    ax_res = fig.add_subplot(gs[1])
    ax_res.axhline(0, color="k", lw=1, ls="--")
    ax_res.scatter(
        log_perimeters, residuals,
        c=thresholds, cmap=cmap, norm=norm, alpha=0.6, edgecolor="k"
    )
    ax_res.errorbar(
        log_perimeters, residuals,
        yerr=sigma_logA,
        fmt='none', ecolor='gray', alpha=0.5, capsize=2
    )

    ax_res.set_xlabel(r"$\log_{10}(P)$", fontsize=15)
    ax_res.set_ylabel("Residuals", fontsize=15)
    ax_res.grid(True)

    # plt.tight_layout()
    plt.show()

def plot_global_fractal_dimension_double_fit():
    pass

def plot_global_fractal_dimension_together(results_OA, results_OB):
    plt.figure(figsize=(7, 8))

    # Plot with error bars
    plt.errorbar(results_OA["thresholds"], results_OA["fractal_dimension"], 
                yerr=results_OA["sigma_D"], fmt='o-', 
                label="Orion A", color='steelblue', markersize=6, markerfacecolor='none', capsize=3)

    plt.errorbar(results_OB["thresholds"], results_OB["fractal_dimension"], 
                yerr=results_OB["sigma_D"], fmt='s--', 
                label="Orion B", color='darkorange', markersize=6, markerfacecolor='none', capsize=3)

    # Title and labels
    plt.title("Fractal Dimension vs. Column Density Threshold", fontweight='bold', fontsize=15, pad=15)
    plt.xlabel('Column Density Threshold [$\mathrm{cm}^{-2}$]', fontsize=13)
    plt.ylabel('Fractal Dimension $D$', fontsize=13)

    # Log scale and grid
    plt.xscale("log")
    plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)

    # Tick adjustments
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=30)

    # Legend
    plt.legend(frameon=False, fontsize=14, loc='lower right')

    # Tight layout for slides
    plt.tight_layout()
