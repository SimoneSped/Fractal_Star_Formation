import matplotlib.gridspec as gridspec
from scipy.stats import linregress
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

def plot_global_fractal_dimension_double_fit(
    results,
    threshold_min,
    threshold_max,
    split_value,
    name_region=""
):

    def fit_segment(x, y, sigma_y):
        coeffs, cov = np.polyfit(
            x,
            y,
            1,
            w=1.0 / sigma_y,
            cov=True
        )

        slope, intercept = coeffs
        slope_err = np.sqrt(cov[0, 0])

        D = 2.0 / slope
        D_err = 2.0 * slope_err / slope**2

        residuals = y - (slope * x + intercept)
        corr = np.corrcoef(x, y)[0, 1]

        return {
            "slope": slope,
            "intercept": intercept,
            "slope_err": slope_err,
            "D": D,
            "D_err": D_err,
            "corr": corr,
            "residuals": residuals,
        }

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    thresholds = np.asarray(results["thresholds"])
    perimeters = np.asarray(results["perimeters"])
    areas = np.asarray(results["areas"])
    
    if not (
        len(thresholds)
        == len(perimeters)
        == len(areas)
    ):
        raise ValueError(
            "thresholds, perimeters, and areas must have equal lengths"
        )

    # ---------------------------------------------------------
    # Threshold selection
    # ---------------------------------------------------------

    mask_keep = (
        (thresholds >= threshold_min)
        & (thresholds <= threshold_max)
    )

    thresholds = thresholds[mask_keep]
    perimeters = perimeters[mask_keep]
    areas = areas[mask_keep]
    
    print(
        f"Keeping {len(thresholds)} points "
        f"between {threshold_min:.2e} and {threshold_max:.2e}"
    )

    # ---------------------------------------------------------
    # Remove invalid values for log space
    # ---------------------------------------------------------

    valid = (
        np.isfinite(perimeters)
        & np.isfinite(areas)
        & (perimeters > 0)
        & (areas > 0)
    )

    thresholds = thresholds[valid]
    perimeters = perimeters[valid]
    areas = areas[valid]

    if len(perimeters) < 5:
        raise ValueError(
            "Too few valid points remain after filtering."
        )

    # ---------------------------------------------------------
    # Log transform
    # ---------------------------------------------------------

    log_perimeters = np.log10(perimeters)
    log_areas = np.log10(areas)

    # ---------------------------------------------------------
    # Uncertainties
    # ---------------------------------------------------------

    rel_uncertainty = 0.016

    sigma_P = rel_uncertainty * perimeters
    sigma_A = rel_uncertainty * areas

    log_sigma_P = sigma_P / (perimeters * np.log(10))
    log_sigma_A = sigma_A / (areas * np.log(10))

    # ---------------------------------------------------------
    # Split data
    # ---------------------------------------------------------

    mask_low = log_perimeters <= split_value
    mask_high = log_perimeters > split_value

    n_low = mask_low.sum()
    n_high = mask_high.sum()

    print(f"Low range points : {n_low}")
    print(f"High range points: {n_high}")

    # ---------------------------------------------------------
    # Fits
    # ---------------------------------------------------------

    fit_low = fit_segment(
        log_perimeters[mask_low],
        log_areas[mask_low],
        log_sigma_A[mask_low]
    )

    fit_high = fit_segment(
        log_perimeters[mask_high],
        log_areas[mask_high],
        log_sigma_A[mask_high]
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print(
        f"\nLow log(P) <= {split_value}"
        f"\n  slope = {fit_low['slope']:.4f}"
        f" ± {fit_low['slope_err']:.4f}"
        f"\n  D = {fit_low['D']:.2f}"
        f" ± {fit_low['D_err']:.2f}"
        f"\n  corr = {fit_low['corr']:.4f}"
    )

    print(
        f"\nHigh log(P) > {split_value}"
        f"\n  slope = {fit_high['slope']:.4f}"
        f" ± {fit_high['slope_err']:.4f}"
        f"\n  D = {fit_high['D']:.2f}"
        f" ± {fit_high['D_err']:.2f}"
        f"\n  corr = {fit_high['corr']:.4f}"
    )

    # ---------------------------------------------------------
    # Residual diagnostics
    # ---------------------------------------------------------

    for label, residuals in [
        ("Low range", fit_low["residuals"]),
        ("High range", fit_high["residuals"]),
    ]:

        print(f"\nResiduals summary ({label})")

        print(
            f"Mean residual: {np.mean(residuals):.4e}"
        )

        print(
            f"Mean absolute residual: "
            f"{np.mean(np.abs(residuals)):.4f}"
        )

        print(
            f"RMS residual: "
            f"{np.sqrt(np.mean(residuals**2)):.4f}"
        )

        print(
            f"Std deviation: "
            f"{np.std(residuals):.4f}"
        )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(9, 9),
        height_ratios=[3, 1],
        sharex=True
    )

    norm = plt.Normalize(
        threshold_min,
        threshold_max
    )

    cmap = plt.cm.viridis

    sc = axs[0].scatter(
        log_perimeters,
        log_areas,
        c=thresholds,
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )

    axs[0].errorbar(
        log_perimeters,
        log_areas,
        xerr=log_sigma_P,
        yerr=log_sigma_A,
        fmt="none",
        ecolor="gray",
        alpha=0.3,
        capsize=2
    )

    x_low = np.linspace(
        log_perimeters[mask_low].min(),
        log_perimeters[mask_low].max(),
        100
    )

    x_high = np.linspace(
        log_perimeters[mask_high].min(),
        log_perimeters[mask_high].max(),
        100
    )

    axs[0].plot(
        x_low,
        fit_low["slope"] * x_low
        + fit_low["intercept"],
        color="blue",
        label=f"Low fit (D={fit_low['D']:.2f}±{fit_low['D_err']:.2f})"
    )

    axs[0].plot(
        x_high,
        fit_high["slope"] * x_high
        + fit_high["intercept"],
        color="red",
        label=f"High fit (D={fit_high['D']:.2f}±{fit_high['D_err']:.2f})"
    )

    split_idx = np.abs(
        log_perimeters - split_value
    ).argmin()

    split_threshold = thresholds[split_idx]

    axs[0].axvline(
        split_value,
        color="gray",
        linestyle="--",
        label=f"Split at N={split_threshold:.2e}"
    )

    cbar = fig.colorbar(sc, ax=axs[0])

    cbar.set_label(
        r"Column Density Threshold [$\mathrm{cm}^{-2}$]",
        fontsize=15
    )

    title = (
        f"Global Fractal Dimension - {name_region}"
        if name_region
        else "Global Fractal Dimension"
    )

    axs[0].set_title(title, fontsize=18)

    axs[0].set_ylabel(
        r"$\log_{10}(A)$",
        fontsize=18
    )

    axs[0].set_xlabel(
        r"$\log_{10}(P)$",
        fontsize=18
    )

    axs[0].set_xlim(
        log_perimeters.min() - 0.03,
        log_perimeters.max() + 0.03
    )

    axs[0].legend(fontsize=12)
    axs[0].grid(True)

    # Residuals

    axs[1].scatter(
        log_perimeters[mask_low],
        fit_low["residuals"],
        color="blue",
        alpha=0.5,
        label="Low residuals"
    )

    axs[1].scatter(
        log_perimeters[mask_high],
        fit_high["residuals"],
        color="red",
        alpha=0.5,
        label="High residuals"
    )

    axs[1].axhline(
        0,
        color="black",
        linestyle="--"
    )

    axs[1].set_xlabel(
        r"$\log_{10}(P)$",
        fontsize=15
    )

    axs[1].set_ylabel(
        "Residual",
        fontsize=15
    )

    axs[1].legend(fontsize=12)
    axs[1].grid(True)

    plt.tight_layout(h_pad=2.5)
    plt.show()

    return {
        "low": fit_low,
        "high": fit_high
    }

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
