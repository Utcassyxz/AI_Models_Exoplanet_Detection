import numpy as np
import pandas as pd
import batman

# Generate time array
def create_time_array():
    return np.arange(0, 30, 2 / (24 * 60))  # 30 days at 2-minute cadence

# Generate physical transit using batman
def generate_physical_transit(time, period, rp, a, inc, ecc=0, w=90):
    params = batman.TransitParams()
    params.t0 = 0                 # Time of transit center
    params.per = period          # Orbital period
    params.rp = rp               # Planet-to-star radius ratio
    params.a = a                 # Semi-major axis to star radius ratio
    params.inc = inc             # Orbital inclination
    params.ecc = ecc             # Eccentricity
    params.w = w                 # Longitude of periastron
    params.u = [0.1, 0.3]        # Limb-darkening coefficients
    params.limb_dark = "quadratic"
    m = batman.TransitModel(params, time)
    return m.light_curve(params)

# Generate synthetic light curves with variability and noise
def create_dataset_with_batman(time, num_curves=1000, transit_fraction=0.5):
    dataset = []
    labels = []
    for _ in range(num_curves):
        period = np.random.uniform(3, 15)
        rp = np.random.uniform(0.01, 0.1)  # Planet-to-star radius ratio
        a = np.random.uniform(5, 15)      # Semi-major axis to star radius ratio
        inc = np.random.uniform(85, 90)  # Orbital inclination in degrees
        if np.random.rand() < transit_fraction:
            flux = generate_physical_transit(time, period, rp, a, inc)
            noise = np.random.normal(0, 0.001, len(time))
            dataset.append(flux + noise)
            labels.append(1)  # Transit
        else:
            variability = 0.01 * np.sin(2 * np.pi * time / np.random.uniform(3, 15))
            noise = np.random.normal(0, 0.001, len(time))
            dataset.append(1 + variability + noise)
            labels.append(0)  # No transit
    return np.array(dataset), np.array(labels)

if __name__ == "__main__":
    time = create_time_array()
    X, y = create_dataset_with_batman(time, num_curves=50000)

    # Save dataset to CSV
    df = pd.DataFrame(X, columns=[f"time_{i}" for i in range(X.shape[1])])
    df['label'] = y
    df.to_csv("synthetic_light_curves_with_batman.csv", index=False)
    print("Dataset saved to synthetic_light_curves_with_batman.csv")