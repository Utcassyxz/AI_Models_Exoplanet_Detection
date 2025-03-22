import numpy as np
import pandas as pd
import batman
import matplotlib.pyplot as plt

# Generate time array: 30 days at a 2-minute cadence
def create_time_array():
    return np.arange(0, 30, 2 / (24 * 60))

# Generate a physical transit light curve using batman
def generate_physical_transit(time, period, rp, a, inc, ecc=0, w=90):
    params = batman.TransitParams()
    params.t0 = 0                # Time of transit center
    params.per = period          # Orbital period
    params.rp = rp               # Planet-to-star radius ratio
    params.a = a                 # Semi-major axis to star radius ratio
    params.inc = inc             # Orbital inclination in degrees
    params.ecc = ecc             # Eccentricity
    params.w = w                 # Longitude of periastron
    params.u = [0.1, 0.3]        # Limb-darkening coefficients
    params.limb_dark = "quadratic"
    m = batman.TransitModel(params, time)
    return m.light_curve(params)

# Add variability (e.g., flares, rotation) to light curves
def add_variability(flux, time):
    # Simulate a flare as a Gaussian bump
    flare_center = np.random.uniform(0, 30)
    flare_width = 0.1
    flare_amplitude = 0.005
    flare = flare_amplitude * np.exp(-((time - flare_center) ** 2) / (2 * flare_width ** 2))
    
    # Simulate rotational modulation as a sinusoidal signal
    rotation_period = np.random.uniform(10, 30)
    rotation_amplitude = 0.01
    rotation = rotation_amplitude * np.sin(2 * np.pi * time / rotation_period)
    
    # Combine the original flux with the variability effects
    return flux + flare + rotation

# Generate synthetic light curves with augmentation
def create_dataset_with_augmentation(time, num_curves=1000, transit_fraction=0.5):
    dataset = []
    labels = []
    for _ in range(num_curves):
        period = np.random.uniform(3, 15)    # Orbital period
        rp = np.random.uniform(0.01, 0.1)      # Planet-to-star radius ratio
        a = np.random.uniform(5, 15)           # Semi-major axis to star radius ratio
        inc = np.random.uniform(85, 90)        # Orbital inclination
        
        if np.random.rand() < transit_fraction:
            # Generate a transit light curve
            flux = generate_physical_transit(time, period, rp, a, inc)
            # Add variability and Gaussian noise
            flux = add_variability(flux, time) + np.random.normal(0, 0.001, len(time))
            dataset.append(flux)
            labels.append(1)  # Transit present
        else:
            # Generate light curve with variability and noise (no transit)
            flux = add_variability(np.ones_like(time), time) + np.random.normal(0, 0.001, len(time))
            dataset.append(flux)
            labels.append(0)  # No transit
    return np.array(dataset), np.array(labels)

if __name__ == "__main__":
    # Create the time array for the light curves
    time = create_time_array()
    
    # Prompt user to select the number of light curves to generate
    print("Please choose the number of light curves to generate:")
    print("1: 1000")
    print("2: 10000")
    print("3: 15000")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        num_curves = 1000
    elif choice == "2":
        num_curves = 10000
    elif choice == "3":
        num_curves = 15000
    else:
        print("Invalid input. Defaulting to 1000 light curves.")
        num_curves = 1000

    print(f"Generating dataset with {num_curves} light curves...")
    X, y = create_dataset_with_augmentation(time, num_curves=num_curves)

    # Save the dataset to a CSV file
    column_names = [f"time_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=column_names)
    df['label'] = y
    output_file = "synthetic_light_curves_with_augmentation_15000.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

    # Plot an example light curve
    example_idx = np.random.randint(0, len(X))
    plt.plot(time, X[example_idx])
    plt.title(f"Example Light Curve (Label: {y[example_idx]})")
    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.show()
