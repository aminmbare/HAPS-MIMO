from dataclasses import dataclass, field
from typing import List

@dataclass
class PathLossParametersSBand:
    los_sf: float      # S-band LOS shadow fading (σ_SF)
    nlos_sf: float     # S-band NLOS shadow fading (σ_SF)
    nlos_cl: float     # S-band NLOS clutter loss (CL)

@dataclass
class ScenarioDataSBand:
    los_probability: float
    path_loss: PathLossParametersSBand

@dataclass
class ElevationDataSBand:
    elevation: int
    dense_urban: ScenarioDataSBand
    urban: ScenarioDataSBand
    suburban_rural: ScenarioDataSBand

@dataclass
class CombinedDataTableSBand:
    data: List[ElevationDataSBand] = field(default_factory=list)
    
    def add_entry(self, elevation: int, 
                  dense_urban_los_prob: float, urban_los_prob: float, suburban_rural_los_prob: float,
                  dense_urban_path_loss: PathLossParametersSBand, urban_path_loss: PathLossParametersSBand, suburban_rural_path_loss: PathLossParametersSBand):
        entry = ElevationDataSBand(
            elevation=elevation,
            dense_urban=ScenarioDataSBand(dense_urban_los_prob, dense_urban_path_loss),
            urban=ScenarioDataSBand(urban_los_prob, urban_path_loss),
            suburban_rural=ScenarioDataSBand(suburban_rural_los_prob, suburban_rural_path_loss)
        )
        self.data.append(entry)

# Create an instance of CombinedDataTableSBand
combined_data_sband = CombinedDataTableSBand()

# Adding entries manually for each elevation angle based on the provided tables

# Elevation 10°
combined_data_sband.add_entry(
    elevation=10,
    dense_urban_los_prob=28.2,
    urban_los_prob=24.6,
    suburban_rural_los_prob=78.2,
    dense_urban_path_loss=PathLossParametersSBand(los_sf=3.5, nlos_sf=15.5, nlos_cl=34.3),
    urban_path_loss=PathLossParametersSBand(los_sf=4, nlos_sf=6, nlos_cl=34.3),
    suburban_rural_path_loss=PathLossParametersSBand(los_sf=1.79, nlos_sf=8.93, nlos_cl=19.52)
)

# Elevation 20°
combined_data_sband.add_entry(
    elevation=20,
    dense_urban_los_prob=33.1,
    urban_los_prob=38.6,
    suburban_rural_los_prob=86.9,
    dense_urban_path_loss=PathLossParametersSBand(los_sf=3.4, nlos_sf=13.9, nlos_cl=30.9),
    urban_path_loss=PathLossParametersSBand(los_sf=4, nlos_sf=6, nlos_cl=30.9),
    suburban_rural_path_loss=PathLossParametersSBand(los_sf=1.14, nlos_sf=9.08, nlos_cl=18.17)
)

# Elevation 30°
combined_data_sband.add_entry(
    elevation=30,
    dense_urban_los_prob=39.8,
    urban_los_prob=49.3,
    suburban_rural_los_prob=91.9,
    dense_urban_path_loss=PathLossParametersSBand(los_sf=2.9, nlos_sf=12.4, nlos_cl=29.0),
    urban_path_loss=PathLossParametersSBand(los_sf=4, nlos_sf=6, nlos_cl=29.0),
    suburban_rural_path_loss=PathLossParametersSBand(los_sf=1.14, nlos_sf=8.78, nlos_cl=18.42)
)

# Elevation 40°
combined_data_sband.add_entry(
    elevation=40,
    dense_urban_los_prob=46.8,
    urban_los_prob=61.3,
    suburban_rural_los_prob=92.9,
    dense_urban_path_loss=PathLossParametersSBand(los_sf=3.0, nlos_sf=11.7, nlos_cl=27.7),
    urban_path_loss=PathLossParametersSBand(los_sf=4, nlos_sf=6, nlos_cl=27.7),
    suburban_rural_path_loss=PathLossParametersSBand(los_sf=0.92, nlos_sf=10.25, nlos_cl=18.28)
)

# Elevation 50°
combined_data_sband.add_entry(
    elevation=50,
    dense_urban_los_prob=53.7,
    urban_los_prob=72.6,
    suburban_rural_los_prob=93.5,
    dense_urban_path_loss=PathLossParametersSBand(los_sf=3.1, nlos_sf=10.6, nlos_cl=26.8),
    urban_path_loss=PathLossParametersSBand(los_sf=4, nlos_sf=6, nlos_cl=26.8),
    suburban_rural_path_loss=PathLossParametersSBand(los_sf=1.42, nlos_sf=10.56, nlos_cl=18.63)
)

# Elevation 60°
combined_data_sband.add_entry(
    elevation=60,
    dense_urban_los_prob=61.2,
    urban_los_prob=80.5,
    suburban_rural_los_prob=94.0,
    dense_urban_path_loss=PathLossParametersSBand(los_sf=2.7, nlos_sf=10.5, nlos_cl=26.2),
    urban_path_loss=PathLossParametersSBand(los_sf=4, nlos_sf=6, nlos_cl=26.2),
    suburban_rural_path_loss=PathLossParametersSBand(los_sf=1.56, nlos_sf=10.74, nlos_cl=17.68)
)

# Elevation 70°
combined_data_sband.add_entry(
    elevation=70,
    dense_urban_los_prob=73.8,
    urban_los_prob=91.9,
    suburban_rural_los_prob=94.9,
    dense_urban_path_loss=PathLossParametersSBand(los_sf=2.5, nlos_sf=10.1, nlos_cl=25.8),
    urban_path_loss=PathLossParametersSBand(los_sf=4, nlos_sf=6, nlos_cl=25.8),
    suburban_rural_path_loss=PathLossParametersSBand(los_sf=0.85, nlos_sf=10.17, nlos_cl=16.50)
)

# Elevation 80°
combined_data_sband.add_entry(
    elevation=80,
    dense_urban_los_prob=82.0,
    urban_los_prob=96.8,
    suburban_rural_los_prob=95.2,
    dense_urban_path_loss=PathLossParametersSBand(los_sf=2.3, nlos_sf=9.2, nlos_cl=25.5),
    urban_path_loss=PathLossParametersSBand(los_sf=4, nlos_sf=6, nlos_cl=25.5),
    suburban_rural_path_loss=PathLossParametersSBand(los_sf=0.72, nlos_sf=11.52, nlos_cl=16.30)
)

# Elevation 90°
combined_data_sband.add_entry(
    elevation=90,
    dense_urban_los_prob=98.1,
    urban_los_prob=99.2,
    suburban_rural_los_prob=99.8,
    dense_urban_path_loss=PathLossParametersSBand(los_sf=1.2, nlos_sf=9.2, nlos_cl=25.5),
    urban_path_loss=PathLossParametersSBand(los_sf=4, nlos_sf=6, nlos_cl=25.5),
    suburban_rural_path_loss=PathLossParametersSBand(los_sf=0.72, nlos_sf=11.52, nlos_cl=16.30)
)

# Accessing the data for verification
for entry in combined_data_sband.data:
    print(f"Elevation: {entry.elevation}°")
    print("Dense Urban - LOS Probability:", entry.dense_urban.los_probability)
    print("Path Loss Parameters (S-band LOS σ_SF):", entry.dense_urban.path_loss.los_sf)
    print("Urban - LOS Probability:", entry.urban.los_probability)
    print("Path Loss Parameters (S-band NLOS CL):", entry.urban.path_loss.nlos_cl)
    print("Suburban/Rural - LOS Probability:", entry.suburban_rural.los_probability)
    print("Path Loss Parameters (S-band LOS σ_SF):", entry.suburban_rural.path_loss.los_sf)
    print("-----")
