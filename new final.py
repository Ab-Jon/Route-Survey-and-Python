import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import numpy as np

# Read data
df = pd.read_csv("intech_data_test.csv", header=None)
df.columns = ["Point ID", "Northing", "Easting", "Elevation", "Position"]

# Filter positions
df_cl = df[df["Position"] == "CL"].sort_values('Point ID').reset_index(drop=True)
df_rhs = df[df["Position"] == "RHS"].sort_values('Point ID').reset_index(drop=True)
df_lhs = df[df["Position"] == "LHS"].sort_values('Point ID').reset_index(drop=True)

# Function to create GeoDataFrame
def create_gdf(data):
    geometry = [Point(xy) for xy in zip(data['Easting'], data['Northing'])]
    return gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:32632')  # UTM Zone 32N


# Create GeoDataFrames
gdf_cl = create_gdf(df_cl)
gdf_rhs = create_gdf(df_rhs)
gdf_lhs = create_gdf(df_lhs)


# Convert to lat/lon for plotting on real map later
gdf_cl_latlon = gdf_cl.to_crs(epsg=4326)
gdf_rhs_latlon = gdf_rhs.to_crs(epsg=4326)
gdf_lhs_latlon = gdf_lhs.to_crs(epsg=4326)


# Plot static map of all three lines
plt.figure(figsize=(8, 8))
plt.plot(gdf_cl_latlon.geometry.x, gdf_cl_latlon.geometry.y, '-o', color='green', label='Center Line (CL)')
plt.plot(gdf_rhs_latlon.geometry.x, gdf_rhs_latlon.geometry.y, '-o', color='blue', label='Right Hand Side (RHS)')
plt.plot(gdf_lhs_latlon.geometry.x, gdf_lhs_latlon.geometry.y, '-o', color='red', label='Left Hand Side (LHS)')
plt.xlabel("Eastings")
plt.ylabel("Northings")
plt.legend()
plt.title("Road Alignment: CL, RHS, LHS")
plt.grid(True)
plt.tight_layout()
plt.savefig("road_alignment.png")
plt.show()


# Distance along CL
distances = [0]
for i in range(1, len(gdf_cl)):
    prev = gdf_cl.geometry[i - 1]
    curr = gdf_cl.geometry[i]
    distances.append(distances[-1] + prev.distance(curr))

gdf_cl['Distance_m'] = distances
gdf_rhs['Distance_m'] = distances
gdf_lhs['Distance_m'] = distances


# Elevation profile for CL
plt.figure(figsize=(10, 5))
plt.plot(gdf_cl['Distance_m'], gdf_cl['Elevation'], marker='o', color='green', linewidth=2)
plt.title('Elevation profile along the road (CL)')
plt.xlabel('Distance along route (metres)')
plt.ylabel('Elevation (metres)')
plt.grid(True)
plt.tight_layout()
plt.savefig("elevation_profile.png")
plt.show()



# Cross Sections 
for i in range(len(gdf_cl)):
    plt.figure(figsize=(6,4))
    x_vals = [-5, 0, 5]  # offsets from CL in metres
    y_vals = [gdf_lhs['Elevation'][i], gdf_cl['Elevation'][i], gdf_rhs['Elevation'][i]]
    plt.plot(x_vals, y_vals, marker='o', color='blue')
    plt.axhline(y=np.mean(y_vals), color='red', linestyle='--', label='Design Level')
    plt.title(f"Cross Section at {gdf_cl['Distance_m'][i]:.1f} m")
    plt.xlabel("Offset (m)")
    plt.ylabel("Elevation (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"cross_section_{i}.png")
    plt.close()


# Design Profile 
start_elev = gdf_cl['Elevation'].iloc[0]
end_elev = gdf_cl['Elevation'].iloc[-1]
design_elev = np.linspace(start_elev, end_elev, len(gdf_cl))
gdf_cl['Design_Elev'] = design_elev 


# Plot Design vs Existing 
plt.figure(figsize=(10,5))
plt.plot(gdf_cl['Distance_m'], gdf_cl['Elevation'], label='Existing Ground', color='blue')
plt.plot(gdf_cl['Distance_m'], gdf_cl['Design_Elev'], label='Design Level', color='red', linestyle='--')
plt.title("Road Profile: Existing vs Design")
plt.xlabel("Distance (m)")
plt.ylabel("Elevation (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("design_vs_existing.png")
plt.show()


# Volume Computation using Average End Area Method
cut_vol = 0
fill_vol = 0
section_width = 10  # m

for i in range(1, len(gdf_cl)):
    # Areas for previous and current section
    area_prev = (gdf_cl['Design_Elev'][i-1] - gdf_cl['Elevation'][i-1]) * section_width
    area_curr = (gdf_cl['Design_Elev'][i] - gdf_cl['Elevation'][i]) * section_width
    
    # Separate into cut/fill
    if area_prev > 0 and area_curr > 0:
        fill_vol += ((area_prev + area_curr) / 2) * (gdf_cl['Distance_m'][i] - gdf_cl['Distance_m'][i-1])
    elif area_prev < 0 and area_curr < 0:
        cut_vol += ((abs(area_prev) + abs(area_curr)) / 2) * (gdf_cl['Distance_m'][i] - gdf_cl['Distance_m'][i-1])

print(f"Total Fill Volume: {fill_vol:.2f} m³")
print(f"Total Cut Volume: {cut_vol:.2f} m³")


