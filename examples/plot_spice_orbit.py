"""
SPICE orbit plotting
====================

How to plot orbits from SPICE kernels.

In this example we download the Parker Solar Probe SPICE kernel, and plot
its orbit for the first year.
"""


import heliopy.data.spice as spicedata
import heliopy.spice as spice
from datetime import datetime, timedelta
from astropy.time import Time
import astropy.units as u
import numpy as np

###############################################################################
# Set up SPICE by loading in some default kernels
spice.setup_spice()

# Load the Parker Solar Probe SPICE kernel. HelioPy will automatically fetch
# the latest kernel.
kernels = spicedata.get_kernel('psp')
kernels += spicedata.get_kernel('psp_pred')

# Furnish the Parker Solar Probe kernels to SPICE
spice.furnish(kernels)
psp = spice.Trajectory('SPP')

###############################################################################
# Generate a time for every day between starttime and endtime
starttime = Time("2018-09-02")
endtime = starttime + 365*u.day
times = []

###############################################################################
# Generate positions, speeds and elevations.  The positions and speeds are
# returned with units attached.  For ease of use later, the units are removed,
# stored in a list, and then that list is converted to an Astropy Quantity
# with the same units.  This is the recommended way of storing a Quantity
# array.
x_positions = []
y_positions = []
z_positions = []
radial_distances = []
speeds = []
elevations = []
while starttime < endtime:
    position = psp.coordinate(starttime)
    x_positions.append(position.x.to(u.au))
    y_positions.append(position.y.to(u.au))
    z_positions.append(position.z.to(u.au))
    radial_distance = np.sqrt(np.sum(position.x**2 + position.y**2 + position.z**2)).value
    radial_distances.append(radial_distance)
    elevations.append(np.rad2deg(np.arcsin(position.z.value / radial_distance)))
    speeds.append(psp.speed(starttime).value)
    times.append(starttime)
    starttime += 6*u.hour

x_positions = x_positions * u.au
y_positions = y_positions * u.au
z_positions = z_positions * u.au
radial_distances = (radial_distances * u.km).to(u.au)
elevations = elevations * u.deg
speeds = speeds * u.km/u.s
times_float = [(t - times[0]).to(u.s).value for t in times] * u.s
times = Time(times)

###############################################################################
# Plot the orbit. The orbit is plotted in 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.visualization import quantity_support
quantity_support()

# Generate a set of timestamps to color the orbits by
with quantity_support():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    kwargs = {'s': 3, 'c': times_float}
    ax.scatter(x_positions, y_positions, z_positions, **kwargs)
    ax.text(0, 0, 0, 'Sun', (1, 1, 0), color='k')
    ax.plot([-1, 1], [0, 0], [0, 0])
    ax.plot([0, 0], [-1, 1], [0, 0])
    ax.plot([0, 0], [0, 0], [-1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

###############################################################################
# Plot radial distance and elevation as a function of time

with quantity_support():
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(times.to_datetime(), radial_distances)
    axs[0].set_ylim(0, 1.1)
    axs[0].set_ylabel('r (AU)')

    axs[1].plot(times.to_datetime(), elevations)
    axs[1].set_ylabel('Elevation (deg)')

    axs[2].plot(times.to_datetime(), speeds)
    axs[2].set_ylabel('Speed (km/s)')

plt.show()
