import os

from heliopy import config
import heliopy.data.spice as dataspice

import numpy as np
import spiceypy
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS

data_dir = config['download_dir']
spice_dir = os.path.join(data_dir, 'spice')

_SPICE_SETUP = False


def setup_spice():
    """
    Method to download some common files that spice needs to do orbit
    calculations.
    """
    if not _SPICE_SETUP:
        for kernel in dataspice.generic_kernels:
            loc = dataspice.get_kernel(kernel.short_name)
            spiceypy.furnsh(loc)


def furnish(fname):
    """
    Furnish SPICE with a kernel.

    Parameters
    ----------
    fname : str or list
        Filename of a spice kernel to load, or list of filenames to load.

    See also
    --------
    heliopy.data.spice.get_kernel : For attempting to automatically download
                                    kernels based on spacecraft name.
    """
    if isinstance(fname, str):
        fname = [fname]
    for f in fname:
        spiceypy.furnsh(f)


class Trajectory:
    """
    A generic class for the trajectory of a single body.

    Objects are initially created using only the body. To perform
    the actual trajectory calculation run :meth:`generate_positions`.
    The generated positions are then available via. the attributes
    :attr:`times`, :attr:`x`, :attr:`y`, and :attr:`z`.

    Parameters
    ----------
    spacecraft : str
        Name of the target. The name must be present in the loaded kernels.

    Notes
    -----
    When an instance of this class is created, a leapseconds kernel and a
    planets kernel are both automatically loaded.
    """
    def __init__(self, target, observing_body='Sun'):
        """
        Generate positions from a spice kernel.

        Parameters
        ----------
        target :
        observing_body : str or int
            The observing body. Output position vectors are given relative to
            the position of this body. See
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html
            for a list of bodies.

        """
        self._target = target
        self._observing_body = observing_body

        # SPICE frame used. The coordinate system to return the positions in. See
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html for a list of frames.
        # This is set to J2000 as this can be used by the Astropy coordinate framework.
        self._spice_frame = 'J2000'

        # SPICE format for times
        self._fmt = '%Y %b %d, %H:%M:%S'

    def coordinate_and_velocity(self, time, correction="NONE"):
        """
        Returns the spatial position and velocity of the target at a specific time.

        Parameters
        ----------
        time :
        correction :

        Returns
        -------
        (position, velocity) : `~tuple`
            A two-element tuple containing position and velocity of the target. The first
            element of the tuple is the position of the target returned as an Astropy
            coordinate in the ICRS frame using the Cartesian representation.
            The second element of the tuple is the three-dimensional velocity components
            of the target velocity and is returned as three-element Astropy Quantity array.
            These entries are the component velocities of the target in the J2000 x, y, and
            z directions.  The velocity is quoted in kilometers per second.
        """

        # Format the input time
        spice_time = spiceypy.str2et(time.strftime(self._fmt))

        # Calculate
        position_and_velocity, lightTimes = spiceypy.spkezr(self._target,
                                                            spice_time,
                                                            self._spice_frame,
                                                            correction,
                                                            self._observing_body)
        # Extract the position and the velocity
        position = np.array(position_and_velocity)[0:3] * u.km
        velocities = np.array(position_and_velocity)[3:] * u.km / u.s

        return SkyCoord(position[0], position[1], position[2],
                        frame=ICRS,
                        representation_type='cartesian', obstime=time), velocities

    def coordinate(self, time, correction="NONE"):
        """
        Returns the spatial position of the target at a specific time.

        Parameters
        ----------
        time :
        correction :

        Returns
        -------
        `~astropy.coordinates.SkyCoord`
        """
        return self.coordinate_and_velocity(time, correction=correction)[0]

    def velocity(self, time, correction="NONE"):
        """
        Returns the velocity of the target at a specific time.

        Parameters
        ----------
        time :
        correction :

        Returns
        -------
        `~astropy.units.Quantity`
        """
        return self.coordinate_and_velocity(time, correction=correction)[1]

    def speed(self, time, correction="NONE"):
        """
        Returns the speed of the target at a specific time.

        Parameters
        ----------
        time :
        correction :

        Returns
        -------

        """
        v = self.velocity(time, correction=correction)
        return np.sqrt(np.sum(v**2))

    @property
    def observing_body(self):
        """
        Observing body. The position vectors are all specified relative to
        this body.
        """
        return self._observing_body

    @property
    def target(self):
        """
        The body whose coordinates are being calculated.
        """
        return self._target

    @property
    def spice_frame(self):
        """
        The coordinate frame used by SPICE.
        """
        return self._spice_frame
