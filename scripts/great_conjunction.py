#!/usr/bin/env python3

################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

#
# Annotate Jason de Freitas's epic Great Conjunction photograph.
#
# Recreate Jason's observation of the ISS passing between Jupiter and Saturn, based on
# https://petapixel.com/2020/12/22/photographer-captures-iss-passing-between-jupiter-and-saturn/
# This verifies both astrometric (ra, dec) and (az, el) coordinates for nearby objects.
#
# Ludwig Schwardt
# 18 February 2021
#

import argparse

import katpoint
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle

parser = argparse.ArgumentParser(description="Annotate Jason de Freitas's Great Conjunction photograph.")
parser.add_argument('filename', help='the image filename')
parser.add_argument('--azel', action='store_true', help='use (az, el) instead of (ra, dec)')
parser.add_argument('--refracted-azel', action='store_true',
                    help='use (az, el) corrected for atmospheric refraction')
args = parser.parse_args()

# COORDINATES

# Jason's Pentax 67 was at Mount Jellore Lookout in the Mount Gibraltar Reserve, near Bowral NSW.
# The quoted altitude is above sea level instead of above the WGS84 ellipsoid, but should be OK.
pentax = katpoint.Antenna('Jellore Lookout NSW, -34.462653, 150.427971, 864')
# The photo was taken on 17 December 2020 "at around 9:54pm" Australian Eastern Daylight Time (AEDT).
# AEDT is 11 hours ahead of UTC => therefore at around 10:54 UTC. Jason confirmed that he pressed
# the shutter button at 21:53:05 AEDT, so pick the middle of the 10-second exposure.
timestamp = katpoint.Timestamp('2020-12-17 10:53:10')
jupiter = katpoint.Target('Jupiter, special')
saturn = katpoint.Target('Saturn, special')
moon = katpoint.Target('Moon, special')
# TLE for ISS on 2020-12-17
iss = katpoint.Target('ISS (ZARYA), tle,'
                      '1 25544U 98067A   20351.71912775  .00000900  00000-0  24328-4 0  9992,'
                      '2 25544  51.6442 165.2978 0001589 133.0028 320.9621 15.49190988260311')
# The sources were quite low above the horizon, so get the weather needed for refraction. This was
# originally from https://www.timeanddate.com/weather/australia/bowral/historic but the data for
# 17 Dec 2020 now appears to be different. These values correlate with those at nearby Moss Vale
# station: http://www.bom.gov.au/climate/dwo/202012/html/IDCJDW2086.202012.shtml.
weather = dict(pressure=1003 * u.mbar, temperature=25 * u.deg_C, relative_humidity=0.69)


def relative_radec(target, reference, times):
    """Astrometric (ra, dec) of `target` relative to `reference` at `times`, in degrees."""
    target_radec = target.radec(times, pentax)
    x, y = reference.sphere_to_plane(target_radec.ra.rad, target_radec.dec.rad,
                                     timestamp, pentax, coord_system='radec')
    # Right ascension increases from right to left
    return -np.degrees(x), np.degrees(y)


def relative_azel(target, reference, times):
    """Topocentric (az, el) of `target` relative to `reference` at `times`, in degrees."""
    target_azel = target.azel(times, pentax)
    x, y = reference.sphere_to_plane(target_azel.az.rad, target_azel.alt.rad, timestamp, pentax)
    return np.degrees(x), np.degrees(y)


def relative_refracted_azel(target, reference, times):
    """Topocentric (az, el) of `target` relative to `reference`, incorporating refraction."""
    frame = AltAz(obstime=times.time, location=pentax.location, obswl=550 * u.nm, **weather)
    target_azel = target.body.compute(frame, times.time, pentax.location)
    az, el = target_azel.az.rad, target_azel.alt.rad
    ref_azel = reference.body.compute(frame, times.time, pentax.location)
    if not times.time.isscalar:
        ref_azel = ref_azel[len(times.time) // 2]
    az0, el0 = ref_azel.az.rad, ref_azel.alt.rad
    x, y = katpoint.sphere_to_plane['ARC'](az0, el0, az, el)
    return np.degrees(x), np.degrees(y)


# Pick Saturn as the reference point / origin
if args.refracted_azel:
    sphere_to_plane = relative_refracted_azel
elif args.azel:
    sphere_to_plane = relative_azel
else:
    sphere_to_plane = relative_radec
j_lon, j_lat = sphere_to_plane(jupiter, saturn, timestamp)
s_lon, s_lat = 0.0, 0.0
m_lon, m_lat = sphere_to_plane(moon, saturn, timestamp)
# Sample the ISS trajectory at 1-second intervals
i_lon, i_lat = sphere_to_plane(iss, saturn, timestamp + np.arange(-5, 6))
moon_diameter = 3474.2 * u.km
moon_width = (moon_diameter / moon.azel(timestamp, pentax).distance) * u.rad

# Now for the photo parameters, in pixels (estimated in Gimp and refined in Matplotlib)
photo_width, photo_height, dpi = 845, 1018, 144  # 5.87" x 7.07", very close to 6x7 negative
j_photo_x, j_photo_y, j_width = 395, 839, 11
s_photo_x, s_photo_y, s_width = 481.5, 792.2, 7  # Gimp: 482, 793, 7
m_photo_x, m_photo_y, m_width = 430, 177.5, 106  # Gimp: 430, 178, 106
i_photo_x1, i_photo_y1, i_photo_x2, i_photo_y2 = 584, 875, 251, 753

# ALIGNMENT

# Focus on two well-separated points (Saturn <-> Moon) to get rotation and scale of photo
origin_pix = np.array([s_photo_x, s_photo_y])
# Offset of the Moon from Saturn, in pixels
m_dx, m_dy = m_photo_x - origin_pix[0], m_photo_y - origin_pix[1]
ruler_deg = np.hypot(m_lon, m_lat)
ruler_pix = np.hypot(m_dx, m_dy)
scale = ruler_pix / ruler_deg
# Flip the y axis because image coordinates run from top to bottom
Fy = np.array([[1., 0.], [0., -1.]])
# Rotation matrix aligning (m_lon, m_lat) with (m_dx, -m_dy)
scaled_cos = m_lon * m_dx + m_lat * (-m_dy)
scaled_sin = m_lon * (-m_dy) - m_lat * m_dx
R = np.array([[scaled_cos, -scaled_sin],
              [scaled_sin,  scaled_cos]]) / ruler_pix / ruler_deg
# Rotate, flip, uniformly scale and translate coordinates to match photo
A = scale * Fy @ R
jupiter_pix = A @ [j_lon, j_lat] + origin_pix
saturn_pix = A @ [s_lon, s_lat] + origin_pix  # same as s_photo_x, s_photo_y
moon_pix = A @ [m_lon, m_lat] + origin_pix   # same as m_photo_x, m_photo_y
iss_pix = A @ [i_lon, i_lat] + origin_pix[:, np.newaxis]
moon_radius_pix = 0.5 * moon_width.to_value(u.deg) * scale

# PLOT

# Plot photo as an image that fills the figure, with pixel indices as data coordinates
fig = plt.figure(figsize=(photo_width / dpi, photo_height / dpi), facecolor='k')
ax = fig.add_axes([0, 0, 1, 1])
img = mpimg.imread(args.filename)
ax.imshow(img, aspect='equal')
# Annotate the image (red = assumed positions, yellow = estimated from red positions)
ax.add_patch(Circle(jupiter_pix, radius=20, ec='y', lw=1.5, fc='none'))
ax.add_patch(Circle(saturn_pix, radius=20, ec='r', lw=1.5, fc='none'))
ax.add_patch(Circle(moon_pix, radius=moon_radius_pix, ec='r', lw=3, fc='none'))
ax.plot(iss_pix[0], iss_pix[1], '.', mfc=(1, 1, 0, 0.5), ms=12, mec='none')
ax.axis('off')

plt.show()
