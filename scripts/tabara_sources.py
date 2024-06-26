#! /usr/bin/python

################################################################################
# Copyright (c) 2009-2022, National Research Foundation (SARAO)
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
# Tool that extracts sources from the "Catalogue of Linear Polarization of Radio
# Sources" (Tabara+ 1980).
#
# This builds a katpoint catalogue of potential leakage calibrators from the
# included tabara.vot file. This file is obtained as follows:
#
# - Visit the VizieR web site: http://vizier.u-strasbg.fr/
# - In the leftmost text entry field for the catalogue name, enter "tabara"
#
# - Click on "J/A+AS/39/379/table3"	("Source Characteristics for N=1510 sources")
# - Under "Preferences", select unlimited maximum entries per table, "VOTable"
#   output, compute J2000 coordinates and position in Decimal degrees
# - Select at least the following columns: Name Type S6cm RM IPA Pmax DEP PKS 3C OName
# - Click on "Submit"
# - This downloads a file named vizier_votable.vot
# - Rename file as tabara.vot
#
# - Go to "J/A+AS/39/379/table1" ("Data of linear polarization")
#   (there is a quick "..table1" link under Tables in Search Criteria)
# - Under "Preferences", select unlimited maximum entries and "VOTable" output
# - Select at least the following columns: Name lambda Pol PA
# - Click on "Submit"
# - This downloads a file named vizier_votable.vot
# - Rename file as tabara_pol.vot
#
# Thereafter, install the vo Python package from http://trac.assembla.com/astrolib
# (also referred to as vo.table). I used vo-0.7.2.tar.gz, which requires at least
# Python 2.6. Then this script can be run to do the rest.
#
# Ludwig Schwardt
# 7 October 2011
#

import numpy as np
import katpoint

import astropy.units as u
import astropy.constants as const
from astropy.table import Table

# Load tables in one shot (don't verify, as the VizieR VOTables contain a deprecated DEFINITIONS element)
table = Table.read('tabara.vot')
pol_table = Table.read('tabara_pol.vot')
# Use Kuehr 1Jy catalogue to provide flux density models
flux_cat = katpoint.Catalogue(open('kuehr1Jy_source_list.csv'))
# Use ATCA calibrator list to provide positions (and as a first-level check of source structure)
atca_cat = katpoint.Catalogue(open('atca_calibrators.csv'))

#
# SELECTION CRITERIA
#
# Select sources with > 1 Jy flux at 6cm (hopefully matching the 1Jy catalogue)
# This also includes flux = nan, which indicates unknown flux density for source
flux_limit = 1.0 * u.Jy
# Select sources south of +5 degrees declination
dec_limit = 5.0
# Select sources with absolute rotation measure less than 30 rad/m^2
# (allows a Q/U model that is reasonably independent of frequency over the KAT-7 band)
# This also includes RM = 0, which indicates unknown RM
rm_limit = 30.0
# Select sources with depolarisation wavelength > 15 cm, so that some linear polarisation remains at KAT-7 wavelengths
# Also include DEP = 0, which indicates unknown DEP
dep_limit = 15.0
# Select sources of these types ('' means unspecified source type)
accepted_types = ('QSO', 'GAL', '')
# Select sources that are also in ATCA calibrator list (and use positions from this list)
use_atca = True
# Select sources with > 0.2 Jy linearly polarised flux at 1822 MHz
# This also includes polflux = nan, which indicates unknown flux density for source (e.g. not in 1Jy catalogue)
frequency = 1822 * u.MHz
polflux_limit = 0.2 * u.Jy

# Iterate through sources
src_strings = []
for src in table:
    # Select sources based on various criteria
    if src['S6cm'] * u.Jy < flux_limit:
        print(f"{src['Name']} skipped: flux @ 6cm: {src['S6cm']:.2f} < {flux_limit:.2f}")
        continue
    if src['_DEJ2000'] > dec_limit:
        print(f"{src['Name']} skipped: dec {src['_DEJ2000']:.2f} > {dec_limit:.2f}")
        continue
    if np.abs(src['RM']) > rm_limit:
        print(f"{src['Name']} skipped: RM abs({src['RM']:.2f}) > {rm_limit:.2f}")
        continue
    if src['DEP'] > 0 and src['DEP'] < dep_limit:
        print(f"{src['Name']} skipped: DEP {src['DEP']:.2f} < {dep_limit:.2f}")
        continue
    if src['Type'] not in accepted_types:
        print(f"{src['Name']} skipped: type '{src['Type']}' not in {accepted_types}")
        continue
    if use_atca and src['Name'] not in atca_cat:
        print(f"{src['Name']} skipped: not an ATCA calibrator")
        continue
    names = '[TI80] ' + src['Name']
    if len(src['_3C']) > 0:
        names += ' | 3C ' + src['_3C']
        if src['_3C'].endswith('.0'):
            names += ' | 3C ' + src['_3C'][:-2]
    if len(src['PKS']) > 0:
        names += ' | PKS ' + src['PKS']
    if len(src['OName']) > 0:
        names += ' | ' + src['OName']
    if use_atca:
        radec_target = katpoint.Target(atca_cat[src['Name']].radec())
    else:
        radec_target = katpoint.Target.from_radec(src['_RAJ2000'] * u.deg,
                                                  src['_DEJ2000'] * u.deg)
    tags_ra_dec = radec_target.add_tags('J2000 ' + src['Type']).description
    # Extract polarisation data for the current source from pol table
    pol_data = pol_table[pol_table['Name'] == src['Name']]
    pol_freqs = const.c / pol_data['lambda']
    pol_percent = pol_data['Pol']
    # Remove duplicate frequencies and fit linear interpolator to data as function of frequency
    pol_freq, pol_perc = [], []
    for freq in np.unique(pol_freqs):
        freqfind = (pol_freqs == freq)
        pol_freq.append(freq)
        pol_perc.append(pol_percent[freqfind].mean())
    pol_interp = np.interp(frequency, pol_freq, pol_perc)
    # Look up source name in 1Jy catalogue and extract its flux density model
    flux_target = flux_cat['1Jy ' + src['Name']]
    if flux_target is None:
        flux_target = flux_cat['1Jy ' + src['Name'][:7]]
    flux_str = flux_target.flux_model.description if flux_target is not None else ''
    target_description = ', '.join((names, tags_ra_dec, flux_str))
    target = katpoint.Target(target_description)
    # Evaluate polarised flux at expected centre frequency and filter on that
    pol_flux = target.flux_density(frequency) * pol_interp / 100.
    if pol_flux < polflux_limit:
        print(f"{src['Name']} skipped: polarised flux @ {frequency:.0f}: {pol_flux:.2f} < {polflux_limit:.2f}")
        continue
    src_strings.append(target_description + '\n')
    print(f"{src['Name']} cat flux @ 6cm: {src['S6cm'] * u.Jy:.2f}, "
          f"model flux @ 6cm: {target.flux_density(6 * u.cm):.2f}, "
          f"@ {frequency:.0f}: {target.flux_density(frequency):.2f}, "
          f"%pol: {pol_interp:.2f} polflux: {pol_flux:.3f}")

with open('tabara_source_list.csv', 'w') as f:
    f.writelines(src_strings)
