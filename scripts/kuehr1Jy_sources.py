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
# Tool that extracts sources from the Catalog of Extragalactic Radio Sources Having Flux Densities
# Greater Than 1 Jy at 5 GHz (1Jy).
#
# This builds a katpoint catalogue from the included kuehr1Jy.vot file.
# This file is obtained as follows:
#
# - Visit the VizieR web site: http://vizier.u-strasbg.fr/
# - In the leftmost text entry field for the catalogue name, enter "1Jy"
#
# - Click on "VIII/5/sources" ("List of the 518 sources of the sample")
# - Add more search columns by clicking on "Columns with UCDs: ALL" button at bottom of page
# - Select unlimited maximum entries per table and computed J2000 output positions
# - Select at least the following fields: 1Jy 3C Fct A B C D
# - Select "VOTable" output layout and click on "Submit query"
# - This downloads a file named vizier_votable.vot
# - Rename file as kuehr1Jy.vot
#
# - Return to the top-level 1Jy page (there is a "VIII/5" button at the top left)
# - Click on "VIII/5/fluxes" ("The flux data for the sources")
# - Select at least the following fields: 1Jy Freq S
# - Select unlimited maximum entries per table and "VOTable" output layout, and click on "Submit query"
# - This downloads a file named vizier_votable.vot
# - Rename file as kuehr1Jy_flux.vot
#
# Thereafter, install the vo Python package from https://www.stsci.edu/trac/ssb/astrolib/
# (also referred to as votable2recarray). I used vo-0.5.tar.gz. Then this script can be
# run for the rest.
#
# Ludwig Schwardt
# 15 March 2010
#

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import katpoint

from astropy.table import Table

# Load tables in one shot (don't verify, as the VizieR VOTables contain a deprecated DEFINITIONS element)
table = Table.read('kuehr1Jy.vot')
flux_table = Table.read('kuehr1Jy_flux.vot')
src_strings = []
plot_freqs = [flux_table['Freq'].min(), flux_table['Freq'].max()]
test_log_freq = np.linspace(np.log10(plot_freqs[0]), np.log10(plot_freqs[1]), 200)
plot_rows = 8
plots_per_fig = plot_rows * plot_rows

# Iterate through sources
for src in table:
    names = '1Jy ' + src['_1Jy']
    if len(src['_3C']) > 0:
        names += ' | *' + src['_3C']
    radec_target = katpoint.Target.from_radec(src['_RAJ2000'] * u.deg,
                                              src['_DEJ2000'] * u.deg)
    tags_ra_dec = radec_target.add_tags('J2000').description
    # Extract flux data for the current source from flux table
    flux = flux_table[flux_table['_1Jy'] == src['_1Jy']]
    # Determine widest possible frequency range where flux is defined (ignore internal gaps in this range)
    # For better or worse, extend range to at least KAT7 frequency band (also handles empty frequency lists)
    flux_freqs = flux['Freq'].tolist() + [800, 2400]
    min_freq, max_freq = min(flux_freqs), max(flux_freqs)
    log_freq, log_flux = np.log10(flux['Freq']), np.log10(flux['S'])
    if src['Fct'] == 'LIN':
        # Coefficients A and B are float32 with precision=2 -> convert to float64 without fluff
        coefs = [float(repr(src['A'])), float(repr(src['B']))]
    elif src['Fct'] == 'EXP':
        coefs = [float(repr(src['A'])), float(repr(src['B'])), 0.0, 0.0, src['C'], src['D']]
    else:
        # No flux data found for source - skip it (only two sources, 1334-127 and 2342+82, are discarded)
        if len(flux) == 0:
            continue
        # Fit straight-line flux model log10(S) = a + b*log10(v) to frequencies close to KAT7 band
        mid_freqs = (flux['Freq'] > 400) & (flux['Freq'] < 12000)
        flux_poly = np.polyfit(log_freq[mid_freqs], log_flux[mid_freqs], 1)
        # Round to 4 significant digits
        coefs = [float(f'{c:.4g}') for c in flux_poly[::-1]]
    flux_model = katpoint.FluxDensityModel(min_freq * u.MHz, max_freq * u.MHz, coefs)
    src_strings.append(', '.join((names, tags_ra_dec, flux_model.description)) + '\n')
    print(src_strings[-1].strip())

    # Display flux model fit
    test_log_flux = np.log10(flux_model.flux_density(10 ** test_log_freq * u.MHz).to_value(u.Jy))
    plot_ind = len(src_strings) - 1
    plt.figure((plot_ind // plots_per_fig) + 1)
    if plot_ind % plots_per_fig == 0:
        plt.clf()
        plt.figtext(0.5, 0.93, 'Spectra (log S vs. log v) for sources %d to %d' %
                    (plot_ind + 1, plot_ind + plots_per_fig), ha='center', va='center')
    plt.subplot(plot_rows, plot_rows, 1 + plot_ind % plots_per_fig)
    plt.plot(log_freq, log_flux, 'ob')
    plt.plot(test_log_freq, test_log_flux, 'r')
    plt.xticks([])
    plt.yticks([])
    colorcode = 'g' if src['Fct'] == 'LIN' else 'y' if src['Fct'] == 'EXP' else 'k'
    plt.axvspan(np.log10(min_freq), np.log10(max_freq), facecolor=colorcode, alpha=0.5)
    plt.xlim(test_log_freq[0], test_log_freq[-1])

with open('kuehr1Jy_source_list.csv', 'w') as f:
    f.writelines(src_strings)

plt.show()
