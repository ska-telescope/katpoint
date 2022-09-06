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

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

import katpoint

ant = katpoint.Antenna('KAT7, -30:43:17.34, 21:24:38.46, 1038, 12.0')
freq = 1800.0 * u.MHz
freq_range = np.arange(900.0, 2100.0, 10.0) * u.MHz

old_all = katpoint.Catalogue(open('source_list.csv'),
                             antenna=ant, flux_frequency=freq)
old = old_all.filter(flux_limit=10 * u.Jy)
pks10 = katpoint.Catalogue(open('pkscat90_source_list.csv'),
                           antenna=ant, flux_frequency=freq)
pks = pks10.filter(flux_limit=10 * u.Jy)
jy1_all = katpoint.Catalogue(open('kuehr1Jy_source_list.csv'),
                             antenna=ant, flux_frequency=freq)
jy1 = jy1_all.filter(flux_limit=10 * u.Jy)

plot_rows = int(np.ceil(np.sqrt(len(old))))

plt.figure(1)
plt.clf()

for n, src in enumerate(old):
    flux = src.flux_density(freq)
    flux_str = f' {flux:.1f}' if not np.isnan(flux) else ''
    print(f'OLD: {src.names}{flux_str}')
    print(src.description)
    plt.subplot(plot_rows, plot_rows, n + 1)
    plt.plot(np.log10(freq_range.to_value(u.MHz)),
             np.log10(src.flux_density(freq_range).to_value(u.Jy)), 'b')
    jy1_src, min_dist = jy1.closest_to(src)
    if min_dist < 3 * u.arcmin:
        jy1_flux = jy1_src.flux_density(freq)
        jy1_flux_str = f' {jy1_flux:.1f}' if not np.isnan(jy1_flux) else ''
        print(f' --> 1JY: {jy1_src.names}{jy1_flux_str}')
        print(f'     {jy1_src.description}')
        plt.plot(np.log10(freq_range.to_value(u.MHz)),
                 np.log10(jy1_src.flux_density(freq_range).to_value(u.Jy)), 'r')
        jy1.remove(jy1_src.name)
    pks_src, min_dist = pks.closest_to(src)
    if min_dist < 3 * u.arcmin:
        pks_flux = pks_src.flux_density(freq)
        pks_flux_str = f' {pks_flux:.1f}' if not np.isnan(pks_flux) else ''
        print(f' --> PKS: {pks_src.names}{pks_flux_str}')
        print(f'     {pks_src.description}')
        plt.plot(np.log10(freq_range.to_value(u.MHz)),
                 np.log10(pks_src.flux_density(freq_range).to_value(u.Jy)), 'g')
        pks.remove(pks_src.name)
    plt.axis((np.log10(freq_range[0].to_value(u.MHz)),
              np.log10(freq_range[-1].to_value(u.MHz)), 0, 4))
    plt.xticks([])
    plt.yticks([])
    print()

plt.figtext(0.5, 0.93, 'Spectra (log S vs. log v) old=b, 1Jy=r, pks=g', ha='center', va='center')
plt.show()
