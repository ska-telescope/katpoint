History
=======

1.0a3 (2024-07-10)
------------------
* Rework tropospheric refraction code to use ERFA routines (!35)
* Add all outstanding changes from GitHub 0.10 release [see 0.10 below] (!30)
* Fix unit tests on newer Astropy and upgrade CI build system (!31 - !36, !38)

1.0a2 (2022-09-16)
------------------
* Use Astropy units for baselines, UVW, flux density, Catalogue (!24)
* Fix unit tests based on the Moon for Astropy 5.0 (!27)
* Upgrade SKAO CI pipeline, remove deprecations, fix warnings (!26, !28)

1.0a1 (2021-04-14)
------------------
* A mammoth rewrite to use Astropy 4.1+ and sgp4 instead of PyEphem (!11, !21)
* Depend on Python 3.6+ and remove all Python 2 fluff (!3, !12, !23)
* Use pytest and compare coordinates against Calc and Skyfield (!4, !7, !14)
* The new Body class encapsulates and vectorises coordinates (!6, !22)
* Timestamp/Antenna/Target mirrors Time/EarthLocation/SkyCoord (!5, !10, !22)
* Add tropospheric delays and proper NIAO to DelayCorrection (!13, !18, !20)

0.10 (2021-04-20)
-----------------
* [The following parts of the 0.10 release were retrofitted as part of 1.0a3]
* Handle out-of-range projection inputs (#67, #68)
* Allow fixed parameters during pointing model fitting (#73)
* Update pointing and other documentation (#65, #70, #71, #74)

0.9 (2019-10-02)
----------------
* Add Antenna.array_reference_antenna utility function (#51)
* Vectorise Target.uvw (#49)
* Improve precision of flux model description string (#52)
* Produce documentation on readthedocs.org (#48)
* Add script that converts PSRCAT database into Catalogue (#16)

0.8 (2019-02-12)
----------------
* Improve UVW coordinates by using local and not global North (#46)
* Allow different target with same name in Catalogue (#44)
* Add support for polarisation in flux density models (#38)
* Fix tab completion in Catalogue (#39)
* More Python 3 and flake8 cleanup (#43)
* The GitHub repository is now public as well

0.7 (2017-08-01)
----------------
* Support Python 3 (#36)
* Improve DelayCorrection, adding description string and offset (#37)

0.6.1 (2017-07-20)
------------------
* Resolve issue with ska-sa/katdal#85 - SensorData rework (#34)

0.6 (2016-09-16)
----------------
* Initial release of katpoint
