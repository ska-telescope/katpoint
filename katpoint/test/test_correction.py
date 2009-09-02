"""Tests for the correction module."""
# pylint: disable-msg=C0103,W0212

import unittest

import numpy as np

import katpoint

def assert_angles_almost_equal(x, y, **kwargs):
    primary_angle = lambda x: x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi
    np.testing.assert_almost_equal(primary_angle(x - y), np.zeros(np.shape(x)), **kwargs)

class TestRefractionCorrection(unittest.TestCase):
    """Test refraction correction."""
    def setUp(self):
        self.rc = katpoint.RefractionCorrection()
        self.el = katpoint.deg2rad(np.arange(0.0, 90.1, 0.1))

    def test_refraction_closure(self):
        """Test closure between refraction correction and its reverse operation."""
        print repr(self.rc)
        self.assertRaises(ValueError, katpoint.RefractionCorrection, 'unknown')
        # Generate random meteorological data (hopefully sensible)
        temp = -10. + 50. * np.random.rand()
        pressure = 900. + 200. * np.random.rand()
        humidity = 5. + 90. * np.random.rand()
        # Test closure on el grid
        refracted_el = self.rc.apply(self.el, temp, pressure, humidity)
        reversed_el = self.rc.reverse(refracted_el, temp, pressure, humidity)
        assert_angles_almost_equal(reversed_el, self.el, decimal=7,
                                   err_msg='Elevation closure error for temp=%f, pressure=%f, humidity=%f' %
                                           (temp, pressure, humidity))

class TestPointingModel(unittest.TestCase):
    """Test pointing model."""
    def setUp(self):
        az_range = katpoint.deg2rad(np.arange(-180.0, 180.0, 5.0))
        el_range = katpoint.deg2rad(np.arange(0.0, 86.0, 1.0))
        mesh_az, mesh_el = np.meshgrid(az_range, el_range)
        self.az = mesh_az.ravel()
        self.el = mesh_el.ravel()
        # Generate random parameter values with this spread
        self.param_stdev = katpoint.deg2rad(20. / 60.)

    def test_pointing_closure(self):
        """Test closure between pointing correction and its reverse operation."""
        # Generate random pointing model
        params = self.param_stdev * np.random.randn(katpoint.PointingModel.num_params)
        pm = katpoint.PointingModel(params)
        self.assertRaises(ValueError, katpoint.PointingModel, params[:-1])
        print repr(pm), pm
        # Test closure on (az, el) grid
        pointed_az, pointed_el = pm.apply(self.az, self.el)
        az, el = pm.reverse(pointed_az, pointed_el)
        assert_angles_almost_equal(az, self.az, decimal=7, err_msg='Azimuth closure error for params=%s' % (params,))
        assert_angles_almost_equal(el, self.el, decimal=7, err_msg='Elevation closure error for params=%s' % (params,))

    def test_pointing_fit(self):
        """Test fitting of pointing model."""
        # Generate random pointing model and corresponding offsets on (az, el) grid
        params = self.param_stdev * np.random.randn(katpoint.PointingModel.num_params)
        params[1] = params[9] = 0.0
        pm = katpoint.PointingModel(params.copy())
        delta_az, delta_el = pm.offset(self.az, self.el)
        enabled_params = (np.arange(katpoint.PointingModel.num_params) + 1).tolist()
        enabled_params.remove(2)
        enabled_params.remove(10)
        # pylint: disable-msg=W0612
        fitted_params, sigma_params = pm.fit(self.az, self.el, delta_az, delta_el, enabled_params=enabled_params)
        np.testing.assert_almost_equal(fitted_params, params, decimal=9)