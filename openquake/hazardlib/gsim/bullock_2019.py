# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2018 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`BullockEtAl2019Asc`
               :class:`BullockEtAl2019SInter`
               :class:`BullockEtAl2019SSlab`
               :class:`BullockEtAl2019AscRotD100`
               :class:`BullockEtAl2019SInterRotD100`
               :class:`BullockEtAl2019SSlabRotD100`
"""
import math

import numpy as np
from scipy.special import erf
import pandas as pd

from openquake.baselib.general import CallableDict
from openquake.hazardlib import const
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable, add_alias
from openquake.hazardlib.imt import CAV, IA

# from B14
def _mu_z1pt0(vs30):
    myexp = 4
    fro = -2.98/4
    a1 = 237
    a2 = 1428
    mu = np.exp(fro*np.log((vs30**myexp+a1**myexp)/(a2**myexp + a1**myexp)))
    return mu

def _get_site_term(C, ctx):
    """
    Returns the linear site scaling term (equation 6)
    """
    fsoil = C["c1"] * np.log(ctx.vs30)
    fsed = C["c2"] * (ctx.z1pt0-_mu_z1pt0(ctx.vs30))
    fsite = fsoil + fsed
    return fsite

# from CB2014 
def _get_Rtvz(R):
    Rtvz = 0.2 * R
    return Rtvz

# from CB2014 
def _path_term(C, ctx):
    R = ctx.rjb
    fdist_log = (C['b1'] + C['b2']*ctx.mag) * np.log(np.sqrt(R ** 2 + C['h'] ** 2))
    fdist_lin = C['b3'] * R
    ftvz = C['b4'] * _get_Rtvz(R)/R
    #breakpoint()
    fpath = fdist_log + fdist_lin + ftvz
    return fpath

# source terms
def _source_term(C, ctx):
    fmag = C['a0'] + C['a1']*ctx.mag + C['a2']*np.log(ctx.mag)
    fdep = C['a3'] * ctx.ztor
    fmech = _get_mechanism(ctx, C)

    fsource = fmag + fdep + fmech
    return fsource

def _get_mechanism(ctx, C):
    """
    Compute the fourth term of the equation described on p. 199:

    ``f1 * Fn + f2 * Fr``
    """
    Fn, Fr = _get_fault_type_dummy_variables(ctx)
    return (C['a4'] * Fn) + (C['a5'] * Fr)

def _get_fault_type_dummy_variables(ctx):
    """
    The original classification considers four style of faulting categories
    (normal, strike-slip, reverse-oblique and reverse).
    """
    Fn, Fr = np.zeros_like(ctx.rake), np.zeros_like(ctx.rake)
    Fn[(ctx.rake >= -30) & (ctx.rake <= -150)] = 1.  # normal
    Fr[(ctx.rake >= 30) & (ctx.rake <= 150)] = 1.
    # joins both the reverse and reverse-oblique categories
    return Fn, Fr

def get_stddevs(C, ctx):
    """
    Returns the standard deviations.
    Generate tau, phi, and total sigma computed
    """
    sig_tot = np.sqrt(C["tau"] ** 2 + C["phi"]** 2)

    return [sig_tot, C["tau"], C["phi"]]


class BullockEtAl2019Asc(GMPE):
    """
    Implements Bullock et al. (2019).
    """

    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST
    #: Supported intensity measure types are spectral acceleration,
    #: peak ground acceleration and peak ground velocity
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {CAV, IA}

    #: Supported intensity measure component is from RotD50 
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD50

    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    #: Site amplification is dependent only upon Vs30
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    #: Required rupture parameters are only magnitude for the interface model
    REQUIRES_RUPTURE_PARAMETERS = {'mag'}

    #: Required distance measure is closest distance to rupture, for
    #: interface events
    REQUIRES_DISTANCES = {'rjb'}
    REQUIRES_ATTRIBUTES = {'region'}
    def __init__(self, region=None, saturation_region=None, basin=None,
                 **kwargs):
        """
        Enable setting regions to prevent messy overriding
        and code duplication.
        """
        super().__init__(region=region, **kwargs)

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """

        # to get name of the class
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]
            fpath = _path_term(C, ctx)
            fsource = _source_term(C, ctx)
            fsite = _get_site_term(C, ctx)

            mean[m] = fpath + fsource + fsite
            sig[m], tau[m], phi[m] = get_stddevs(C, ctx)
            #breakpoint()
            print("check bullock_2019.py")


    # BullockEtAl2019Asc
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5         b1          b2         b3         b4        h        c1           c2      tau       phi
    ia       -6.263     0.342      10.369      0.005 	 -0.977	    0.455     -3.314       0.156        0        0    10.37    -0.765     -0.00004    0.740     1.069
    cav       0.363     0.336       4.822      0.002 	 -0.412	    0.240     -1.394       0.079        0        0    10.18    -0.428      0.00020    0.312     0.497
    """,
    )

class BullockEtAl2019SInter(BullockEtAl2019Asc):
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTERFACE
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5         b1           b2         b3          b4        h        c1           c2      tau       phi
    ia       18.998     2.123      12.112      0.010 	  0.155	      0     -1.126       -0.256        0    -2.43450    35.94    -0.640     -0.00043    0.760     1.080
    cav      -2.839     1.216       3.929      0.004      0.092       0     -0.748       -0.048        0    -0.28556    35.93    -0.383     -0.00003    0.345     0.542
    """,
    )

class BullockEtAl2019SSlab(BullockEtAl2019Asc):
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT           a0        a1          a2         a3         a4        a5         b1           b2         b3          b4        h        c1           c2      tau       phi
    ia       -22.907       0      22.219        0 	     0	 0.054     -0.432       -0.271        0    -2.77890    13.41    -1.344     -0.00049    0.664     0.881
    cav       -8.549       0      12.308        0        0     0.026     -0.102       -0.140        0     0.61950    19.53    -0.777      0.00007    0.321     0.428 
    """,
    )


class BullockEtAl2019AscRotD100(BullockEtAl2019Asc):
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST
    COEFFS = CoeffsTable(
        table="""\
    IMT           a0        a1          a2         a3         a4        a5         b1           b2         b3          b4        h        c1           c2      tau       phi
    ia        -4.410       0.585     8.888      0.002 	  -0.961	 0.433     -3.320        0.157        0         0    10.41    -0.764     -0.00004    0.742     1.071
    cav       -1.505	  -0.367	 8.402	   -0.006     -0.440	 0.213	   -1.385	     0.080	      0	      0	  8.59	  -0.422	  0.00018	 0.310     0.498
    """,
    )

class BullockEtAl2019SInterRotD100(BullockEtAl2019Asc):
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTERFACE
    COEFFS = CoeffsTable(
        table="""\
    IMT           a0         a1          a2         a3         a4        a5         b1           b2         b3           b4        h        c1           c2      tau       phi
    ia       -17.647      2.391      10.840      0.010 	    0.156	    0     -1.107       -0.259        0     -1.97490    35.68    -0.639     -0.00043    0.770     1.082
    cav       -2.518	  1.234	      3.881	     0.004      0.092	    0	    -0.737	     -0.050	       0	   -0.30976    35.86	-0.383	   -0.00003	   0.345     0.542
    """,
    )

class BullockEtAl2019SSlabRotD100(BullockEtAl2019Asc):
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB
    COEFFS = CoeffsTable(
        table="""\
    IMT           a0         a1          a2         a3         a4        a5         b1           b2           b3             b4        h        c1           c2      tau       phi
    ia       -22.147        0      22.183        0 	      0     0.069     -0.450       -0.269          0       -2.36843    13.64    -1.342     -0.00050    0.659     0.883
    cav       -7.959	    0	     12.210	       0        0	  0.034	    -0.145	     -0.137	         0        0.80036    20.69	-0.773	    0.00005	   0.321     0.428
    """,
    )
