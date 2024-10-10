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
Module exports :class:`BullockEtAl2021V1`
               :class:`BullockEtAl2021V2`
               :class:`BullockEtAl2021V3`
               :class:`BullockEtAl2021V4`
               :class:`BullockEtAl2021V5`
               :class:`BullockEtAl2021V6`
               :class:`BullockEtAl2021V7`
               :class:`BullockEtAl2021V8`
               :class:`BullockEtAl2021V9`
               :class:`BullockEtAl2021V1RotD100`
               :class:`BullockEtAl2021V2RotD100`
               :class:`BullockEtAl2021V3RotD100`
               :class:`BullockEtAl2021V4RotD100`
               :class:`BullockEtAl2021V5RotD100`
               :class:`BullockEtAl2021V6RotD100`
               :class:`BullockEtAl2021V7RotD100`
               :class:`BullockEtAl2021V8RotD100`
               :class:`BullockEtAl2021V9RotD100`
"""
import math

import numpy as np
from scipy.special import erf
import pandas as pd

from openquake.baselib.general import CallableDict
from openquake.hazardlib import const
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable, add_alias
from openquake.hazardlib.imt import CAV

CONSTS = {"Vc": 900, "Vref": 760}

# from B14
def _mu_z1pt0(region, vs30):

    if region == 'CA':
        myexp = 4
        fro = -7.15/4
        a1 = 570.94
        a2 = 1360
    if region == 'JP':
        myexp = 2 
        fro = -5.23/2
        a1 = 412.39 
        a2 = 1360
    if region == 'NZ':
        myexp = 4 
        fro = -2.98/4
        a1 = 237 
        a2 = 1428

    mu = 1/1000 * np.exp(fro*np.log((vs30**myexp+a1**myexp)/(a2**myexp + a1**myexp)))

    return mu

def _get_site_term(C, ctx, region):
    """
    Returns the linear site scaling term (equation 6)
    """
    fsite = C["c1"] * np.log(np.fmax(ctx.vs30, CONSTS["Vc"]) / CONSTS["Vref"])
    #fsite[ctx.vs30 > CONSTS["Vc"]] = C["c1"] * CONSTS["Vc"] / CONSTS["Vref"]
    fsed = C["c2"] * np.fmax(ctx.z1pt0-_mu_z1pt0(region,ctx.vs30)-3, np.zeros_like(ctx.z1pt0))
    return fsite + fsed

def _dist_taper(rrup):
    # from CY14,  30 is in CY14 and 20 is in B21
    dist_taper = np.fmax(1 - (np.fmax(rrup - 40,
                              np.zeros_like(rrup)) / 20.), 
                         np.zeros_like(rrup))
    dist_taper = dist_taper.astype(np.float64)
    return dist_taper

def _path_term(C, ctx, version):
    """
    Path term.
    """
    fpath = (C['b1'] + C['b2']*ctx.mag)\
        * np.log(np.sqrt(ctx.rrup ** 2 + C['h'] ** 2))\
        + (C['b3'] + C['b4'] / (
        np.cosh(np.clip(ctx.mag - 5.5, np.zeros_like(ctx.mag), None) * ctx.rrup))) \
        + C['b5'] * _dist_taper(ctx.rrup)
    return fpath 

def _path_term_2(C, ctx, version):
# from B14 #     
    fpath = (C['b1'] + C['b2']*(ctx.mag-4.5)) * np.log(ctx.rrup + C['h'])
    return fpath

# from CB2014 
def _path_term_3(C, ctx, version):
    fpath = (C['b1'] + C['b2']*ctx.mag) * np.log(np.sqrt(ctx.rrup ** 2 + C['h'] ** 2))
    return fpath

# source terms
def _source_term(C, ctx, version):
    fs = C['a0'] + C['a1']*ctx.mag + C['a2']*np.log(ctx.mag) + _fz(ctx,C) + _fdip(ctx,C) + _get_mechanism(ctx, C)
    return fs

def _source_term_2(C, ctx, version):
    fs = C['a0'] + C['a1']*ctx.mag + _fz(ctx,C) + _fdip(ctx,C) + _get_mechanism(ctx, C)
    fs[ctx.mag > 4.5] = fs[ctx.mag > 4.5] + C['a2']*(ctx.mag[ctx.mag > 4.5] - 4.5)
    fs[ctx.mag > 5.5] = fs[ctx.mag > 5.5] + C['a3']*(ctx.mag[ctx.mag > 5.5] - 5.5)
    fs[ctx.mag > 6.5] = fs[ctx.mag > 6.5] + C['a4']*(ctx.mag[ctx.mag > 6.5] - 6.5)
    return fs

def _source_term_3(C, ctx, version):
    fs = C['a0'] + C['a1']*(ctx.mag - 5.5) + _fz(ctx,C) + _fdip(ctx,C) + _get_mechanism(ctx, C)
    fs[ctx.mag <= 5.5] = C['a0'] + C['a1']*(ctx.mag[ctx.mag <= 5.5] - 5.5) + C['a2']*(ctx.mag[ctx.mag <= 5.5] - 5.5)**2 + _fz(ctx,C)[ctx.mag <= 5.5] + _fdip(ctx,C)[ctx.mag <= 5.5] + _get_mechanism(ctx, C)[ctx.mag <= 5.5]

    # ????? should I do a subset of the output from the functions here?
    # if ctx.mag <= 5.5:
    #     fs = C['a0'] + C['a1']*(ctx.mag - 5.5) + C['a2']*(ctx.mag - 5.5)**2 + _fz(ctx,C) + _fdip(ctx,C) + _get_mechanism(ctx, C)
    # else:
    #     fs = C['a0'] + C['a1']*(ctx.mag - 5.5) + _fz(ctx,C) + _fdip(ctx,C) + _get_mechanism(ctx, C)

    return fs

def _fz(ctx,C):
    FZ = _fzz(ctx) * _fzm(ctx, C) 
    return FZ

def _fzz(ctx):
    FZZ = np.full_like(ctx.ztor, 10)
    FZZ[ctx.ztor <= 15] = ctx.ztor[ctx.ztor <= 15]-5
    return FZZ

def _fzm(ctx,C):
    FZM = np.full_like(ctx.ztor,C['a6'])
    FZM[ctx.mag <= 5.5] = C['a5']
    FZM[ctx.mag <= 6.5] = C['a5'] + (C['a6']-C['a5'])*(ctx.mag[ctx.mag <= 6.5] - 5.5)
    return FZM

def _fdip(ctx,C):
    fdip = np.zeros_like(ctx.mag)
    fdip[ctx.mag <= 4.5] = C['a7']*ctx.dip[ctx.mag <= 4.5]
    fdip[ctx.mag <= 5.5] = C['a7']*(5.5-ctx.mag[ctx.mag <= 5.5])*ctx.dip[ctx.mag <= 5.5]
    return fdip 

def _get_mechanism(ctx, C):
    """
    Compute the fourth term of the equation described on p. 199:

    ``f1 * Fn + f2 * Fr``
    """
    Fn, Fr = _get_fault_type_dummy_variables(ctx)
    return (C['a8'] * Fr) + (C['a9'] * Fn)

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

def _get_sig_e(mag, rrup):
    # Note: Within Sig_e has to be < 0.085, for Between Sig_e to be effective 

        # 0.1 km,0.3 km,0.5 km,0.8 km,1 km,2 km,5 km,8 km,10 km,15 km,20 km,30 km,40 km,50 km,75 km,100 km,125 km,150 km,200 km,300 km
    sig_e_set =np.array([
        [0.701,0.685,0.669,0.646,0.631,0.562,0.399,0.28,0.218,0.112,0.084,0.132,0.158,0.108,0.043,0.058,0.113,0.174,0.3,0.549],
        [0.691,0.675,0.659,0.637,0.622,0.554,0.392,0.275,0.215,0.116,0.095,0.14,0.164,0.111,0.041,0.057,0.113,0.176,0.303,0.554],
        [0.681,0.665,0.65,0.627,0.613,0.545,0.386,0.271,0.212,0.12,0.104,0.147,0.168,0.113,0.039,0.056,0.115,0.179,0.307,0.56],
        [0.671,0.655,0.64,0.618,0.603,0.537,0.38,0.267,0.21,0.124,0.113,0.153,0.171,0.114,0.036,0.058,0.118,0.183,0.313,0.568],
        [0.66,0.645,0.63,0.608,0.594,0.528,0.374,0.263,0.208,0.128,0.12,0.157,0.173,0.114,0.035,0.061,0.123,0.189,0.321,0.577],
        [0.65,0.635,0.62,0.598,0.585,0.52,0.368,0.26,0.207,0.133,0.126,0.161,0.174,0.113,0.036,0.067,0.13,0.197,0.329,0.588],
        [0.637,0.622,0.608,0.587,0.573,0.51,0.36,0.255,0.204,0.136,0.132,0.165,0.177,0.114,0.036,0.07,0.134,0.202,0.336,0.597],
        [0.625,0.61,0.596,0.575,0.561,0.499,0.352,0.25,0.202,0.139,0.138,0.169,0.178,0.114,0.038,0.075,0.141,0.209,0.345,0.607],
        [0.612,0.597,0.583,0.563,0.55,0.489,0.345,0.246,0.199,0.142,0.142,0.171,0.178,0.113,0.042,0.082,0.148,0.217,0.354,0.619],
        [0.598,0.584,0.57,0.55,0.538,0.478,0.337,0.241,0.196,0.144,0.145,0.172,0.177,0.111,0.046,0.09,0.157,0.227,0.365,0.632],
        [0.585,0.571,0.557,0.538,0.525,0.467,0.329,0.236,0.193,0.145,0.147,0.172,0.175,0.109,0.052,0.099,0.167,0.238,0.377,0.646],
        [0.571,0.557,0.544,0.525,0.512,0.455,0.321,0.231,0.19,0.145,0.148,0.171,0.172,0.105,0.059,0.109,0.179,0.25,0.391,0.662],
        [0.556,0.543,0.53,0.512,0.499,0.443,0.313,0.225,0.187,0.145,0.149,0.169,0.169,0.102,0.067,0.121,0.191,0.264,0.406,0.678],
        [0.542,0.529,0.516,0.498,0.486,0.431,0.304,0.22,0.183,0.144,0.148,0.167,0.164,0.099,0.078,0.135,0.206,0.279,0.422,0.696],
        [0.527,0.514,0.502,0.484,0.473,0.419,0.296,0.214,0.179,0.143,0.148,0.164,0.16,0.097,0.092,0.15,0.222,0.295,0.439,0.716],
        [0.512,0.5,0.488,0.47,0.459,0.407,0.287,0.209,0.176,0.143,0.148,0.162,0.156,0.097,0.108,0.168,0.24,0.314,0.459,0.737],
        [0.489,0.477,0.465,0.448,0.437,0.387,0.27,0.192,0.159,0.124,0.129,0.143,0.137,0.086,0.127,0.192,0.265,0.338,0.482,0.757],
        [0.466,0.454,0.443,0.427,0.416,0.367,0.253,0.177,0.143,0.108,0.113,0.128,0.122,0.083,0.143,0.207,0.277,0.347,0.485,0.745],
        [0.443,0.432,0.421,0.405,0.395,0.347,0.237,0.162,0.128,0.094,0.1,0.117,0.113,0.085,0.153,0.212,0.277,0.342,0.468,0.704],
        [0.421,0.41,0.399,0.384,0.374,0.328,0.221,0.148,0.115,0.081,0.089,0.109,0.107,0.089,0.157,0.21,0.267,0.324,0.434,0.637],
        [0.398,0.388,0.378,0.363,0.354,0.309,0.206,0.136,0.104,0.071,0.082,0.103,0.103,0.093,0.158,0.201,0.249,0.296,0.386,0.548],
        [0.376,0.366,0.357,0.342,0.333,0.291,0.192,0.124,0.094,0.064,0.077,0.1,0.102,0.096,0.154,0.187,0.224,0.26,0.327,0.445],
        [0.355,0.345,0.336,0.322,0.314,0.273,0.179,0.114,0.085,0.059,0.074,0.099,0.102,0.099,0.149,0.17,0.195,0.218,0.262,0.331],
        [0.334,0.325,0.316,0.303,0.295,0.256,0.167,0.106,0.079,0.058,0.074,0.099,0.103,0.102,0.142,0.152,0.164,0.175,0.194,0.217],
        [0.313,0.305,0.296,0.284,0.276,0.24,0.156,0.1,0.076,0.059,0.075,0.099,0.104,0.105,0.136,0.134,0.133,0.132,0.13,0.125],
        [0.294,0.285,0.278,0.266,0.259,0.225,0.147,0.096,0.075,0.064,0.079,0.101,0.105,0.108,0.13,0.117,0.105,0.095,0.083,0.129],
        [0.273,0.265,0.258,0.247,0.24,0.208,0.135,0.088,0.069,0.06,0.074,0.094,0.098,0.104,0.12,0.097,0.074,0.059,0.077,0.215],
        [0.254,0.247,0.24,0.23,0.223,0.194,0.127,0.086,0.07,0.062,0.074,0.091,0.093,0.103,0.113,0.078,0.048,0.044,0.118,0.319],
        [0.237,0.231,0.225,0.215,0.21,0.183,0.124,0.09,0.077,0.072,0.08,0.091,0.09,0.104,0.108,0.063,0.03,0.058,0.17,0.422],
        [0.224,0.218,0.213,0.204,0.199,0.176,0.126,0.099,0.09,0.086,0.091,0.096,0.091,0.108,0.106,0.052,0.028,0.084,0.221,0.519],
        [0.214,0.209,0.204,0.197,0.193,0.173,0.133,0.114,0.107,0.103,0.105,0.105,0.096,0.115,0.108,0.046,0.04,0.109,0.267,0.608],
        [0.209,0.205,0.201,0.195,0.191,0.175,0.145,0.131,0.127,0.123,0.123,0.117,0.105,0.126,0.113,0.047,0.053,0.132,0.308,0.688],
        [0.209,0.205,0.202,0.197,0.195,0.182,0.16,0.151,0.148,0.145,0.142,0.133,0.117,0.139,0.122,0.052,0.065,0.15,0.343,0.759],
        [0.213,0.211,0.208,0.205,0.203,0.194,0.179,0.173,0.171,0.167,0.164,0.151,0.133,0.155,0.135,0.061,0.076,0.165,0.373,0.822],
        [0.223,0.221,0.219,0.217,0.215,0.209,0.2,0.196,0.194,0.191,0.187,0.172,0.151,0.174,0.15,0.074,0.084,0.177,0.397,0.876],
        [0.236,0.235,0.234,0.232,0.231,0.227,0.222,0.22,0.219,0.216,0.211,0.195,0.172,0.195,0.167,0.089,0.092,0.185,0.416,0.921],
        [0.254,0.253,0.252,0.251,0.251,0.248,0.246,0.245,0.245,0.241,0.236,0.219,0.195,0.218,0.188,0.106,0.099,0.19,0.43,0.96],
        [0.274,0.274,0.273,0.273,0.273,0.272,0.271,0.271,0.271,0.268,0.262,0.244,0.22,0.242,0.21,0.125,0.106,0.193,0.44,0.99],
        [0.297,0.297,0.297,0.297,0.297,0.297,0.297,0.298,0.298,0.295,0.289,0.271,0.246,0.269,0.234,0.146,0.116,0.194,0.445,1.015],
        [0.323,0.323,0.323,0.323,0.323,0.323,0.324,0.325,0.325,0.322,0.317,0.299,0.274,0.296,0.26,0.169,0.127,0.194,0.447,1.033],
        [0.35,0.35,0.35,0.35,0.35,0.35,0.352,0.353,0.353,0.351,0.346,0.328,0.303,0.325,0.288,0.193,0.141,0.194,0.446,1.046],
        [0.378,0.378,0.378,0.378,0.379,0.379,0.381,0.381,0.382,0.38,0.375,0.358,0.333,0.356,0.317,0.22,0.158,0.195,0.442,1.053],
        [0.408,0.408,0.408,0.408,0.408,0.408,0.41,0.41,0.411,0.409,0.405,0.389,0.364,0.387,0.348,0.249,0.177,0.198,0.435,1.056],
        [0.439,0.439,0.439,0.439,0.439,0.439,0.439,0.44,0.44,0.439,0.436,0.42,0.397,0.42,0.38,0.279,0.2,0.203,0.428,1.055],
        [0.47,0.47,0.47,0.47,0.47,0.47,0.469,0.47,0.47,0.47,0.467,0.453,0.43,0.453,0.414,0.31,0.225,0.211,0.418,1.05],
        [0.503,0.503,0.502,0.502,0.502,0.501,0.5,0.5,0.501,0.501,0.499,0.486,0.464,0.488,0.448,0.343,0.252,0.222,0.409,1.042],
    ])

    rrup_dtpts = np.array([0.1, 0.3, 0.5, 0.8, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200, 300])
    pre_mag_dtpts = np.arange(4,8.6,0.1)

    # sig_e_set = np.array([ 
    #     [0.701, 0.631, 0.399, 0.218, 0.084, 0.132, 0.108, 0.058, 0.300, 0.549],
    #     [0.650, 0.585, 0.368, 0.207, 0.126, 0.161, 0.113, 0.067, 0.329, 0.588],
    #     [0.585, 0.525, 0.329, 0.193, 0.147, 0.172, 0.109, 0.099, 0.377, 0.646],
    #     [0.512, 0.459, 0.287, 0.176, 0.148, 0.162, 0.097, 0.168, 0.459, 0.737],
    #     [0.398, 0.354, 0.206, 0.104, 0.082, 0.103, 0.093, 0.201, 0.386, 0.548],
    #     [0.344, 0.304, 0.173, 0.082, 0.074, 0.099, 0.101, 0.161, 0.228, 0.274],
    #     [0.294, 0.259, 0.147, 0.075, 0.079, 0.101, 0.108, 0.117, 0.083, 0.129],
    #     [0.245, 0.216, 0.125, 0.073, 0.077, 0.091, 0.103, 0.070, 0.144, 0.371],
    #     [0.214, 0.193, 0.133, 0.107, 0.105, 0.105, 0.115, 0.046, 0.267, 0.608],
    #     [0.210, 0.198, 0.169, 0.159, 0.153, 0.142, 0.147, 0.056, 0.359, 0.792],
    #     [0.236, 0.231, 0.222, 0.219, 0.211, 0.195, 0.195, 0.089, 0.416, 0.921],
    #     [0.286, 0.284, 0.284, 0.284, 0.275, 0.257, 0.255, 0.135, 0.443, 1.003],
    #     [0.350, 0.350, 0.352, 0.353, 0.346, 0.328, 0.325, 0.193, 0.446, 1.046]
    #     ])
    # rrup_dtpts = np.array([0.1, 1, 5, 10, 20, 30, 50, 100, 200, 300])
    # pre_mag_dtpts = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8])

    mag_dtpts = np.round(pre_mag_dtpts*10)/10
    sig_e_dtpts = pd.DataFrame(sig_e_set, index=mag_dtpts, columns=rrup_dtpts)
    
    sig_e = np.interp(np.log(rrup), np.log(rrup_dtpts), sig_e_dtpts.loc[mag[0],:])

    return sig_e

def get_stddevs(C, ctx):
    """
    Returns the standard deviations.
    Generate tau, phi, and total sigma computed
    """
    phi_s = C["phi1"] + C["phi2"] * np.full_like(ctx.rrup,30)
    phi_s[ctx.rrup <= 30] = C["phi1"] + C["phi2"] * ctx.rrup[ctx.rrup <= 30]
    phi_s[100 < ctx.rrup] = C["phi1"] + C["phi2"] * 30 + C["phi3"]*(ctx.rrup[100 < ctx.rrup]-100)
    phi = phi_s

    sig_e = _get_sig_e(ctx.mag,ctx.rrup)

    sig_tot = np.sqrt(C["tau"] ** 2 + phi ** 2 + sig_e **2)
    #sig_tot = np.sqrt(C["tau"] ** 2 + phi ** 2)

    return [sig_tot, C["tau"], phi]


class BullockEtAl2021V1(GMPE):
    """
    Implements Bullock et al. (2021) for subduction interface.
    """

    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are spectral acceleration,
    #: peak ground acceleration and peak ground velocity
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {CAV}

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
    REQUIRES_DISTANCES = {'rrup'}
    REQUIRES_ATTRIBUTES = {'region'}
    def __init__(self, region=None, saturation_region=None, basin=None,
                 **kwargs):
        """
        Enable setting regions to prevent messy overriding
        and code duplication.
        """
        super().__init__(region=region, **kwargs)
        self.region = region

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        version = int(self.__class__.__name__[16])

        # to get name of the class
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]
            if version in [1, 2, 3]:
                fpath = _path_term(C, ctx, version)
            elif version in [4, 5, 6]:
                fpath = _path_term_2(C, ctx, version)
            elif version in [7, 8, 9]:
                fpath = _path_term_3(C, ctx, version)

            if version in [1, 4, 7]:
                fsource = _source_term(C, ctx, version)
            elif version in [2, 5, 8]:
                fsource = _source_term_2(C, ctx, version)
            elif version in [3, 6, 9]:
                fsource = _source_term_3(C, ctx, version)

            fsite = _get_site_term(C, ctx, self.region)

            mean[m] = fpath + fsource + fsite
            sig[m], tau[m], phi[m] = get_stddevs(C, ctx)
            # breakpoint()


    # BullockEtAl2021V1
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5        a6         a7         a8         a9        b1          b2         b3         b4        b5           h        c1          c2     tau       phi     phi1     phi2     phi3
    cav       5.6336  -0.4233      2.7951          0	      0	   0.0408   -0.0018     0.0027    -0.0973    -0.0571   -3.5332      0.4365    -0.0178     0.0224   -0.2455          13	 -0.7644      0.0403  0.2695    0.5489   0.5699  -0.0028   0.0018
    """,
    )

    #cav 5.63356558094253	-0.423345012684359	2.79513809663422	0.0	0.0	0.0408423435069524	-0.00175195184961421	0.0026654137351502	-0.0972506805827566	-0.0571105739918207	-3.53323211889372	0.436457951978744	-0.0177516258387534	0.0224454998067286	-0.245504113897333	13	-0.764431352457597	0.0403040453381035	0.269541069826975	0.54888554069351	0.569913654984463	-0.00280234139878648	0.00176480730589055



class BullockEtAl2021V2j(BullockEtAl2021V1):
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5        a6         a7         a8         a9        b1          b2         b3         b4        b5           h        c1          c2     tau       phi     phi1     phi2     phi3
    cav      6.6673    0.2748     -0.0519    -0.2752     0.0816    0.0411   -0.0043     0.0031    -0.0958    -0.0481   -3.5332      0.4365    -0.0178     0.0224   -0.2455          13   -0.7644      0.0403  0.2697    0.5503   0.5699  -0.0028   0.0018
    """,
    )

class BullockEtAl2021V3(BullockEtAl2021V1):
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5        a6         a7         a8         a9        b1          b2         b3         b4        b5           h        c1          c2     tau       phi     phi1     phi2     phi3
    cav      8.0955    0.0948     -0.0583        NaN        NaN    0.0392    0.0360     0.0027    -0.0859    -0.0740   -3.5332      0.4365    -0.0178     0.0224   -0.2455          13   -0.7644      0.0403  0.2746    0.5541   0.5701  -0.0028   0.0018
    """,
    )

class BullockEtAl2021V4(BullockEtAl2021V1):
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5        a6         a7         a8         a9        b1          b2         b3         b4        b5           h        c1          c2     tau       phi     phi1     phi2     phi3
    cav      1.3042   -1.7648     10.0866        NaN        NaN    0.0407    0.0256     0.0016    -0.0564    -0.0250   -1.5767      0.3354        NaN        NaN       NaN          10   -0.7390      0.1359  0.2664    0.5827   0.6216  -0.0040   0.0014
    """,
    )

class BullockEtAl2021V5(BullockEtAl2021V1):
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5        a6         a7         a8         a9        b1          b2         b3         b4        b5           h        c1          c2     tau       phi     phi1     phi2     phi3
    cav      5.0158    0.7512     -0.2320    -0.7386    -0.2131    0.0434    0.0119     0.0037    -0.0605     0.0041   -1.5767      0.3354        NaN        NaN       NaN          10   -0.7390      0.1359  0.2623    0.5795   0.6220  -0.0040   0.0014
    """,
    )

class BullockEtAl2021V6(BullockEtAl2021V1):
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5        a6         a7         a8         a9        b1          b2         b3         b4        b5           h        c1          c2     tau       phi     phi1     phi2     phi3
    cav      8.8040   -0.1108     -0.2972        NaN        NaN    0.0424    0.0383     0.0007    -0.0480    -0.0429   -1.5767      0.3354        NaN        NaN       NaN          10   -0.7390      0.1359  0.2729    0.5844   0.6216  -0.0040   0.0014
    """,
    )

class BullockEtAl2021V7(BullockEtAl2021V1):
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5        a6         a7         a8         a9        b1          b2         b3         b4        b5           h        c1          c2     tau       phi     phi1     phi2     phi3
    cav      2.8943   -1.7208      9.6333        NaN        NaN    0.0353    0.0142     0.0011    -0.0754    -0.0527   -3.3893      0.3465        NaN        NaN       NaN          29   -0.7460      0.1281  0.2678    0.5721   0.5308  -0.0011   0.0014
    """,
    )

class BullockEtAl2021V8(BullockEtAl2021V1):
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5        a6         a7         a8         a9        b1          b2         b3         b4        b5           h        c1          c2     tau       phi     phi1     phi2     phi3
    cav      6.4395    0.6713     -0.1400    -0.8058    -0.2250    0.0385   -0.0041     0.0036    -0.0803    -0.0142   -3.3893      0.3465        NaN        NaN       NaN          29   -0.7460      0.1281  0.2587    0.5686   0.5313  -0.0012   0.0014
    """,
    )

class BullockEtAl2021V9(BullockEtAl2021V1):
    # RotD50
    COEFFS = CoeffsTable(
        table="""\
    IMT          a0        a1          a2         a3         a4        a5        a6         a7         a8         a9        b1          b2         b3         b4        b5           h        c1          c2     tau       phi     phi1     phi2     phi3
    cav      9.8657   -0.1286     -0.2785        NaN        NaN    0.0300    0.0327     0.0003    -0.0670    -0.0721   -3.3893      0.3465        NaN        NaN       NaN          29   -0.7460      0.1281  0.2749    0.5759   0.5305  -0.0011   0.0014
    """,
    )




class BullockEtAl2021V1RotD100(BullockEtAl2021V1):
    # RotD100
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    COEFFS = CoeffsTable(
        table="""\
    IMT             a0         a1        a2            a3        a4         a5        a6        a7         a8          a9          b1       b2         b3         b4         b5    h         c1       c2       tau       phi      phi1     phi2     phi3
    cav         5.8184	  -0.3731	 2.6355	          NaN	    NaN	    0.0407	  0.0094	0.0026	  -0.0950	  -0.0673	  -3.5736	0.4408	  -0.0178	  0.0224	-0.2568	  13	-0.7720	  0.0534	0.2707	  0.5539	0.5861	-0.0032	  0.0017
    """,
    )

class BullockEtAl2021V2RotD100(BullockEtAl2021V1):
    # RotD100
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    COEFFS = CoeffsTable(
        table="""\
    IMT             a0         a1        a2            a3        a4         a5        a6        a7         a8          a9          b1       b2         b3         b4         b5    h         c1       c2       tau       phi      phi1     phi2     phi3
    cav         6.8047	   0.2785	-0.0149	      -0.3004	 0.0851	    0.0412	  0.0058	0.0032	  -0.0935	  -0.0556	  -3.5736	0.4408	  -0.0178	  0.0224	-0.2568	  13	-0.7720	  0.0534	0.2706	  0.5551	0.5860	-0.0032	  0.0017
    """,
    )

class BullockEtAl2021V3RotD100(BullockEtAl2021V1):
    # RotD100
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    COEFFS = CoeffsTable(
        table="""\
    IMT             a0         a1        a2            a3        a4         a5        a6        a7         a8          a9          b1       b2         b3         b4         b5    h         c1       c2       tau       phi      phi1     phi2     phi3
    cav         8.2778	   0.1019	-0.0603	          NaN	    NaN	    0.0405	  0.0371	0.0025	  -0.0863	  -0.0806	  -3.5736	0.4408	  -0.0178	  0.0224	-0.2568	  13	-0.7720	  0.0534	0.2737	  0.5581	0.5862	-0.0032	  0.0017
    """,
    )
    

class BullockEtAl2021V4RotD100(BullockEtAl2021V1):
    # RotD100
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    COEFFS = CoeffsTable(
        table="""\
    IMT             a0         a1        a2            a3        a4         a5        a6        a7         a8          a9          b1       b2         b3         b4         b5    h         c1       c2       tau       phi      phi1     phi2     phi3
    cav         1.4417	  -1.7265	 9.9963	          NaN	    NaN	    0.0407	  0.0370	0.0015	  -0.0555	  -0.0367	  -1.5996	0.3394	      NaN	     NaN	    NaN	  10	-0.7474	  0.1554	0.2661	  0.5869	0.6402	-0.0045	  0.0013
    """,
    )

class BullockEtAl2021V5RotD100(BullockEtAl2021V1):
    # RotD100
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    COEFFS = CoeffsTable(
        table="""\
    IMT             a0         a1        a2            a3        a4         a5        a6        a7         a8          a9          b1       b2         b3         b4         b5    h         c1       c2       tau       phi      phi1     phi2     phi3
    cav         5.1325	   0.7598	-0.1938	      -0.7734	-0.2078	    0.0436	  0.0222	0.0037	  -0.0596	  -0.0044	  -1.5996	0.3394	      NaN	     NaN	    NaN	  10	-0.7474	  0.1554	0.2614	  0.5839	0.6405	-0.0045	  0.0013
    """,
    )

class BullockEtAl2021V6RotD100(BullockEtAl2021V1):
    # RotD100
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    COEFFS = CoeffsTable(
        table="""\
    IMT             a0         a1        a2            a3        a4         a5        a6        a7         a8          a9          b1       b2         b3         b4         b5    h         c1       c2       tau       phi      phi1     phi2     phi3
    cav         8.9915	  -0.1049	-0.3015	          NaN	    NaN	    0.0444	  0.0395	0.0005	  -0.0496	  -0.0510	  -1.5996	0.3394	      NaN	     NaN	    NaN	  10	-0.7474	  0.1554	0.2728	  0.5871	0.6400	-0.0045	  0.0013
    """,
    )

class BullockEtAl2021V7RotD100(BullockEtAl2021V1):
    # RotD100
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    COEFFS = CoeffsTable(
        table="""\
    IMT             a0         a1        a2            a3        a4         a5        a6        a7         a8          a9          b1       b2         b3         b4         b5    h         c1       c2       tau       phi      phi1     phi2     phi3
    cav         3.0596	  -1.6812	 9.5315	          NaN	    NaN	    0.0352	  0.0254	0.0010	  -0.0749	  -0.0648	  -3.4350	0.3507	      NaN	     NaN	    NaN	  29	-0.7545	  0.1476	0.2671	  0.5762	0.5485	-0.0016	  0.0012
    """,
    )

class BullockEtAl2021V8RotD100(BullockEtAl2021V1):
    # RotD100
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    COEFFS = CoeffsTable(
        table="""\
    IMT             a0         a1        a2            a3        a4         a5        a6        a7         a8          a9          b1       b2         b3         b4         b5    h         c1       c2       tau       phi      phi1     phi2     phi3
    cav         6.5796	   0.6783	-0.1007	      -0.8407	  -0.22	    0.0386	  0.0059	0.0037	  -0.0797	  -0.0230	  -3.4350	0.3507	      NaN	     NaN	    NaN	  29	-0.7545	  0.1476	0.2572	  0.5729	0.5489	-0.0016	  0.0012
    """,
    )

class BullockEtAl2021V9RotD100(BullockEtAl2021V1):
    # RotD100
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD100
    COEFFS = CoeffsTable(
        table="""\
    IMT             a0         a1        a2            a3        a4         a5        a6        a7         a8          a9         b1        b2         b3         b4         b5    h         c1       c2       tau       phi      phi1     phi2     phi3 
    cav        10.0689	  -0.1228	-0.2823	          NAN	    NaN	    0.0317	  0.0338	0.0001	  -0.0689	  -0.0806	  -3.4350	0.3507	      NaN	     NaN	    NaN	  29	-0.7545	  0.1476	0.2739	  0.5787	0.5482	-0.0016	  0.0012
    """,
    )
