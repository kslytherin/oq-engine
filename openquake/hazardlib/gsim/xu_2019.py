# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2023 GEM Foundation
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
Module exports :class:`Xu2019Shallow`
               :class:`Xu2019Deep`
"""
import numpy as np
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import CAV


def get_Svals(ctx):
    a_ind = np.digitize(ctx.vs30, [0, 180, 360, 760, 1500])

    SB = np.zeros_like(ctx.vs30)
    SB[a_ind == 4] = 1

    SC = np.zeros_like(ctx.vs30)
    SC[a_ind == 3] = 1

    SD = np.zeros_like(ctx.vs30)
    SD[a_ind == 2] = 1

    SE = np.zeros_like(ctx.vs30)
    SE[a_ind == 1] = 1

    return SB, SC, SD, SE


def get_mean_values(C, ctx):
    """
    Returns the mean values for a specific IMT
    """
    SB, SC, SD, SE = get_Svals(ctx)
    hyp_dist = np.sqrt(ctx.repi**2 + ctx.hypo_depth**2)
    # breakpoint()
    return (
        C["c1"]
        + C["c2"] * (8.5 - ctx.mag) ** 2
        + (C["c3"] + C["c4"] * ctx.mag) * np.log(hyp_dist)
        + C["c5"] * np.log(ctx.vs30)
        + C["c6"] * SB
        + C["c7"] * SC
        + C["c8"] * SD
        + C["c9"] * SE
    )


class Xu2019Shallow(GMPE):
    """
    Implements TSMIP GMPE developed by Yun Xu and others,
    published as "Prediction models and seismic hazard assesment: A case study from Taiwan"
    (2019, Soil Dynamics and Earthquake Engineering, Volume 122, pages 94 - 106).
    """

    #: What tectonic region type?
    # const.TRT.SUBDUCTION_INTERFACE
    # const.TRT.SUBDUCTION_INTRASLAB
    # const.TRT.INDUCED
    # const.TRT.STABLE_CONTINENTAL
    # const.TRT.UPPER_MANTLE
    # const.TRT.VOLCANIC
    # const.TRT.ACTIVE_SHALLOW_CRUST
    # const.IMC.VERTICAL

    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {CAV}

    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN

    #: Supported standard deviation types are inter-event, intra-event
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT,
    }

    #: Required site parameters are Vs30, Vs30 type (measured or inferred),
    REQUIRES_SITES_PARAMETERS = {"vs30"}

    #: Required rupture parameters are
    REQUIRES_RUPTURE_PARAMETERS = {"mag", "hypo_depth"}

    #: Required distance measures are
    REQUIRES_DISTANCES = {"repi"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """

        # C_PGA = self.COEFFS[PGA()]
        # Get mean and standard deviation
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]
            # Get mean and standard deviations for IMT
            mean[m] = get_mean_values(C, ctx)
            sig[m] = np.sqrt(C["sigma"] ** 2 + C["tau"] ** 2)
            tau[m] = C["tau"]
            phi[m] = C["sigma"]

    COEFFS = CoeffsTable(
        table="""\
    IMT        c1        c2       c3      c4       c5      c6      c7      c8      c9     tau   sigma 
    cav     1.153    -0.117   -1.565   0.127   -0.114   0.465   0.978   1.245   1.465   0.335   0.475
    """,
    )


class Xu2019Deep(Xu2019Shallow):
    """
    Implements GMPE for deep earthquakes (> 30 km depth)
    """

    # We are not sure what this is suppose to be, it could be slab or interface
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.UPPER_MANTLE
    COEFFS = CoeffsTable(
        table="""\
    IMT        c1       c2       c3      c4       c5      c6      c7      c8      c9     tau   sigma 
    cav     0.974    0.064   -2.873   0.309   -0.208   1.087   1.485   1.542   1.467   0.187   0.485
    """,
    )
