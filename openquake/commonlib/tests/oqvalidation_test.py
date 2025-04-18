# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
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

import os
import unittest.mock as mock
import unittest
import tempfile
import pytest
from openquake.qa_tests_data import event_based
from openquake.baselib.general import gettemp
from openquake.hazardlib import InvalidFile
from openquake.commonlib import readinput
from openquake.commonlib.oqvalidation import OqParam

TMP = tempfile.gettempdir()

GST = {'gsim_logic_tree': gettemp('''\
<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">
    <logicTree logicTreeID='lt1'>
        <logicTreeBranchingLevel branchingLevelID="bl1">
            <logicTreeBranchSet uncertaintyType="gmpeModel" branchSetID="bs1"
                    applyToTectonicRegionType="Active Shallow Crust">
                <logicTreeBranch branchID="b1">
                    <uncertaintyModel>SadighEtAl1997</uncertaintyModel>
                    <uncertaintyWeight>1.0</uncertaintyWeight>
                </logicTreeBranch>
            </logicTreeBranchSet>
        </logicTreeBranchingLevel>
    </logicTree>
</nrml>'''),
       "job_ini": "job.ini",
       "source_model_logic_tree": "fake"}

fakeinputs = {"job_ini": "job.ini", "source_model_logic_tree": "fake"}
fakeinputs_risk = {"job_ini": "job.ini", "source_model_logic_tree": "fake",
                   "structural_vulnerability": "fake"}


class OqParamTestCase(unittest.TestCase):

    def test_unknown_parameter(self):
        # if the job.ini file contains an unknown parameter, print a warning
        with mock.patch('logging.warning') as w:
            OqParam(
                calculation_mode='classical', inputs=GST,
                hazard_calculation_id=None, hazard_output_id=None,
                maximum_distance='10', sites='0.1 0.2',
                reference_vs30_value='200',
                truncation_level='3',
                not_existing_param='XXX', export_dir=TMP,
                intensity_measure_types_and_levels="{'PGA': [0.1, 0.2]}",
                rupture_mesh_spacing='1.5').validate()
        self.assertEqual(
            w.call_args[0][0],
            "The parameter 'not_existing_param' is unknown, ignoring")

    def test_truncation_level_None(self):
        with self.assertRaises(ValueError):
            OqParam(calculation_mode='disaggregation',
                    hazard_calculation_id=None, hazard_output_id=None,
                    inputs=fakeinputs, maximum_distance='10',
                    sites='',
                    intensity_measure_types_and_levels="{'PGA': [0.1, 0.2]}",
                    truncation_level=None).validate()

    def test_region_grid_spacing(self):
        # if there is a region there must be a region_grid_spacing
        with self.assertRaises(ValueError):
            OqParam(
                calculation_mode='classical_risk',
                hazard_calculation_id=None, hazard_output_id=None,
                maximum_distance=10,
                region='-78.182 15.615, -78.152 15.615, -78.152 15.565, '
                '-78.182 15.565', inputs=dict(site_model='')
            ).validate()

    def test_poes(self):
        # if hazard_maps or uniform_hazard_spectra are set, poes
        # cannot be empty
        with self.assertRaises(ValueError):
            OqParam(
                calculation_mode='classical_risk',
                truncation_level='3',
                hazard_calculation_id=None, hazard_output_id=None,
                inputs=fakeinputs_risk, maximum_distance='10', sites='',
                hazard_maps='true',  poes='').validate()
        with self.assertRaises(ValueError):
            OqParam(
                calculation_mode='classical_risk',
                truncation_level='3',
                hazard_calculation_id=None, hazard_output_id=None,
                inputs=fakeinputs_risk, maximum_distance='10', sites='',
                uniform_hazard_spectra='true',  poes='').validate()

    def test_site_model(self):
        # if the site_model_file is missing, reference_vs30_type and
        # the other site model parameters cannot be None
        with self.assertRaises(ValueError):
            OqParam(
                calculation_mode='classical_risk', inputs={},
                maximum_distance='10',
                hazard_calculation_id=None, hazard_output_id=None,
                reference_vs30_type=None).validate()

    def test_missing_maximum_distance(self):
        with self.assertRaises(ValueError):
            OqParam(
                calculation_mode='classical', inputs=fakeinputs,
                truncation_level='3', sites='0.1 0.2').validate()

        oq = OqParam(
            calculation_mode='event_based', inputs=GST,
            intensity_measure_types_and_levels="{'PGA': [0.1, 0.2]}",
            intensity_measure_types='PGV', sites='0.1 0.2',
            reference_vs30_value='200',
            truncation_level='3',
            maximum_distance='{"wrong TRT": 200}')
        oq.inputs['source_model_logic_tree'] = 'something'

        oq._trts = {'Active Shallow Crust'}
        self.assertFalse(oq.is_valid_maximum_distance())
        self.assertIn('setting the maximum_distance for wrong TRT', oq.error)

        oq._trts = {'Active Shallow Crust', 'Stable Continental Crust'}
        oq.maximum_distance = {'Active Shallow Crust': 200}
        self.assertFalse(oq.is_valid_maximum_distance())
        self.assertEqual('missing distance for Stable Continental Crust '
                         'and no default', oq.error)

    def test_imts_and_imtls(self):
        oq = OqParam(
            calculation_mode='event_based', inputs=GST,
            intensity_measure_types_and_levels="{'PGA': [0.1, 0.2]}",
            intensity_measure_types='PGV', sites='0.1 0.2',
            truncation_level='3',
            reference_vs30_value='200',
            maximum_distance='400')
        oq.validate()
        self.assertEqual(list(oq.imtls.keys()), ['PGA'])

    def test_create_export_dir(self):
        # FIXME: apparently this fails only when --with-doctest is set
        raise unittest.SkipTest
        EDIR = os.path.join(TMP, 'nonexisting')
        OqParam(
            calculation_mode='event_based',
            sites='0.1 0.2',
            reference_vs30_value='200',
            intensity_measure_types='PGA', inputs=GST,
            maximum_distance='400',
            export_dir=EDIR,
        ).validate()
        self.assertTrue(os.path.exists(EDIR))

    def test_invalid_export_dir(self):
        # FIXME: apparently this fails only when --with-doctest is set
        raise unittest.SkipTest
        with self.assertRaises(ValueError) as ctx:
            OqParam(
                calculation_mode='event_based', inputs=GST,
                sites='0.1 0.2',
                maximum_distance='400',
                reference_vs30_value='200',
                intensity_measure_types='PGA',
                export_dir='/non/existing',
            ).validate()
        self.assertIn('The `export_dir` parameter must refer to a '
                      'directory', str(ctx.exception))

    def test_invalid_imt(self):
        imt = 'PGD'
        imtls = '{"%s": [0.4, 0.5, 0.6]}' % imt
        with self.assertRaises(ValueError) as ctx:
            OqParam(
                calculation_mode='event_based', inputs=fakeinputs,
                sites='0.1 0.2',
                maximum_distance='400',
                truncation_level='3',
                ground_motion_correlation_model='JB2009',
                intensity_measure_types_and_levels=imtls,
            ).validate()
        self.assertEqual(
            str(ctx.exception),
            f'Correlation model JB2009 does not accept IMT={imt}')

    def test_duplicated_levels(self):
        with self.assertRaises(ValueError) as ctx:
            OqParam(
                calculation_mode='classical', inputs={},
                sites='0.1 0.2',
                reference_vs30_type='measured',
                reference_vs30_value='200',
                reference_depth_to_2pt5km_per_sec='100',
                reference_depth_to_1pt0km_per_sec='150',
                maximum_distance='400',
                intensity_measure_types_and_levels='{"PGA": [0.4, 0.4, 0.6]}',
            ).validate()
        self.assertEqual(
            str(ctx.exception),
            'Found duplicated levels for PGA: [0.4, 0.4, 0.6]: could not '
            'convert to intensity_measure_types_and_levels: '
            'intensity_measure_types_and_levels={"PGA": [0.4, 0.4, 0.6]}')

    def test_missing_levels_hazard(self):
        with self.assertRaises(ValueError) as ctx:
            OqParam(
                calculation_mode='classical', inputs=fakeinputs,
                sites='0.1 0.2',
                maximum_distance='400',
                truncation_level='3',
                intensity_measure_types='PGA',
            ).validate()
        self.assertIn('`intensity_measure_types_and_levels`',
                      str(ctx.exception))

    def test_missing_levels_event_based(self):
        with self.assertRaises(ValueError) as ctx:
            oq = OqParam(
                calculation_mode='event_based', inputs=fakeinputs,
                sites='0.1 0.2',
                maximum_distance='400',
                truncation_level='3',
                intensity_measure_types='PGA',
                hazard_curves_from_gmfs='true')
            oq.validate()
        self.assertIn('`intensity_measure_types_and_levels`',
                      str(ctx.exception))

    def test_ambiguous_gsim(self):
        with self.assertRaises(InvalidFile) as ctx:
            OqParam(
                calculation_mode='scenario', inputs=GST,
                gsim='AbrahamsonEtAl2014',
                sites='0.1 0.2',
                maximum_distance='400',
                intensity_measure_types='PGA',
            ).validate()
        self.assertIn('there must be no `gsim` key', str(ctx.exception))

    def test_not_accepted_IMT(self):
        with self.assertRaises(ValueError) as ctx:
            OqParam(
                calculation_mode='scenario',
                gsim='ToroEtAl2002',
                sites='0.1 0.2',
                maximum_distance='400',
                intensity_measure_types='PGV',
                inputs=fakeinputs,
            ).validate()
        self.assertIn('The IMT PGV is not accepted by the GSIM [ToroEtAl2002]',
                      str(ctx.exception))

    def test_required_site_param(self):
        with self.assertRaises(ValueError) as ctx:
            OqParam(
                calculation_mode='scenario',
                gsim='AbrahamsonSilva2008',
                sites='0.1 0.2',
                maximum_distance='400',
                reference_vs30_value='760',
                intensity_measure_types='PGA',
                inputs=fakeinputs,
            ).validate()
        self.assertIn(
            "Please set a value for 'reference_depth_to_1pt0km_per_sec', this "
            "is required by the GSIM [AbrahamsonSilva2008]",
            str(ctx.exception))

    def test_uniform_hazard_spectra(self):
        with self.assertRaises(ValueError) as ctx:
            OqParam(
                calculation_mode='classical',
                gsim='BooreAtkinson2008',
                reference_vs30_value='200',
                sites='0.1 0.2, 0.3 0.4',
                poes='0.2',
                maximum_distance='400',
                truncation_level='3',
                intensity_measure_types_and_levels="{'PGV': [0.1, 0.2, 0.3]}",
                uniform_hazard_spectra='1',
                inputs=fakeinputs,
            ).set_risk_imts({})
        self.assertIn("The `uniform_hazard_spectra` can be True only if "
                      "the IMT set contains SA(...) or PGA",
                      str(ctx.exception))

        with mock.patch('logging.warning') as w:
            OqParam(
                calculation_mode='classical',
                gsim='BooreAtkinson2008',
                reference_vs30_value='200',
                sites='0.1 0.2, 0.3 0.4',
                poes='0.2',
                maximum_distance='400',
                truncation_level='3',
                intensity_measure_types_and_levels="{'PGA': [0.1, 0.2, 0.3]}",
                uniform_hazard_spectra='1',
                inputs=fakeinputs,
            ).set_risk_imts({})
        self.assertIn("There is a single IMT, the uniform_hazard_spectra plot "
                      "will contain a single point", w.call_args[0][0])

    def test_gmfs_but_no_sites(self):
        inputs = fakeinputs.copy()
        inputs['gmfs'] = ['fake.csv']
        with self.assertRaises(InvalidFile) as ctx:
            OqParam(
                calculation_mode='scenario_damage',
                inputs=inputs,
                maximum_distance='400')
        self.assertIn('You forgot to specify a site_model', str(ctx.exception))

    def test_disaggregation(self):
        with self.assertRaises(InvalidFile) as ctx:
            OqParam(
                calculation_mode='disaggregation',
                inputs=fakeinputs,
                gsim='BooreAtkinson2008',
                reference_vs30_value='200',
                sites='0.1 0.2',
                maximum_distance='400',
                truncation_level='3',
                intensity_measure_types_and_levels="{'PGV': [0.1, 0.2, 0.3]}",
                uniform_hazard_spectra='1')
        self.assertIn("poes_disagg or iml_disagg must be set",
                      str(ctx.exception))

        with self.assertRaises(InvalidFile) as ctx:
            OqParam(
                calculation_mode='disaggregation',
                inputs=fakeinputs,
                gsim='BooreAtkinson2008',
                reference_vs30_value='200',
                sites='0.1 0.2',
                poes='0.2',
                maximum_distance='400',
                truncation_level='3',
                iml_disagg="{'PGV': 0.1}",
                poes_disagg="0.1",
                uniform_hazard_spectra='1')
        self.assertIn("poes_disagg != poes: [0.1]!=[0.2]",
                      str(ctx.exception))


###################### test conversion to .ini format #####################
inis = []
base = os.path.dirname(event_based.__file__)
for case in os.listdir(base):
    job_ini = os.path.os.path.join(base, case, 'job.ini')
    if os.path.exists(job_ini):
        inis.append(job_ini)


@pytest.mark.parametrize('job_ini', inis)
def test_to_ini(job_ini):
    # make sure the file generated by to_ini can be read by get_oqparam
    oq = readinput.get_oqparam(job_ini)
    tmp_ini = os.path.join(oq.base_path, 'tmp.ini')
    with open(tmp_ini, 'w', encoding='utf-8-sig') as f:
        f.write(oq.to_ini())
    readinput.get_oqparam(tmp_ini)
