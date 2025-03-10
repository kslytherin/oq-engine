Developing with the Engine
==========================

Some advanced users are interested in developing with the engine,
usually to contribute new GMPEs and sometimes to submit a bug
fix. There are also users interested in implementing their own
customizations of the engine. This part of the manual is for them.

Prerequisites
-------------

It is assumed here that you are a competent scientific Python
programmer, i.e. that you have a good familiarity with the Python
ecosystem (including pip and virtualenv) and its scientific stack
(numpy, scipy, h5py, …).

Since engine v2.0 there is no need to know anything about databases
and web development (unless you want to develop on the WebUI part) so
the barrier for contribution to the engine is much lower than it used
to be. However, contributing is still nontrivial, and it absolutely
necessary to know git and the tools of Open Source development in
general, in particular about testing. If this is not the case, you
should do some study on your own and come back later. There is a huge
amount of resources on the net about these topics. Familiarity with
the standard Python coding guidelines (PEP8) is expected.
The engine enforces a few additional constraints:

1. functions and methods can have at most 90 lines
2. functions and methods can have at most 16 arguments

Pull requests breaking the above constraints will break the
automatic tests executed by GitHub, i.e. the GitHub Actions.

This manual will focus solely on the OpenQuake engine and it assumes
that you already know how to use it, i.e. you have read the User
Manual first. It is also useful to have an idea of the architecture
of the engine and its components, like the DbServer and the
WebUI. For that you should read the :ref:`Architecture of the
OpenQuake engine <architecture-of-oq-engine>` section.

There are also external tools which are able to interact with the
engine, like the QGIS plugin to run calculations and visualize the
outputs and the IPT tool to prepare the required input files (except
the hazard models). Unless you are developing for such tools you can
safely ignore them.

The first thing to do
---------------------

The first thing to do if you want to develop with the engine is to remove any non-development installation of the engine
that you may have. While it is perfectly possible to install on the same machine both a development and a production
instance of the engine (it is enough to configure the ports of the DbServer and WebUI) it is easier to work with a
single instance. In that way you will have a single code base and no risks of editing the wrong code. A development
installation the engine works as any other development installation in Python: you should clone the engine repository,
create and activate a virtualenv and then perform a *pip install -e* . from the engine main directory, as normal. You can
find the details here:

:ref:`development page <development>`

It is also possible to develop on Windows (:ref:`development page <development>`) but very few people in GEM are doing that, so you are on your
own, should you encounter difficulties. We recommend Linux, but Mac also works.

Since you are going to develop with the engine, you should also install the development dependencies that by default are
not installed. They are listed in the setup.py file, and currently (January 2020) they are pytest, flake8, pdbpp, silx
and ipython. They are not required but very handy and recommended. It is the stack we use daily for development.

Understanding the engine
------------------------

Once you have the engine installed you can run calculations. We recommend starting from the demos directory which
contains examples of hazard and risk calculations. For instance you could run the area source demo with the following
command::

	$ oq run demos/hazard/AreaSourceClassicalPSHA/job.ini

You should notice that we used here the command ``oq run`` while the engine manual recommends the usage of ``oq engine
--run``. There is no contradiction. The command ``oq engine --run`` is meant for production usage, but here we are doing
development, so the recommended command is ``oq run`` which will will be easier to debug thanks to the flag ``--pdb``,
which will start the python debugger should the calculation fail. Since during development is normal to have errors and
problems in the calculation, this ability is invaluable.

If you want to understand what happened during the calculation you should generate the associated .rst report, which can
be seen with the command ``$ oq show fullreport``. There you will find a lot of interesting information that it is worth
studying and we will discuss in detail in the rest of this manual. The most important section of the report is probably
the last one, titled “Slowest operations”. For that one can understand the bottlenecks of a calculation and, with
experience, the user can understand which part of the engine needs to optimize. Also, it is very useful to play with the
parameters of the calculation (like the maximum distance, the area discretization, the magnitude binning, etc.) and see
how the performance changes. There is also a command to plot hazard curves and a command to compare hazard curves between
different calculations: it is common to be able to get big speedups simply by changing the input parameters in the
*job.ini* of the model, without changing much the results.

There a lot of *oq commands*: if you are doing development you should study all of them. They are documented
`here <https://docs.openquake.org/oq-engine/master/advanced/oq-commands.md>`_.

Running calculations programmatically
-------------------------------------

Starting from engine v3.21 the recommended way to run a calculation
programmatically is via the pair ``create_jobs/run_jobs``::

	>> from openquake.engine import engine
        >> jobs = engine.create_jobs(['job_ini'])  # one-element list
        >> engine.run_jobs(jobs)

Then the results can be read from the datastore by using the extract API::

	>> from openquake.commonlib import datastore
	>> from openquake.calculators.extract import extract
	>> extract(datastore.read(jobs[0].calc_id), 'something')

The advantage of ``create_jobs`` is that it also accepts dictionaries
of parameters. So, instead of generating multiple `job.ini`
files, you can generate dictionaries, which is a lot more convenient.

There is an example in the directory
demos/risk/ScenarioRisk, called ``sensitivity.py``, which is performing
scenario risk calculations starting for the same planar rupture, but with
different values of the strike angle (0, 90 and 180 degrees).
The relevant code is something like this:

.. python:

  """Sensitivity of the risk from the strike parameter"""
  import os
  from openquake.engine import engine

  # template for the ini parameters
  base_ini = dict(
      description="scenario_risk with strike ",
      calculation_mode="scenario_risk",
      region="78.0 31.5, 89.5 31.5, 89.5 25.5, 78.0 25.5",
      inputs={'exposure': ["exposure_model.xml"],
              'structural_vulnerability': 'structural_vulnerability_model.xml'},
      reference_vs30_value="760.0",
      reference_depth_to_1pt0km_per_sec='100.0',
      intensity_measure_types="PGA",
      truncation_level="0",  # ignore stochastic uncertainty
      maximum_distance="500",
      gsim="ChiouYoungs2008",
      number_of_ground_motion_fields="1")

  def run_risk(strikes):
      # build ini dictionaries with different strikes
      inis = []
      for strike in strikes:
          ini = base_ini.copy()
          ini['description'] += str(strike)
          ini['rupture_dict'] = str({
              'lon': 80, 'lat': 30, 'dep': 10, 'mag': 6, 'rake': 0,
              'strike': strike, 'dip': 90})
          inis.append(ini)
      # run sequentially the calculations
      engine.run_jobs(engine.create_jobs(inis))
  
  if __name__ == '__main__':
      run_risk(strikes=[0, 90, 180])

Notice that this documentation can get out of sync with the code. The version
which is tested and guaranteed to run is the one at https://github.com/gem/oq-engine/blob/master/demos/risk/ScenarioRisk/sensitivity.py, which also sets the environment
variable OQ_DISTRIBUTE to ``zmq``. This is the easiest way to parallelize the jobs,
which makes sense since in this case the jobs are small.

After running the script you will have 3 calculations and you can see the effect
on the risk by looking at the portfolio_loss::

   $ oq show portfolio_loss -3  # strike=0
   +------+------------+
   | loss | structural |
   +------+------------+
   | avg  | 77_607_416 |
   +------+------------+

   $ oq show portfolio_loss -2  # strike=90
   +------+------------+
   | loss | structural |
   +------+------------+
   | avg  | 78_381_808 |
   +------+------------+

   $ oq show portfolio_loss -1  # strike=180
   +------+------------+
   | loss | structural |
   +------+------------+
   | avg  | 77_601_176 |
   +------+------------+

The exact numbers may change depending on the version of the engine.

Case study: computing the impact of a source on a site
------------------------------------------------------

As an exercise showing off how to use the engine as a library, we will solve the problem of computing the hazard on a
given site generated by a given source, with a given GMPE logic tree and a few parameters, i.e. the intensity measure
levels and the maximum distance.

The first step is to specify the site and the parameters; let’s suppose that we want to compute the probability of
exceeding a Peak Ground Accelation (PGA) of 0.1g by using the ToroEtAl2002SHARE GMPE::

	>>> from openquake.commonlib import readinput
	>>> oq = readinput.get_oqparam(dict(
	... calculation_mode='classical',
	... sites='15.0 45.2',
	... reference_vs30_type='measured',
	... reference_vs30_value='600.0',
	... intensity_measure_types_and_levels="{'PGA': [0.1]}",
	... investigation_time='50.0',
	... gsim='ToroEtAl2002SHARE',
	... truncation_level='99.0',
	... maximum_distance='200.0'))

Then we need to specify the source::

	>>> from openquake.hazardlib import nrml
	>>> src = nrml.get('''
	...         <areaSource
	...         id="126"
	...         name="HRAS195"
	...         >
	...             <areaGeometry discretization="10">
	...                 <gml:Polygon>
	...                     <gml:exterior>
	...                         <gml:LinearRing>
	...                             <gml:posList>
	...                                 1.5026169E+01 4.5773603E+01
	...                                 1.5650548E+01 4.6176279E+01
	...                                 1.6273108E+01 4.6083465E+01
	...                                 1.6398742E+01 4.6024744E+01
	...                                 1.5947759E+01 4.5648318E+01
	...                                 1.5677179E+01 4.5422577E+01
	...                             </gml:posList>
	...                         </gml:LinearRing>
	...                     </gml:exterior>
	...                 </gml:Polygon>
	...                 <upperSeismoDepth>0</upperSeismoDepth>
	...                 <lowerSeismoDepth>30</lowerSeismoDepth>
	...             </areaGeometry>
	...             <magScaleRel>WC1994</magScaleRel>
	...             <ruptAspectRatio>1</ruptAspectRatio>
	...             <incrementalMFD binWidth=".2" minMag="4.7">
	...                 <occurRates>
	...                     1.4731083E-02 9.2946848E-03 5.8645496E-03
	...                     3.7002807E-03 2.3347193E-03 1.4731083E-03
	...                     9.2946848E-04 5.8645496E-04 3.7002807E-04
	...                     2.3347193E-04 1.4731083E-04 9.2946848E-05
	...                     1.7588460E-05 1.1097568E-05 2.3340307E-06
	...                 </occurRates>
	...             </incrementalMFD>
	...             <nodalPlaneDist>
	...                 <nodalPlane dip="5.7596810E+01" probability="1"
	...                             rake="0" strike="6.9033586E+01"/>
	...             </nodalPlaneDist>
	...             <hypoDepthDist>
	...                 <hypoDepth depth="1.0200000E+01" probability="1"/>
	...             </hypoDepthDist>
	...         </areaSource>
	... ''')

Then the hazard curve can be computed as follows::

	>>> from openquake.hazardlib.calc.hazard_curve import calc_hazard_curve
	>>> from openquake.hazardlib import valid
	>>> sitecol = readinput.get_site_collection(oq)
	>>> gsims = readinput.get_gsim_lt(oq).values['*']
	>>> calc_hazard_curve(sitecol, src, gsims, oq)
	array([[0.00508004]], dtype=float32)


Working with GMPEs directly: the ContextMaker
---------------------------------------------

If you are an hazard scientist, you will likely want to interact with the GMPE library in ``openquake.hazardlib.gsim``.
The recommended way to do so is in terms of a ``ContextMaker`` object.::

	>>> from openquake.hazardlib.contexts import ContextMaker

In order to instantiate a ``ContextMaker`` you first need to populate a dictionary of parameters::

	>>> param = dict(maximum_distance=oq.maximum_distance, imtls=oq.imtls,
	...              truncation_level=oq.truncation_level,
	...              investigation_time=oq.investigation_time)
	>>> cmaker = ContextMaker(src.tectonic_region_type, gsims, param)

Then you can use the ``ContextMaker`` to generate context arrays from the sources::

	>>> [ctx] = cmaker.from_srcs([src], sitecol)

In our example, there are 15 magnitudes::

	>>> len(src.get_annual_occurrence_rates())
	15

and the area source contains 47 point sources::

	>>> len(list(src))
	47

so in total there are 15 x 47 = 705 ruptures::

	>>> len(ctx)
	705

The ``ContextMaker`` takes care of the maximum_distance filtering, so in general the number of contexts is lower than the
total number of ruptures, since some ruptures are normally discarded, being distant from the sites.

The contexts contain all the rupture, site and distance parameters.

Then you have::

	>>> ctx.mag[0]
	4.7
	>>> round(ctx.rrup[0], 1)
	106.4
	>>> round(ctx.rjb[0], 1)
	105.9

In this example, the GMPE ``ToroEtAl2002SHARE`` does not require site parameters, so calling ``ctx.vs30`` will raise an
``AttributeError`` but in general the contexts contain also arrays of site parameters. There is also an array of indices
telling which are the sites affected by the rupture associated to the context::

	>>> import numpy
	>>> numpy.unique(ctx.sids)
	array([0], dtype=uint32)

Once you have the contexts, the ``ContextMaker`` is able to compute means and standard deviations from the underlying
GMPEs as follows (for engine version >= v3.13):::

	>>> mean, sig, tau, phi = cmaker.get_mean_stds([ctx])

Since in this example there is a single gsim and a single IMT you will get::

	>>> mean.shape
	(1, 1, 705)
	>>> sig.shape
	(1, 1, 705)

The shape of the arrays in general is (G, M, N) where G is the number of GSIMs, M the number of intensity measure types
and N the total size of the contexts. Since this is an example with a single site, each context has size 1, therefore
N = 705 * 1 = 705. In general if there are multiple sites a context M is the total number of affected sites. For instance
if there are two contexts and the first affect 1 sites and the second 2 sites then N would be 1 + 2 = 3. This example
correspond to 1 + 1 + … + 1 = 705.

From the mean and standard deviation is possible to compute the probabilities of exceedence. The ``ContextMaker`` provides
a method to compute directly the probability map, which internally calls ``cmaker.get_pmap([ctx])`` which gives exactly
the result provided by ``calc_hazard_curve(sitecol, src, gsims, oq)`` in the section before.

If you want to know exactly how ``get_pmap`` works you are invited to look at the source code in
``openquake.hazardlib.contexts``.

Generating ground motion fields from a rupture
----------------------------------------------

The easiest way to create a finite size rupture (a.k.a. planar rupture)
is to use the factory function `get_planar`:

>>> from openquake.hazardlib.source.rupture import get_planar

The function requires in input a site and a magnitude scaling relationship,
so first you have to build such objects:

>>> [site] = sitecol  # since there is a single site
>>> from openquake.hazardlib.scalerel import WC1994
>>> msr = WC1994()  # magnitude scaling relationship
>>> mag = 6.
>>> rup = get_planar(site, msr, mag, aratio=1., strike=11., dip=38.,
...                  rake=55., trt=cmaker.trt)

If you want to generate the GMF produced by a rupture (i.e. to emulate
a scenario calculation) you need to supplement the number of
occurrences of the rupture and a random seed, i.e. you need to convert the
hazardlib rupture into an EBRupture:

>>> from openquake.hazardlib.source.rupture import EBRupture
>>> ebr = EBRupture(rup, n_occ=2, seed=42)

Then you can use the GmfComputer class to perform the calculation:

>>> from openquake.hazardlib.calc.gmf import GmfComputer
>>> gc = GmfComputer(ebr, sitecol, cmaker)
>>> gc.compute_all()  # returns a DataFrame
        PGA  eid  sid  rlz
0  0.660239    0    0    0
1  0.301583    1    0    0

`PGA` is the value of the ground motion field for the first IMT, `eid` the event ID, `sid` the site ID (there is a single site in this case)
and `rlz` the realization index.
In scenario calculations there is a realization for each GSIM and in this case
there is a single GSIM, so rlz=0. The total number of events is the number
of realizations times the number of occurrences and therefore in this case
the event ID (`eid`) can only have the values 0 or 1.

It is also possible to perform calculations with point-like ruptures
(i.e. ignoring the finite-size effects):

>>> from openquake.hazardlib.source.rupture import PointRupture
>>> occ_rate = None  # not used in the GmfComputer
>>> rup =  PointRupture(mag, cmaker.trt, site.location, occ_rate, cmaker.tom)
>>> ebr = EBRupture(rup, n_occ=2, seed=42)
>>> GmfComputer(ebr, sitecol, cmaker).compute_all()
        PGA  eid  sid  rlz
0  0.541180    0    0    0
1  0.247199    1    0    0

The event based calculator is able to generate a set of ruptures
from the source models and to store then on the datastore.
Then it is possible to read the ruptures, filter them and perform
calculations; the function to use is
`openquake.calculators.getters.get_ebruptures` which takes in input
a DataStore object and returns a list of `EBRupture` instances.

Working with verification tables
--------------------------------

Hazard scientists implementing a new GMPE must provide verification tables, i.e. CSV files containing inputs and expected
outputs.

For instance, for the Atkinson2015 GMPE (chosen simply because is the first GMPE in lexicographic order in hazardlib)
the verification table has a structure like this::

	rup_mag,dist_rhypo,result_type,pgv,pga,0.03,0.05,0.1,0.2,0.3,0.5
	2.0,1.0,MEAN,5.50277734e-02,3.47335058e-03,4.59601700e-03,7.71361460e-03,9.34624779e-03,4.33207607e-03,1.75322233e-03,3.44695521e-04
	2.0,5.0,MEAN,6.43850933e-03,3.61047741e-04,4.57949482e-04,7.24558049e-04,9.44495571e-04,5.11252304e-04,2.21076069e-04,4.73435138e-05
	...

The columns starting with ``rup_`` contain rupture parameters (the magnitude in this example) while the columns starting
with ``dist_`` contain distance parameters. The column ``result_type`` is a string in the set {“MEAN”, “INTER_EVENT_STDDEV”,
“INTRA_EVENT_STDDEV”, “TOTAL_STDDEV”}. The remaining columns are the expected results for each intensity measure type;
in the the example the IMTs are PGV, PGA, SA(0.03), SA(0.05), SA(0.1), SA(0.2), SA(0.3), SA(0.5).

Starting from engine version v3.13, it is possible to instantiate a ``ContextMaker`` and the associated contexts from a
GMPE and its verification tables with a few simple steps. First of all one must instantiate the GMPE::

	>>> from openquake.hazardlib import valid
	>>> gsim = valid.gsim("Atkinson2015")

Second, one can determine the path names to the verification tables as follows (they are in a subdirectory of
*hazardlib/tests/gsim/data*)::

	>>> import os
	>>> from openquake.hazardlib.tests.gsim import data
	>>> datadir = os.path.join(data.__path__[0], 'ATKINSON2015')
	>>> fnames = [os.path.join(datadir, f) for f in ["ATKINSON2015_MEAN.csv",
	...           "ATKINSON2015_STD_INTER.csv", "ATKINSON2015_STD_INTRA.csv",
	...           "ATKINSON2015_STD_TOTAL.csv"]]

Then it is possible to instantiate the ``ContextMaker`` associated to the GMPE and a pandas DataFrame associated to the
verification tables in a single step::

	>>> from openquake.hazardlib.tests.gsim.utils import read_cmaker_df, gen_ctxs
	>>> cmaker, df = read_cmaker_df(gsim, fnames)
	>>> list(df.columns)
	['rup_mag', 'dist_rhypo', 'result_type', 'damping', 'PGV', 'PGA', 'SA(0.03)', 'SA(0.05)', 'SA(0.1)', 'SA(0.2)', 'SA(0.3)', 'SA(0.5)', 'SA(1.0)', 'SA(2.0)', 'SA(3.0)', 'SA(5.0)']

Then you can immediately compute mean and standard deviations and compare with the values in the verification table::

	>>> mean, sig, tau, phi = cmaker.get_mean_stds(gen_ctxs(df))

*sig* refers to the “TOTAL_STDDEV”, *tau* to the “INTER_EVENT_STDDEV” and *phi* to the “INTRA_EVENT_STDDEV”. This is how
the tests in hazardlib are implemented. Interested users should look at the code in gem/oq-engine.

Running the engine tests
------------------------

If you are a hazard scientist contributing a bug fix to a GMPE (or any other kind of bug fix) you may need to run the
engine tests and possibly change the expected files if there is a change in the numbers. The way to do it is to give the following command from the repository root::

	$ pytest -vx openquake/calculators

If you get an error like the following::

	openquake/calculators/tests/__init__.py:218: in assertEqualFiles
	    raise DifferentFiles('%s %s' % (expected, actual))
	E   openquake.calculators.tests.DifferentFiles: /home/michele/oq-engine/openquake/qa_tests_data/classical/case_1/expected/hazard_curve-PGA.csv /tmp/tmpkdvdhlq5/hazard_curve-mean-PGA_27249.csv

you need to change the expected file, i.e. copy the file ``/tmp/tmpkdvdhlq5/hazard_curve-mean-PGA_27249.csv`` over
``classical/case_1/expected/hazard_curve-PGA.csv``.
