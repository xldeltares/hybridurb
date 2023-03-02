from pathlib import Path
from hybridurb.utils import *
import warnings
warnings.filterwarnings("ignore")
import glob
import datetime
import time
import cProfile
import pandas as pd
import numpy as np
from patsy import dmatrices
from multiprocessing import Pool
import itertools
from scipy import stats
import pstats
from pstats import SortKey
import io
import logging
import re
import xarray as xr
from pathlib import Path
from meshkernel import Mesh1d
from xugrid import Ugrid1d

logger = logging.getLogger(__name__)
# TODO use propoer logger

# run model option 1 (bottom up)
def runGraphModel_option1(X, rgRR, rg_id):
    def WallingfordMethod(A, S, i10):
        # S - slope (m/m)
        # A - the area (m2) of the surface, for example the area of the paved surface
        # i10 - running ten minute average of rainfall intensity
        # Now using T2 rainfall 73mm/hr
        if S < 0.002: S = 0.002
        if A < 1000: A = 1000
        if A > 10000: A = 10000
        C = 0.117 * S ** (-0.13) * A ** (0.24)
        i_ = 0.5 * (1 + i10)
        k = C * i_ ** (-0.39)
        return k

    def linearReservior(R, A):
        # R is effective rainfall, m^3/hr - numpy array
        # A is reservior constant - hr
        # Q is m3/s
        A = 1. / A
        Q = np.zeros(len(R))
        for i in range(1, len(R)):
            Q[i] = Q[i - 1] * np.exp(-A * 5. / 60) + R[i - 1] * (1 - np.exp(-A * 5. / 60))
        return Q / 3600.

    ###################################
    # Assign rainfall
    ###################################
    # sim_n_RR: rainfall profile at node
    sim_n_RR_dur = len(rgRR)
    sim_n_RR = {n: rgRR.loc[:, str(i.rg_id)].values \
                for n, i in rg_id.iterrows()}
    ###################################
    # Bottom up method
    ###################################
    # sim_n_RO: Runoff generation and routing at node
    #   1. fixed runoff method (sim_n_RO = I(R,t)*cont_area * 0.8), not output, integrated in the enxt step
    #   2. single linear reservior following wallingford method, rainfall = T2
    sim_n_RO = {n: linearReservior(sim_n_RR[n] / (10 ** 3) * X.nodes[n]['cont_area_m2'] * 0.8,
                                   WallingfordMethod(X.nodes[n]['cont_area_m2'], X.nodes[n]['subcatslope'], 73.))
                for n in sim_n_RR.keys()}  # m3/s
    # sim_nc_RO: Runoff accumulation in lumped sewer subnetwork
    #   simple sum
    # sim_nc_Q: Flow routing in lumped sewer subnetwork
    #   single linear reservior with tc as time constant
    sim_nc_RO = {}
    sim_nc_Q = {}
    XX = X.reverse()
    for n in XX.nodes:
        if X.nodes[n]['tc_min'] > 0:
            # upstream catchment
            ns = list(nx.dfs_postorder_nodes(XX, n))
            # include only nodes with contributing subcatchment
            ns_ = set(ns) - (set(ns) - set(sim_n_RO.keys()))
            # runoff - accumulation in lumped sewer subnetwork manholes
            if len(ns_) > 0:
                RO_t = np.array([sim_n_RO[nsi] for nsi in ns_]).sum(axis=0)  # m3/s
                # Flow routing in lumped sewer subnetwork
                Q_t = linearReservior(RO_t * 3600, X.nodes[n]['tc_min'] / 60)  # m3/s
            else:
                RO_t = np.zeros(shape=(sim_n_RR_dur,))
                Q_t = np.zeros(shape=(sim_n_RR_dur,))
            sim_nc_RO[n] = RO_t[:]
            sim_nc_Q[n] = Q_t[:]
    ###################################
    # shared method
    ###################################
    # sim_o_T
    outfalls = [x for x in X.nodes if X.nodes[x]['type'].lower() in ['outfall']]
    sim_o_T = {o: np.nan * np.zeros(shape=(sim_n_RR_dur,)) for o in outfalls}
    for o in outfalls:
        if X.nodes[o]['tc_min']:
            # upstream catchment
            ns = list(nx.dfs_postorder_nodes(XX, o))
            # include only nodes with contributing rg
            ns_ = set(ns) - (set(ns) - set(sim_n_RR.keys()))
            if len(ns_) > 0:
                # rainfall averaging
                I_t = np.array([sim_n_RR[nsi] for nsi in ns_]).mean(axis=0)  # should not include 0
                # rainfall rounting
                window = int(X.nodes[n]['tc_min'] / 5)
                I_tc = np.convolve(I_t, np.ones(window, ) / window, mode='full')[:I_t.size]
            else:
                I_tc = np.zeros(shape=(sim_n_RR_dur,))
            I_I1month = I_idf(1. / 12., X.nodes[o]['tc_min'])
            sim_o_T[o] = np.array([T_idf(intensity, X.nodes[o]['tc_min']) \
                                       if intensity > I_I1month else 0. for intensity in I_tc])
            # result
    sim_nc_RO = pd.DataFrame.from_dict(sim_nc_RO);
    sim_nc_RO.index = rgRR.index
    sim_nc_Q = pd.DataFrame.from_dict(sim_nc_Q);
    sim_nc_Q.index = rgRR.index
    sim_o_T = pd.DataFrame.from_dict(sim_o_T);
    sim_o_T.index = rgRR.index
    return sim_nc_RO, sim_nc_Q, sim_o_T


# get predictor option 1 (NS,PS,OT,d,Cat)
def getPredictors_option1(X, sim_nc_RO, sim_nc_Q, sim_o_T):
    # static FIss
    table_Cs = pd.DataFrame.from_dict(dict(X.nodes(data=True)), orient='index')
    table_Cs.loc[:, 'd'] = (table_Cs.loc[:, 'chambvolume_m3'] / table_Cs.loc[:, 'chambarea_m2']).rename('d')
    table_Cs = table_Cs[['Outfall', 'd', 'Cat', 'Qo_m3ds', 'us_area_m2']]
    # dynamic FIs
    outfalls = [x for x in X.nodes if X.nodes[x]['type'].lower() in ['outfall']]
    table_Qs = sim_nc_Q.max().to_frame(name='nc_Qmq_m3ds');
    table_Qs = table_Qs.drop([o for o in outfalls if o in table_Qs.index])
    for n in table_Qs.index:
        mq_idx = sim_nc_Q[n].idxmax();
        table_Qs.loc[n, 'flooding_time'] = mq_idx
        table_Qs.loc[n, 'nc_ROmq_m3ds'] = sim_nc_RO.loc[mq_idx, n]
        table_Qs.loc[n, 'o_Tmq_yr'] = sim_o_T.loc[mq_idx, X.nodes[n]['Outfall']]
        # all
    d = pd.concat([table_Cs, table_Qs], axis=1)
    d.loc[:, 'NS'] = d.loc[:, 'nc_ROmq_m3ds'].values * 300 / d.loc[:, 'us_area_m2'].values * 1000.0
    d.loc[:, 'PS'] = d.loc[:, 'nc_Qmq_m3ds'].values / d.loc[:, 'Qo_m3ds'].values
    d.loc[:, 'OT'] = d.loc[:, 'o_Tmq_yr'];
    d.loc[:, 'OT'] = d.loc[:, 'OT'].round(1);
    d.loc[d.OT < 0.1, 'OT'] = 0.01
    d = d[['NS', 'PS', 'd', 'OT', 'Cat', 'flooding_time']]
    d = d.replace([np.inf, -np.inf], np.nan);
    d = d.dropna(how='any', axis=0)
    return d


# predict with confidence interval, increase time consumption by 1/2
def predictCI(model, X_test):
    y_pred = model.predict(X_test)
    proba = y_pred.values;
    X = X_test.values
    cov = model.cov_params()
    gradient = (proba * (1 - proba) * X.T).T  # matrix of gradients for each observation
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
    c = 1.96  # multiplier for confidence interval
    upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
    lower = np.maximum(0, np.minimum(1, proba - std_errors * c))
    y_pred = y_pred.to_frame(name='mean')
    y_pred.loc[:, 'lower'] = lower
    y_pred.loc[:, 'upper'] = upper
    return y_pred


# predict mean results
def _predict(at_time, rgrr_filename, rgid_filename, X, lams, model, equ, output_dir=None, hazard_map=False):
    # time
    at_time_ = datetime.datetime.strptime(at_time, '%Y%m%d%H%M')
    t0 = time.time()

    # ensemble member (derive from rainfall file)
    ens_id = re.findall("ens([0-9]+)", Path(rgrr_filename).name)[0]

    # rainfall preprocessing
    rg_id = pd.read_csv(rgid_filename, index_col=0)
    rgRR = pd.read_csv(rgrr_filename, index_col=0, parse_dates=True, sep = ';|,')
    if isinstance(rgRR.index[0], str): # extra parsing of datetime #TODO improve when standard timeseries becomes available
        rgRR.index = pd.to_datetime(rgRR.index, format='%d/%m/%Y %H.%M', errors='ignore')

    # graph prepreocessing # FIXME: remove no outfall locations -> unavailable for maing predictions
    _nodes = []
    for node in X.nodes:
        if 'Outfall' not in X.nodes[node]:
            pass
        else:
            _nodes.append(node)
    X = nx.subgraph(X, _nodes)

    # Run graph model
    sim_nc_RO, sim_nc_Q, sim_o_T = runGraphModel_option1(X, rgRR, rg_id)

    # prepare predictors (d - predictor, - ft flooding time)
    d = getPredictors_option1(X, sim_nc_RO, sim_nc_Q, sim_o_T)  # ['NS', 'PS', 'd',  'OT','Cat']
    d['n_F'] = d['d'] * 0
    for v in ['d', 'NS', 'PS', 'OT']:  d.loc[:, v] = stats.boxcox(d.loc[:, v], lams[v])

    # logistric reression
    _, X_test = dmatrices(equ, d, return_type='dataframe')
    # confidence interval prediction
    # y_pred = predictCI(model, X_test)
    # mean prediction
    y_pred = model.predict(X_test); y_pred = y_pred.to_frame(name = ens_id)
    ft = (d.flooding_time - at_time_) / np.timedelta64(1, 'm')
    # print(f'\tmember: %s timeit: {time.time() - t0})' % rgrr_filename.split('ens')[-1].split('.csv')[0])

    # write output
    y_pred.to_csv(Path(output_dir).joinpath(f"{at_time}_ens{ens_id}_ypred.csv"))

    # hazard map
    if hazard_map:
        drawHazardMap(X, y_pred, ft=ft)
        plt.savefig(rgrr_filename + '_ypred.png')
        print(f'\thazard map saved')

    return y_pred # TODO: support ft output

def predict(rgRR, rgID, X, lams, model, equ, event_id = None):
    """
    Predict the probability of flooding for a given rainfall event `rgRR`.

    :param rgRR: pd.DataFrame
        timeseries of the rainfall event. must have time as index and meteo catchments as columns
    :param rgID: pd.DataFrame
        mapping table to map rainfall at gauges to rainfall at nodes. must contain rg_id. index is node_id
    :param X: nx.Graph
        graph model (calibrated)
    :param lams:
        lambda parameter (calibrated)
    :param model:
        statistical model (calirbated)
    :param equ:
        equation used for the `model`
    :param event_id: str
        used as the identifier for the results

    :return: y_pred
    """

    # strat time
    at_time_ = rgRR.index[0]

    # rainfall preprocessing # FIXME: remove rainfall catchments are are not used in the model
    rgID = rgID.drop(index=list(set(rgID.index)-set(X.nodes())))

    # graph prepreocessing # FIXME: remove no outfall locations -> unavailable for maing predictions
    _nodes = []
    for node in X.nodes:
        if 'Outfall' not in X.nodes[node]:
            pass
        else:
            _nodes.append(node)
    X = nx.subgraph(X, _nodes)

    # Run graph model # FIXME: remove the parameter rg_id in function runGraphModel_option1
    sim_nc_RO, sim_nc_Q, sim_o_T = runGraphModel_option1(X, rgRR, rgID)

    # prepare predictors (d - predictor, - ft flooding time)
    d = getPredictors_option1(X, sim_nc_RO, sim_nc_Q, sim_o_T)  # ['NS', 'PS', 'd',  'OT','Cat']
    d['n_F'] = d['d'] * 0
    for v in ['d', 'NS', 'PS', 'OT']:  d.loc[:, v] = stats.boxcox(d.loc[:, v], lams[v])

    # logistric reression
    _, X_test = dmatrices(equ, d, return_type='dataframe')
    # confidence interval prediction
    # y_pred = predictCI(model, X_test)
    # mean prediction
    y_pred = model.predict(X_test); y_pred = y_pred.to_frame()
    ft = (d.flooding_time - at_time_) / np.timedelta64(1, 'm')
    # print(f'\tmember: %s timeit: {time.time() - t0})' % rgrr_filename.split('ens')[-1].split('.csv')[0])

    # add dientifier
    if event_id is not None:
        y_pred.columns = [event_id]

    return y_pred # TODO: support ft output

# TODO make it abstract class
# FIXME this is option 1 runner
class NowcastRunner(object):
    def __init__(self, root_dir:Path, t0: str, name:str = "", crs = "EPSG:4326"):
        self.t0 = t0
        self.name = name
        self.crs = crs
        self._model_dir = root_dir / 'model'
        self._input_dir = root_dir / 'input'
        self._output_dir = root_dir / 'output'
        self._output_dir.mkdir(exist_ok=True)
        self._graphmodel = None
        self._lambdamodel = None
        self._statsmodel = None
        self._initialise_model()
        self._inputs = None
        self._outputs = None

    def _initialise_model(self):
        self._graphmodel = read_gpickle(self._model_dir / "GraphModel.gpickle")
        self._lambdamodel = read_pickle(self._model_dir / "LambdaModel.pickle")
        self._statsmodel = read_pickle( self._model_dir / "StatsModel.pickle")
        return self._graphmodel, self._lambdamodel, self._statsmodel

    def _initialise_rainfall(self):
        # initialse rainfall nowcasts at_time 201605300900
        # remember to change filename to obs, or ens for other types of input e.g. obs.csv
        # rg_id id predefined
        at_time = self.t0
        rgrr_filenames = glob.glob(
            f"{self._input_dir}/{at_time}*ens*.csv")
        rgid_filenames = [Path(rgrr_filenames[0]).with_name("grid_to_node_mapping.csv") for
                          i in range(len(rgrr_filenames))]

        # helper function to read rainfall timeseries
        def _read_rainfall(filename):
            rgRR = pd.read_csv(filename, index_col=0, parse_dates=True, infer_datetime_format=True,sep=';|,')
            if isinstance(rgRR.index[0],
                          str):  # extra parsing of datetime #TODO improve when standard timeseries becomes available
                rgRR.index = pd.to_datetime(rgRR.index, format='%d/%m/%Y %H.%M', errors='ignore')
            return rgRR

        # read rainfall
        rgRR_list = [_read_rainfall(rgrr_filename) for rgrr_filename in rgrr_filenames]
        # read mapping
        rgID_list = [pd.read_csv(rgid_filename, index_col=0) for rgid_filename in rgid_filenames]
        # get event list
        event_list = [re.findall("ens([0-9]+)", Path(rgrr_filename).name)[0] for rgrr_filename in rgrr_filenames]

        return rgRR_list, rgID_list, event_list

    def run(self):
        """
        Run HybridUrb model using a bottom up approach that contains detailed physical process.

        Equations used for bottom up appraoch:
            'n_F ~ C(Cat)+NS+PS+OT+d'
                n_F: probability of flooding
                C(Cat): category of the noes
                NS: saturation of the nodes
                PS: saturation of the pipes
                OT: return periods of the outfall location
                d: depth of the nodes
        Reads rainfall input and model files:
            Rainfall inputs:
                [T0]_ens[ENSEMBLE_ID].csv and grid_to_nodes_mapping.csv.
                The former contains rainfall intensity (unit mm/h) per rainfall catchment for each ensemble member.
                The latter contains the mapping of rainfall catchments.
            Model files:
                GraphModel.gpickle: nx.Graph.  graph model of the sewer network.
                LambdaModel.pickle lambda model for boxcox transformation.
                StatsModel.pickle statistical model for the logistic regression.
        Writes output:
            Probabilistic  output:
                [T0]_ypred_summary.csv
                A summary table containing the probability of flooding of each node for each rainfall ensemble.
                Note that the output is not a timeseries.
            TODO: expected time of flooding
        """

        # initialise model
        equ = 'n_F ~ C(Cat)+NS+PS+OT+d' # FIXME: this is the place to select options, which is not ideal. Question, is it possible to get this equation from statsmodel?
        X = self._graphmodel
        lams = self._lambdamodel
        model = self._statsmodel

        # locate rainfall
        rgRR_list, rgID_list, event_list = self._initialise_rainfall()

        # start
        at_time = self.t0
        total_sims = len(rgRR_list)
        print(f'Starting simulation for {at_time}: \n\tTotal simulation:{total_sims}')
        pr = cProfile.Profile()
        pr.enable()

        # predict
        with Pool(processes=4) as pool:
            results = pool.starmap(predict, zip(rgRR_list, rgID_list,
                                                itertools.repeat(X), itertools.repeat(lams), itertools.repeat(model),
                                                itertools.repeat(equ)))

        # finish
        pr.disable()
        self._outputs = results

        # print simulation performance
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(20)
        print(s.getvalue())

        # write output
        y_preds = pd.concat(results, axis=1)
        y_preds.columns = list(range(total_sims))
        y_preds.to_csv(self._output_dir / f'{self.t0}_ypred_summary.csv')

    def run_preadaptor(self):
        """
        Run pre-adaptor for FEWS.

        Converts rainfall inputs prepared by Delft-FEWS into rainfall inputs supported by HybridUrb.
        Rainfall inputs prepared by Delft-FEWS:
            [T0]_precipitation_on_catchments.nc.
            It contains the rainfall quantity (unit mm) per rainfall catchments.
        Rainfall inputs supported by HybridUrb:
            [T0]_ens[ENSEMBLE_ID].csv and grid_to_nodes_mapping.csv.
            The former contains rainfall intensity (unit mm/h) per rainfall catchment for each ensemble member.
            The latter contains the mapping of rainfall catchments.
            #TODO: consider to support gridded input and do idmapping internally

        T0 (str)  is the start time of the input, supported format: yyyymmddHHMM, e.g. 201605300900;
        ENSEMBLE_ID (int) is the identifier of each ensemble member, known as dimension realization in [T0]_precipitation_on_catchments.nc
        """

        # open input file
        filename = self._input_dir.joinpath(f"{self.t0}_precipitation_on_catchments.nc")
        da = xr.open_dataset(filename)

        # unit conversion from mm to mm/hour
        dt_s = (da.time.values[1]-da.time.values[0]).astype('timedelta64[s]').astype(np.int32)
        da = da.assign(I_fc=lambda x: x.P_fc / dt_s * 3600)
        _station_ids = da["station_id"].values.astype(str).tolist()

        # an event per ensemble member
        events = []
        for realisation in da["realization"].values:
            # slice data
            di = da["I_fc"].sel(realization = realisation, drop  = True).drop_vars(['lat', 'lon'])
            # format dataframe
            df = di.to_dataframe()["I_fc"].unstack(level=1)
            df.columns = _station_ids
            df.to_csv(self._input_dir.joinpath(f"{self.t0}_ens{realisation}.csv"))

        # mapping table
        rg_id = pd.DataFrame({"rg_id": _station_ids}, index=_station_ids)
        rg_id.to_csv(self._input_dir.joinpath(f"grid_to_node_mapping.csv"))

    def run_postadaptor(self):
        """
        Run post-adaptor for FEWS.

        Converts outputs generated by HybridUrb to standard netcdf files supported by Delft-FEWS.
        Probabilistic nputs generated by HybridUrb:
            [T0]_ypred_summary.csv
            A summary table containing the probability of flooding of each node for each rainfall ensemble.
            Note that the output is not a timeseries.
        Outputs supported by Delft-FEWS:
            [T0]_predictions_at_nodes.nc.
            It contains the probabilistic predictions per model nodes.
            It has the dimension of realization which contains ENSEMBLE_ID.
            # TODO: check if the ensemble member actually match with input

        T0 (str)  is the start time of the input, supported format: yyyymmddHHMM, e.g. 201605300900;
        ENSEMBLE_ID (int) is the identifier of each ensemble member, known as dimension realization in [T0]_precipitation_on_catchments.nc
        """

        # prepares output
        y_preds = pd.concat(self._outputs, axis=1)

        # prepare output
        probability = y_preds.values
        analysis_time = pd.Timestamp(self.t0)
        realization = range(len(y_preds.columns))

        # prepare output grid
        node_loc = {n: i for i,n in enumerate(y_preds.index)} #FIXME for now only non-nan locations are written
        node_x = np.array([self._graphmodel.nodes[n]['geo'][0] for n in y_preds.index])
        node_y = np.array([self._graphmodel.nodes[n]['geo'][1] for n in y_preds.index])
        edge_nodes = np.zeros(node_x.size * 2, np.int32)
        edge_index = 0
        for start_node, end_node in self._graphmodel.edges:
            if np.logical_and(start_node in node_loc, end_node in node_loc):
                edge_nodes[edge_index] = node_loc[start_node]
                edge_index += 1
                edge_nodes[edge_index] = node_loc[end_node]
                edge_index += 1
        mesh1d = Mesh1d(node_x, node_y, edge_nodes)
        grid = Ugrid1d.from_meshkernel(mesh1d, name = "mesh1d") #FIXME add crs
        grid.to_dataset().to_netcdf(self._output_dir.joinpath(f"{self.name}_grid.nc"))
        ds = grid.to_dataset()
        ds.mesh1d_node_x.attrs = {'units': 'm', 'standard_name': 'projection_x_coordinate', 'long_name': 'x-coordinate of mesh nodes'}
        ds.mesh1d_node_y.attrs = {'units': 'm', 'standard_name': 'projection_y_coordinate', 'long_name': 'y-coordinate of mesh nodes'}
        # write grid
        ds.to_netcdf(self._output_dir.joinpath(f"{self.name}_grid.nc"))

        # time dimension
        ds.coords["analysis_time"] = xr.DataArray([analysis_time], dims=["analysis_time"])
        ds.analysis_time.attrs = {'standard_name': 'forecast_reference_time', 'long_name': 'forecast_reference_time'}
        ds.coords["time"] = xr.DataArray(pd.date_range(start = analysis_time, freq = "5T", periods = 2), dims=["time"])
        ds.time.attrs = {'standard_name': 'time', 'long_name': 'time', 'axis': 'T'}

        # data variable
        # use realization (NOTE: this is the desirable option which is not yet supported by FEWS)
        ds.coords["realization"] = xr.DataArray(realization, dims=["realization"])
        ds.realization.attrs = {'standard_name': 'realization', 'long_name': 'Index of an ensemble member within an ensemble', 'units': '1'}
        ds["mesh1d_probability"] = xr.DataArray(np.stack([probability.T, probability.T]), dims=["time", "realization", grid.node_dimension])
        ds.mesh1d_probability.attrs = {'mesh': 'mesh1d', 'location': 'node', 'standard_name': 'thunderstorm_probability', 'long_name': 'Probability of flooding',
         'units': '1', 'grid_mapping': ''}
        ds.to_netcdf(self._output_dir.joinpath(f"{self.t0}_predictions_at_nodes_ensemble.nc"))

        # do not use realization, run in loop. (NOTE deprecate this when the above becomes supported).
        for r in realization:
            ds["mesh1d_probability"] = xr.DataArray(np.stack([probability[:,r], probability[:,r]]), dims=["time", grid.node_dimension])
            ds.mesh1d_probability.attrs = {'mesh': 'mesh1d', 'location': 'node', 'standard_name': 'thunderstorm_probability', 'long_name': 'Probability of flooding',
             'units': '1', 'grid_mapping': ''}
            ds.to_netcdf(self._output_dir.joinpath(f"{self.t0}_predictions_at_nodes_{r}.nc"))


    def run_fews(self):

        # run preadaptor
        print(f'Starting preadaptor for fews')
        self.run_preadaptor()
        print(f'Finished preadaptor for fews')

        # initialise model
        print(f'Starting run')
        self.run()
        print(f'Finished run')

        # run postadaptor
        print(f'Starting postadaptor for fews')
        self.run_postadaptor()
        print(f'Finished postadaptor for fews')
