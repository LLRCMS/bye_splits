# coding: utf-8

_all_ = [ 'validation', 'stats_collector' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common, params

import re
import yaml
import numpy as np
import pandas as pd
import h5py
from dataclasses import dataclass

def validation_cmssw(pars, **kw):
    in1_valid = common.fill_path(kw['ClusterOutValidation'], **pars)
    in2_valid = common.fill_path(kw['FillOut'], **pars)

    with pd.HDFStore(in1_valid, mode='r') as sloc, h5py.File(in2_valid, mode='r') as scmssw:
        local_keys = sloc.keys()
        cmssw_keys = [x for x in scmssw.keys() if '_cl' in x]
        assert(len(local_keys) == len(cmssw_keys))
     
        for key1, key2 in zip(local_keys, cmssw_keys):
            local = sloc[key1]
            cmssw = scmssw[key2]
            cmssw_cols = list(cmssw.attrs['columns'])
            cmssw = cmssw[:]
     
            search_str = '{}_([0-9]{{1,7}})_ev'.format(kw['FesAlgo'])
            event_number = re.search(search_str, key1).group(1)
     
            locEta = local['eta'].to_numpy()
            locPhi = local['phi'].to_numpy()
            locRz  = local['Rz'].to_numpy()
            locEn  = local['en'].to_numpy()
            cmsswEta = cmssw[:][cmssw_cols.index('cl3d_eta')]
            cmsswPhi = cmssw[:][cmssw_cols.index('cl3d_phi')]
            cmsswRz  = cmssw[:][cmssw_cols.index('cl3d_rz')]
            cmsswEn  = cmssw[:][cmssw_cols.index('cl3d_en')]
     
            if (len(locEta) != len(cmsswEta) or len(locPhi) != len(cmsswPhi) or
                len(locRz) != len(cmsswRz) or len(locEn) != len(cmsswEn)):
                print('Event Number: ', event_number)
                print(local, len(locEta))
                print(cmssw, len(cmsswEta))
            else:
                eThresh = 5.E-3
                for i in range(len(locEta)):
                    if (abs(locEta[i]-cmsswEta[i]) > eThresh or abs(locPhi[i]-cmsswPhi[i]) > eThresh or
                        abs(locRz[i]-cmsswRz[i]) > eThresh  or abs(locEn[i]-cmsswEn[i]) > eThresh):
                        print('Differences in event {}:'.format(event_number))
                        print('\tEta: {}'.format(locEta[i] - cmsswEta[i]))
                        print('\tPhi: {}'.format(locPhi[i] - cmsswPhi[i]))
                        print('\tRz: {}'.format(locRz[i] - cmsswRz[i]))
                        print('\tEn: {}'.format(locEn[i] - cmsswEn[i]))

@dataclass
class Datum:
    en: float
    eta: float
    phi: float
    pt: float
            
class EventsData:
    """
    Stores information of all events.
    """
    def __init__(self, group_labels):
        self.lab = group_labels
        self.ngroups = len(self.lab)

        self.en     = [[] for _ in range(self.ngroups)]
        self.eta    = [[] for _ in range(self.ngroups)]
        self.phi    = [[] for _ in range(self.ngroups)]
        self.pt     = [[] for _ in range(self.ngroups)]
        self.resen  = [[] for _ in range(self.ngroups)]
        self.reseta = [[] for _ in range(self.ngroups)]
        self.resphi = [[] for _ in range(self.ngroups)]
        self.respt  = [[] for _ in range(self.ngroups)]
        self.nclusters = []

    def append_groups(self, groups):
        """ The group order has to correspond to the `self.lab` parameter."""
        assert isinstance(groups, (list, tuple))
        assert all(isinstance(x, Datum) for x in groups)
        assert len(groups) == self.ngroups
        for ig,g in enumerate(groups):
            self.en[ig].append(g.en)
            self.eta[ig].append(g.eta)
            self.phi[ig].append(g.phi)
            self.pt[ig].append(g.pt)
            self.resen[ig].append(g.en/groups[0].en - 1.)
            self.reseta[ig].append(g.eta-groups[0].eta)
            self.resphi[ig].append(g.phi-groups[0].phi)
            self.respt[ig].append(g.pt/groups[0].pt - 1.)

    def append_singletons(self, nclusters):
        self.nclusters.append(nclusters)
                    
    def to_pandas(self):
        # convert class content to pandas dataframe
        df = pd.DataFrame()
        for ig in range(self.ngroups):
            df[self.lab[ig] + 'en']     = self.en[ig]
            df[self.lab[ig] + 'eta']    = self.eta[ig]
            df[self.lab[ig] + 'phi']    = self.phi[ig]
            df[self.lab[ig] + 'pt']     = self.pt[ig]
            df[self.lab[ig] + 'resen']  = self.resen[ig]
            df[self.lab[ig] + 'reseta'] = self.reseta[ig]
            df[self.lab[ig] + 'resphi'] = self.resphi[ig]
            df[self.lab[ig] + 'respt']  = self.respt[ig]
        df['nclusters'] = self.nclusters
        return df

class Collector:
    def __init__(self):
        with open(params.CfgPath, 'r') as afile:
            self.cfg = yaml.safe_load(afile)

    def collect_seed(self, pars, chain_mode, debug=False, **kw):
        """Statistics collector for ROI seeding results."""
        if debug:
            print('Running the seed validation...')
        mdef, mroi = self._decode_modes(chain_mode)

        outseed, outtc, outgen = ({} for _ in range(3))
        if mdef:
            outseed.update({'def': common.fill_path(self.cfg['seed']['SeedOut'],      **pars)})
            outtc.update(  {'def': common.fill_path(self.cfg['fill']['FillOutTcAll'], **pars)})
            outgen.update( {'def': common.fill_path(self.cfg['fill']['FillOutGenCl'], **pars)})
        if mroi:
            extra = common.seed_extra_name(self.cfg)
            outseed.update({'roi': common.fill_path(self.cfg['seed_roi']['SeedOut'] + extra,            **pars)})
            outtc.update(  {'roi': common.fill_path(self.cfg['roi'][self.cfg['seed_roi']['InputName']], **pars)})
            outgen.update( {'roi': common.fill_path(self.cfg['roi']['ROIclOut'],                        **pars)})
            
        sseed = {k: h5py.File(  v, mode='r') for k,v in outseed.items()}
        stc   = {k: pd.HDFStore(v, mode='r') for k,v in outtc.items()}
        sgen  = {k: pd.HDFStore(v, mode='r') for k,v in outgen.items()}
        
        data, ret = ({} for _ in range(2))
        for chain in outseed.keys():
            kseeds = sseed[chain].keys()
            ktc    = [x for x in stc[chain].keys() if 'central' not in x]
     
            dfgen = sgen[chain]['/df']
            data[chain] = {'genen': [], 'geneta': [], 'genphi': [], 'genpt': [],
                           'nseeds': [], 'nrois': [], 'nseedsperroi': []}

            assert len(kseeds)==len(ktc)
            search_ev = '{}_([0-9]{{1,7}})_ev'.format(kw['FesAlgo'])
            for ks,kt in zip(kseeds,ktc):
                evn = re.search(search_ev, ks).group(1)
                assert evn in kt
                    
                dftc = stc[chain][kt]
                dfseed = sseed[chain][ks]
                colsseed = list(dfseed.attrs['columns'])
         
                nseeds = len(dfseed[:][colsseed.index('seedEn')])
                if mroi and chain=='roi' and self.cfg['seed_roi']['InputName'] != 'NoROItcOut':
                    nrois = len(dftc['roi_id'].unique())
                else: # ROIs are ignored, there is only one "region of interest"
                    nrois = 1
                
                genEn, genEta, genPhi, genPt = self._get_gen_info(dfgen, int(evn))
                data[chain]['genen'].append(genEn)
                data[chain]['geneta'].append(genEta)
                data[chain]['genphi'].append(genPhi)
                data[chain]['genpt'].append(genPt)
                data[chain]['nseeds'].append(nseeds)
                data[chain]['nrois'].append(nrois)
                data[chain]['nseedsperroi'].append(float(nseeds) / nrois)
     
            ret[chain] = pd.DataFrame(data[chain])
            stc[chain].close()
            sgen[chain].close()
            sseed[chain].close()

        if len(outseed.keys())==1:
            return ret[list(outseed.keys())[0]]
        else:
            ret = self._postprocessing_multi_chain_seed(ret)
        return ret

    def collect_cluster(self, pars, chain_mode, debug=True, **kw):
        """Statistics collector for ROI clustering results."""
        mdef, mroi = self._decode_modes(chain_mode)
        outgen, outtc, outcl = ({} for _ in range(3))
        if mdef:
            outtc.update( {'def': common.fill_path(self.cfg['fill']['FillOutTcAll'],             **pars)})
            outgen.update({'def': common.fill_path(self.cfg['fill']['FillOutGenCl'],             **pars)})
            outcl.update( {'def':  common.fill_path(self.cfg['cluster']['ClusterOutValidation'], **pars)})

        if mroi:
            outgen.update( {'roi': common.fill_path(self.cfg['roi']['ROIclOut'], **pars)})
            if self.cfg['cluster']['ROICylinder']:
                outcl.update({'roi': common.fill_path(self.cfg['cluster']['ClusterOutValidationROI']  + '_cyl', **pars)})
                outtc.update({'roi': common.fill_path(self.cfg['roi']['ROIregionOut'], **pars)})
            else:
                outcl.update({'roi': common.fill_path(self.cfg['cluster']['ClusterOutValidationROI'], **pars)})
                # performance is compared to all TCs, not just the ones within the ROIs
                outtc.update({'roi': common.fill_path(self.cfg['roi']['NoROItcOut'], **pars)})

        sgen = {k: pd.HDFStore(v, mode='r') for k,v in outgen.items()}
        scl  = {k: pd.HDFStore(v, mode='r') for k,v in outcl.items()}
        stc  = {k: pd.HDFStore(v, mode='r') for k,v in outtc.items()}

        data, ret = ({} for _ in range(2))
        for chain in outgen.keys():
            kall, kcl = stc[chain].keys(), scl[chain].keys()
            search_ev = '{}_([0-9]{{1,7}})_'.format(kw['FesAlgo'])
     
            # remove ROI keys where no cluster exists
            to_remove = []
            for ik,k in enumerate(kall):
                evn = re.search(search_ev, k).group(1)
                if '/ThresholdDummyHistomaxnoareath20_' + evn + '_ev' not in kcl:
                    to_remove.append(k)
            for i in to_remove:
                kall.remove(i)
     
            kall = [x for x in kall if 'central' not in x]
            assert len(kall) == len(kcl)
         
            ntotal = len(kcl)

            dfgen = sgen[chain]['/df']
         
            c_cl1, c_cl2 = 0, 0
         
            evData = EventsData(
                group_labels=['gen', 'tcall',
                              *['tc'+str(t).replace('.','p') for t in self.cfg['valid_cluster']['tcDeltaRthresh']],
                              'cl'])
            for kall, kcl in zip(kall, kcl):
                dfcl = scl[chain][kcl]
                if dfcl.empty:
                    continue
                evn = re.search(search_ev, kall).group(1)
                if evn not in kcl:
                    print('Event {} was not in the cluster dataset.'.format(evn))
                    continue
                dfall = stc[chain][kall]
                
                genEn, genEta, genPhi, genPt = self._get_gen_info(dfgen, int(evn))
                genDatum = Datum(genEn, genEta, genPhi, genPt)
         
                # filter noise around shower center
                dfall['dR'] = common.deltaR(genEta, genPhi, dfall.tc_eta, dfall.tc_phi)
                wght = self._weight_array(dfall[['tc_eta','tc_phi']], dfall.tc_pt)
                datumAll = Datum(dfall.tc_energy.sum(), wght.tc_eta, wght.tc_phi, dfall.tc_pt.sum())
                
                datum = []
                for thresh in self.cfg['valid_cluster']['tcDeltaRthresh']:
                    tmp_df = dfall[dfall.dR<thresh]
                    wght_tmp = self._weight_array(tmp_df[['tc_eta', 'tc_phi']], tmp_df.tc_pt)
                    datum.append(Datum(tmp_df.tc_energy.sum(), wght_tmp.tc_eta, wght_tmp.tc_phi, tmp_df.tc_pt))
         
                clEta = dfcl['eta'].to_numpy()
                clPhi = dfcl['phi'].to_numpy()
                clEn  = dfcl['en'].to_numpy()
                clPt  = dfcl['pt'].to_numpy()
                clEnMax = max(clEn) # ignoring the lowest energy clusters when there is a splitting
                clPtMax = max(clPt)
                index_max_energy_cl = np.where(clEn==clEnMax)[0][0]
                index_max_pt_cl = np.where(clPt==clPtMax)[0][0]
                assert type(index_max_energy_cl) == np.int64 or type(index_max_energy_cl) == np.int32
                assert type(index_max_pt_cl) == np.int64 or type(index_max_pt_cl) == np.int32
                clDatum = Datum(clEnMax, clEta[index_max_energy_cl], clPhi[index_max_energy_cl], clPtMax)
     
                # self._multiplicity_test(clEn)
                evData.append_groups([genDatum, datumAll, *datum, clDatum])
                evData.append_singletons(nclusters=len(clEn))
         
            clrat1 = float(c_cl1) / ntotal
            clrat2 = float(c_cl2) / ntotal
         
            if debug:
                print()
                print('Cluster ratio singletons: {} ({})'.format(clrat1, c_cl1))
                print('Cluster ratio splits: {} ({})'.format(clrat2, c_cl2))
                
            ret[chain] = evData.to_pandas()
        
            scl[chain].close()
            stc[chain].close()
            sgen[chain].close()

        if len(outgen.keys())==1:
            return ret[list(outgen.keys())[0]]
        else:
            ret = self._postprocessing_multi_chain_cluster(ret)
        return ret

    def _decode_modes(self, mode):
        assert mode in ('default', 'roi', 'both')
        if mode == 'default':
            mode_def = True
            mode_roi = False
        elif mode == 'roi':
            mode_def = False
            mode_roi = True
        elif mode == 'both':
            mode_def = True
            mode_roi = True
        return mode_def, mode_roi

    def _get_gen_info(self, dfgen, event):
        genEvent = dfgen[dfgen.event==event]
     
        genEn  = genEvent['gen_en'].to_numpy()
        genEta = genEvent['gen_eta'].to_numpy()
        genPhi = genEvent['gen_phi'].to_numpy()
        genPt  = genEvent['gen_pt'].to_numpy()
        
        #when the cluster is split we will have two rows
        if len(genEn) > 1:
            assert genEn[1]  == genEn[0]
            assert genEta[1] == genEta[0]
            assert genPhi[1] == genPhi[0]
            assert genPt[1]  == genPt[0]
        genEn  = genEn[0]
        genEta = genEta[0]
        genPhi = genPhi[0]
        genPt  = genPt[0]
     
        return genEn, genEta, genPhi, genPt

    def _multiplicity_test(self, en_cluster):
        """test if cluster multiplicity affects results (answer: no!)"""
        en_max = sum(en_cluster)
        if len(en_cluster) > 1:
            print(en_max / sum(en_cluster))

    def _postprocessing_multi_chain_cluster(self, df):
        """
        Merge cluster results from multiple chains into single output dataframe.
        Currenty focusing on the 'default' and 'roi' chains.
        """
        new_df = {}
        new_df.update({'genen':    df['def']['genen'],
                       'geneta':   df['def']['geneta'],
                       'genphi':   df['def']['genphi'],
                       'genpt':   df['def']['genpt'],
                       'tcallen':  df['def']['tcallen'],
                       'tcalleta': df['def']['tcalleta'],
                       'tcallphi': df['def']['tcallphi'],
                       'tcallpt': df['def']['tcallpt'],
                       'tcallresen':  df['def']['tcallresen'],
                       'tcallreseta': df['def']['tcallreseta'],
                       'tcallresphi': df['def']['tcallresphi'],
                       'tcallrespt': df['def']['tcallrespt'],
                       })

        for suf in ('def', 'roi'):
            new_df.update({'clen_'+suf:  df[suf]['clen'],
                           'cleta_'+suf: df[suf]['cleta'],
                           'clphi_'+suf: df[suf]['clphi'],
                           'clpt_'+suf: df[suf]['clpt'],
                           
                           'clresen_'+suf:  df[suf]['clresen'],
                           'clreseta_'+suf: df[suf]['clreseta'],
                           'clresphi_'+suf: df[suf]['clresphi'],
                           'clrespt_'+suf: df[suf]['clrespt'],

                           'nclusters_'+suf: df['def']['nclusters']})

        for t in self.cfg['valid_cluster']['tcDeltaRthresh']:
            for binvar in ('en', 'eta', 'phi', 'pt'):
                ktc    = 'tc'+str(t).replace('.','p') + binvar
                new_df.update({ktc: df['def'][ktc]})
                ktcres = 'tc'+str(t).replace('.','p') + 'res' + binvar
                new_df.update({ktcres: df['def'][ktcres]})

        # Note: the output dataframe likely contains NaN values
        # due to different number of events in the two chains
        return pd.DataFrame(new_df)

    def _postprocessing_multi_chain_seed(self, df):
        """
        Merge seed results from multiple chains into single output dataframe.
        Currenty focusing on the 'default' and 'roi' chains.
        """
        new_df = pd.DataFrame({
            'nrois_def':        df['def']['nrois'],
            'nseeds_def':       df['def']['nseeds'],
            'nseedsperroi_def': df['def']['nseedsperroi'],
            'nrois_roi':        df['roi']['nrois'],
            'nseeds_roi':       df['roi']['nseeds'],
            'nseedsperroi_roi': df['roi']['nseedsperroi'],
            'genen':            df['def']['genen'],
            'geneta':           df['def']['geneta'],
            'genphi':           df['def']['genphi'],
            'genpt':            df['def']['genpt'],
            })
        return new_df

    def _weight_array(self, arr, wght):
        """Weights array `arr` by weight array `wght`."""
        return arr.multiply(wght, axis=0).sum() / wght.sum()
    
if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='ROI chain validation standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()

    valid_d = params.read_task_params('valid_roi')
    stats_collector_roi(vars(FLAGS), mode='resolution', **valid_d)
