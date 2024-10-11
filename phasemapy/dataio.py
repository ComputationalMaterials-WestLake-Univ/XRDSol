import json

from monty.json import MontyDecoder
from scipy.constants import h, c, e
import numpy as np

# suppose this is after raw fitting
from scipy.interpolate import interp1d


class InstanceData:

    def __init__(self, chemsys, photon_e, log_q, sample_xrd, comp_dict):
        self.chemsys = sorted(chemsys)
        self.log_q = log_q
        self.sample_xrd = sample_xrd
        self.comp_dict = comp_dict
        self.photon_e = photon_e
        self.wavelength = 1e10 * h * c / (self.photon_e * e)  # in A

    @classmethod
    def from_file(cls, instance_file, chemsys, photon_e):

        with open(instance_file) as f:
            lines = f.readlines()
        amps = []
        comp_dict = {}

        for line in lines:
            head = line.split('=')[0]
            if not head:
                continue
            elif head == 'Q':
                q = np.array(list(map(float, line.split('=')[1].split(','))))
            elif head in chemsys:
                comp_dict[head] = np.array(list(map(float, line.split('=')[1].split(','))))
            elif head[0] == 'I' and len(head) > 1:
                amp = np.array(list(map(float, line.split('=')[1].split(','))))
                amps.append(amp)


        sample_xrd = np.array(amps)
        return InstanceData(chemsys, photon_e, np.log(q), sample_xrd, comp_dict)

    # added by Dongfang Yu
    @classmethod
    def from_json (cls, instancefile, chemsys, photon_e,XRD_mask=None):
        with open(f'{instancefile}.json') as f:
            Instance_data = json.load(f, cls=MontyDecoder)
        sample_xrd = [Instance_data[i]['Instance_data_info']['sample_xrd'] for i in range(len(Instance_data))]
        comp_dict = {}
        comp_dict[chemsys[0]] = np.array([Instance_data[i]['Instance_data_info']['comp'][0] for i in range(len(Instance_data))])
        comp_dict[chemsys[1]] = np.array([Instance_data[i]['Instance_data_info']['comp'][1] for i in range(len(Instance_data))])
        comp_dict[chemsys[2]] = np.array([Instance_data[i]['Instance_data_info']['comp'][2] for i in range(len(Instance_data))])
        q = np.array(Instance_data[0]['Instance_data_info']['q'])
        if XRD_mask:
            mask = q < XRD_mask[0]
            q[mask] = XRD_mask[0]
            for i in range(len(Instance_data)):
                np.array(sample_xrd[i])[mask] = 0.001
            mask = q > XRD_mask[1]
            q[mask] = XRD_mask[1]
            for i in range(len(Instance_data)):
                np.array(sample_xrd[i])[mask] = 0.001

        return InstanceData(chemsys, photon_e, np.log(q), np.array(sample_xrd), comp_dict)



    def renormalize(self, norm):
        self.sample_xrd = self.sample_xrd / np.sum(self.sample_xrd, axis=1, keepdims=True) * norm

    def normalize(self):
        self.sample_xrd = [_/np.max(_) for _ in self.sample_xrd]


    @property
    def qmin(self):
        return self.q[0]

    @property
    def qmax(self):
        return self.q[-1]

    @property
    def sample_comp(self):
        comp = np.array([self.comp_dict[el] for el in self.chemsys]).T
        comp /= np.sum(comp, axis=1, keepdims=True)
        return comp

    @property
    def dim(self):
        return len(self.chemsys)

    @property
    def sample_num(self):
        return self.sample_comp.shape[0]

    @property
    def twotheta(self):
        return np.arcsin(np.array(self.q) / np.pi / 2 / 10 * self.wavelength / 2) / 2 / np.pi * 360 * 2

    @property
    def q(self):
        return np.exp(self.log_q)

    def resample_xrd(self, resample_density):
        new_log_q = np.linspace(self.log_q[0], self.log_q[-1], resample_density)
        new_amps = []
        for i in range(self.sample_num):
            f = interp1d(self.log_q, self.sample_xrd[i])
            new_amp = f(new_log_q)
            new_amps.append(new_amp)
        new_amps = np.array(new_amps)
        #new_q = np.exp(new_log_q)
        return InstanceData(self.chemsys, self.photon_e, new_log_q, new_amps, self.comp_dict)

    #
# class BasisLoader:
#
#     def __init__(self, filename, chemsys, oxide_system):
#         with open(filename)as f:
#             self.group_entries = json.load(f, cls=MontyDecoder)
#         self.basis_num = len(self.group_entries)
#         self.chemsys = sorted(chemsys)
#         self.oxide_system = oxide_system


# class PhaseMapperSolver():
#     def __init__(self, chemsys, photon_e, oxide_system):
#         self.chemsys = sorted(chemsys)
#         self.wavelength = 1e10 * h * c / (photon_e * e)  # in A
#         self.els = chemsys + ['O'] if oxide_system else chemsys
#         self.dim = len(chemsys)
#         pass
#

# oxide_system = True
# photon_e = 13e3
#
#
#
#
# WINDOW,XRD_MATCH_CUTOFF = int(0.9/(qmax-qmin)*1000),11


# two_theta_range = np.arcsin(np.array([qmin, qmax]) / np.pi / 2 / 10 * WAVELENGTH / 2) / 2 / np.pi * 360 * 2

# solver = PhaseMapperSolver(chemsys, photon_e, oxide_system)
        return cls(entry['instance_info']['name'],entry['instance_info']['instance_comp'],entry['instance_info']['instance_xrd'],entry['instance_info'][ 'name'],entry['instance_info']['entry_id'],entry['instance_info']['name'],entry['instance_info']['q'],entry['instance_info']['amp'],entry['instance_info']['crystal_system'])