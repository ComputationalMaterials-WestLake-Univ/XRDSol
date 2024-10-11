import os
import re
import xml.etree.cElementTree as ET
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core import Element, Composition
from scipy.cluster.hierarchy import fcluster, linkage
from monty.json import MSONable
from pymatgen.io.cif import CifParser
from scipy.ndimage import gaussian_filter1d
from pymatgen.core import Lattice, Structure, PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher
# import qmpy_rester as qr

SITE_OCC_TOL = 0.15
COMP_TOL = 0.1
pd.options.display.width = 1000
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class ICDDEntry(MSONable):
    def __init__(self, entry_id, chemical_formula, empirical_formula, status, quality_mark, pressure_temperature, database_comments,
                 spgr, common_name, cross_refs, xstal_system, chemical_name, xtldx, dm, rep_struc, cell_vol, redcellvol, z, xtla, xtlb, xtlc,
                 xtlal, xtlbe, xtlga, structure, data, hkl, reference, name=None, leader=None, stability=None,  material_id=None,
                 energy_per_atom=None,):
        # # composition,
        # name, atomic_coords, has_struct, struct, ):
        self.entry_id = entry_id
        self.chemical_formula = "".join(chemical_formula.split())
        self.empirical_formula = empirical_formula
        self.composition = Composition(empirical_formula)
        self.status = status
        self.quality_mark = quality_mark
        self.pressure_temperature = pressure_temperature
        self.database_comments = database_comments
        self.spgr = spgr
        self.name = name if name else self.chemical_formula
        self.common_name = common_name
        self.cross_refs = cross_refs
        self.structure = structure
        self.data = data if data else {}
        self.hkl = hkl if hkl else []
        self.leader = leader if leader else self.entry_id
        self.stability = stability
        self.xstal_system = xstal_system
        self.chemical_name = chemical_name
        self.xtldx = xtldx
        self.dm = dm
        self.rep_struc = rep_struc
        self.cell_vol = float(cell_vol) if cell_vol else []
        self.redcellvol = float(redcellvol) if redcellvol else []
        self.a = float(xtla) if xtla else []
        self.b = float(xtlb) if xtlb else []
        self.c = float(xtlc) if xtlc else []
        self.alpha = float(xtlal) if xtla else []
        self.beta = float(xtlbe) if xtlbe else []
        self.gamma = float(xtlga) if xtlga else []
        self.z = round(float(z)*self.redcellvol/self.cell_vol) if self.redcellvol and self.cell_vol and z else None
        #self.z = round(self.cell_vol/self.redcellvol) if self.redcellvol and self.cell_vol else None
        self.material_id = material_id
        self.energy_per_atom = energy_per_atom
        self.reference = reference



    @classmethod
    def from_icdd_xml(cls, xmlfile):
        root = ET.parse(xmlfile).getroot()[0]
        reference = root.find('.//reference').text
        # reference = list(reference)
        # print(reference)

        labels = ['pdf_number', 'chemical_formula', 'empirical_formula', 'status', 'quality_mark', 'pressure_temperature',
                  'database_comments','spgr', 'common_name', 'cross_ref_pdf_numbers','xstal_system','chemical_name',
                  'xtldx','dm','rep_struc','cell_vol', 'redcellvol', 'z', 'redcell_a','redcell_b','redcell_c','redcell_al','redcell_be', 'redcell_ga']

        def get_xml_value(root, key):
            el = root.findall(key)
            return el[0].text if el else ''

        params = [get_xml_value(root, label) for label in labels]

        index = labels.index('cross_ref_pdf_numbers')
        former = get_xml_value(root, 'former_pdf_number')
        if former:
            params[index] = ', '.join(params[index].split(', ') + [f"{_} (Former)" for _ in former.split(', ')])
            if params[index][0] == ',':
                params[index] = params[index][2:]
        params[index] = re.findall('\d{2}-\d{3}-\d{4}', params[index])

        # atomic_coords_el = root[0].findall('atomic_coords')
        atomic_coords_el = root.findall('.//atomic_coord')
        # print(atomic_coords_el)
        # atomic_coords = 'Y' if atomic_coords_el else 'N'
        if atomic_coords_el:
            atomic_coords = []
            species = []
            sofs = []
            for atomic_coord in root.findall('.//atomic_coord'):
                atom = atomic_coord.find('atom').text
                x = float(atomic_coord.find('x').text)
                y = float(atomic_coord.find('y').text)
                z = float(atomic_coord.find('z').text)
                sof = float(atomic_coord.find('sof').text)
                atomic_coords.append([x, y, z,])
                species.append(atom)
                if sof > 1:
                    sof = 0.96369
                sofs.append(sof)
            species = [item[:1] + item[1:].lower() for item in species]
            species_sofs = [{species[i]: sofs[i]} for i in range(len(species))]
            # print(params[0])
            # print(species_sofs )
            # print(atomic_coords)
            # print(params[-6])

            struct = Structure(
                lattice=Lattice.from_parameters(a=float(params[-6]),
                                        b=float(params[-5]),
                                        c=float(params[-4]),
                                        alpha=float(params[-3]),
                                        beta=float(params[-2]),
                                        gamma=float(params[-1]),),
                species=species_sofs, coords=atomic_coords, coords_are_cartesian=False)
        else:
            struct = None
        root = ET.parse(xmlfile).getroot()[1][2]
        theta = np.array([float(_.text) for _ in root.iter('theta')])
        try:
            h = np.array([int(_.text) for _ in root.iter('h')])
            k = np.array([int(_.text) for _ in root.iter('k')])
            l = np.array([int(_.text) for _ in root.iter('l')])
            hkl = list(map(lambda x, y, z: (x, y, z), h, k, l))
        except Exception as e:
            hkl = []
        intensity = np.array(
            [float(''.join(list(filter(str.isdigit, _.text)))) for _ in root.iter('intensity') if _.text is not '\n'])
        element_counts = Counter(theta)
        multi_counts = [element_counts[element] for element in theta]
        intensity = np.array(intensity) / np.array(multi_counts)
        intensity = intensity / np.max(intensity)
        d = (theta, intensity)
        data = {}
        data['xrd'] = d

        stability = None


        return cls(*params, struct, data, hkl, reference, stability)

    @classmethod
    def from_icsd_cif(cls, cif_file):
        struct = CifParser(cif_file, occupancy_tolerance=2).get_structures(primitive=False)[0]
        entry_id = cif_file.split('Code')[1].split('.')[0]
        chemical_formula = Composition(struct.formula).reduced_formula
        empirical_formula = chemical_formula
        common_name = chemical_formula
        structure = struct
        status = quality_mark = pressure_temperature = database_comments = spgr = cross_refs = name = leader = data = hkl = stability = None

        return cls(entry_id, chemical_formula, empirical_formula, status, quality_mark, pressure_temperature, database_comments,
                   spgr, common_name, cross_refs, structure, name, leader, data, hkl, stability)

    @property
    def has_icsd_ref(self):
        if re.findall('ICSD Collection Code: [0-9]+', self.database_comments):
            return True
        # elif re.findall('01-\d{3}-\d{4} \(Former\)', self.cross_ref):
        #     return True
        else:
            return False
        # return ('ICSD' in self.database_comments)

    @property
    def former_icsd_ref(self):
        return re.findall('01-\d{3}-\d{4} \(Former\)', self.cross_refs)

    @property
    def qm_rank(self):
        if self.quality_mark is None:
            return 0
        else:
            return {'Star': 2, 'Indexed': 1, 'Prototyping': 1, 'Calculated': 1, 'Hypothetical': 0, 'Blank': 0,
                    'Low-Precision': 0}[self.quality_mark]

    @property
    def status_rank(self):
        if self.status is None:
            return 0
        else:
            return {'Primary': 2, 'Alternate': 1, 'Deleted': 0}[self.status]

    @property
    def rank(self):
        if self.structure is None:
            rank = (self.status_rank * 1 + self.qm_rank * 10, self.entry_id)
        else:
            rank = (100 + 50*(self.structure.is_ordered) + self.status_rank * 1 + self.qm_rank * 10, self.entry_id)
        return rank

    @property
    def icsd_id(self):
        icsd_entry_id = ''
        if self.has_icsd_ref:
            icsd_label = re.findall('ICSD Collection Code: [0-9]+', self.database_comments)
            icsd_entry_id = int(icsd_label[0].split(':')[-1])
        return icsd_entry_id

    def __eq__(self, other):
        return self.entry_id == other.entry_id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.entry_id)


class ICDDEntryPreprocessor:
    def __init__(self, all_entries, chemsys, oxide_system):
        self.all_entries = all_entries
        self.chemsys = chemsys
        self.oxide_system = oxide_system

    @property
    def entries(self):
        return [_ for _ in self.all_entries if _.leader == _.entry_id]

    @property
    def ordered_entries(self):
        return [_ for _ in self.entries if _.structure is not None and _.structure.is_ordered]


    def merge_by_cross_ref(self):

        n = len(self.all_entries)
        nn_matrix = np.ones([n, n])
        entry_ids = [_.entry_id for _ in self.entries]

        for i, e in enumerate(self.all_entries):
            for sibling in e.cross_refs:
                try:
                    j = entry_ids.index(sibling)
                    nn_matrix[i, j] = 0
                    nn_matrix[j, i] = 0
                except ValueError as e:
                    pass

        link = linkage(nn_matrix[np.triu_indices(n, k=1)])
        clusters = fcluster(link, 0.5, criterion='distance')
        clusters = clusters - 1

        groups = [[] for _ in set(clusters)]

        for i, e in zip(clusters, self.entries):
            groups[i].append(e)
        for g in groups:
            g.sort(key=lambda x: x.rank, reverse=True)

        for group in groups:
            for e in group:
                e.leader = group[0].entry_id

    def process_disorder(self):
        global SITE_OCC_TOL
        for e in [_ for _ in self.entries if _.structure is not None and not _.structure.is_ordered]:
            remove_index = []
            for i, site in enumerate(e.structure):
                if site.is_ordered:
                    continue
                if max(site.species.values()) <= SITE_OCC_TOL:
                    remove_index.append(i)
                    continue
                for el, amt in site.species.items():
                    if amt >= 1 - SITE_OCC_TOL:
                        site.species = el
                        break
            e.structure.remove_sites(remove_index)
            e.composition = e.structure.composition
            if '.' in e.name and ('.' not in e.structure.composition.reduced_formula):
                e.name = e.structure.composition.reduced_formula
        if self.oxide_system:
            exclude_entry_indexes = [_ for _ in self.all_entries if e.composition[Element('O')] <= 1e-4]
            self.all_entries = [e for e in self.all_entries if e.composition[Element('O')] > 0]

    def process_frac_name(self):
        multiplier = np.array([1, 2, 3, 4, 5, 6, 11])
        global COMP_TOL

        def reduce_comp(composition, cutoff=50):
            if sum(composition.values()) <= cutoff:
                return composition
            comps = [composition + Composition({el: 1}) for el in composition.keys()]
            comps += [composition - Composition({el: 1}) for el in composition.keys()]
            comps.append(composition)
            comps = [_.reduced_composition for _ in comps]
            comps.sort(key=lambda x: sum(x.values()))
            return comps[0]

        def get_interger_name(composition):
            els = list(composition.keys())
            amts = [composition[el] for el in els]
            amts = np.array(amts)
            frac = np.abs(amts - np.around(amts))
            if max(frac) <= COMP_TOL:
                amts = np.around(amts)
                return Composition.from_dict({el: amt for el, amt in zip(els, amts)})

            prod = np.array([list(composition.values())]) * multiplier.reshape(-1, 1)
            residual = np.max(np.abs(prod - np.around(prod)), axis=1) / multiplier
            if np.min(residual) <= COMP_TOL:
                multi = multiplier[np.argmin(residual)]
                comp_dict = composition.as_dict()
                for el, amt in comp_dict.items():
                    comp_dict[el] = np.around(amt * multi)
                composition = Composition.from_dict(comp_dict).reduced_composition

                while reduce_comp(composition) != composition:
                    composition = reduce_comp(composition)
            return composition.reduced_composition

        for e in self.entries:
            if '.' in e.name:
                comp1 = get_interger_name(e.composition)
                e.composition = comp1.reduced_composition
                e.name = e.composition.reduced_formula

    def theta_to_q(self):
        for e in self.entries:
            if len(e.entry_id) == 11:
                q_vectors = 4 * np.pi / 1.54056 * np.sin(np.radians(e.data['xrd'][0]) / 2) * 10
                data = (q_vectors, e.data['xrd'][1])
                e.data['xrd'] = data


    def get_xrd(self, leader_only=True):
        if leader_only:
            entries = self.entries
        else:
            entries = self.all_entries

        for e in entries:
            if e.structure is not None:
                xrdcal = XRDCalculator()
                s = e.structure
                xrd = xrdcal.get_pattern(s, scaled=False)
                d = np.array(xrd.d_hkls)
                amplitude = np.array(xrd.y) / s.volume ** 2  # This is irrelevant of volume
                amplitude = amplitude / np.max(amplitude)
                q_vectors = 4 * np.pi / (2 * d) * 10
                data = (q_vectors, amplitude)
                e.data['xrd'] = data
                e.hkl = [xrd.hkls[i][0]['hkl'] for i in range(len(xrd.hkls))]


    def merge_by_structure(self, plot=False):
        sm = StructureMatcher(comparator=ElementComparator(), attempt_supercell=True)
        names = set([_.name for _ in self.entries])
        name_dict = {name: [_ for _ in self.entries if _.name == name] for name in names}

        for name, polymorphs in name_dict.items():
            if len(polymorphs) > 1:
                struct_match_tri = []
                for i, j in combinations(range(len(polymorphs)), 2):
                    e1, e2 = polymorphs[i], polymorphs[j]
                    s1, s2 = e1.structure, e2.structure
                    if 0.9 < (s1.volume / len(s1)) / (s2.volume / len(s2)) < 1.1:
                        if sm.fit(s1, s2):
                            struct_match_tri.append(0)
                        else:
                            struct_match_tri.append(1)
                    else:
                        struct_match_tri.append(1)
                links = linkage(struct_match_tri)
                clusters = fcluster(links, 0.5, 'distance')

                for e, c in zip(polymorphs, clusters):
                    e.color = f'C{c - 1}'
                polymorphs.sort(key=lambda x: (x.color, x.rank, x.entry_id), reverse=True)

                if plot:
                    # xrdcal=XRDCalculator()
                    print(name, len(polymorphs), len(set(clusters)), [_.entry_id for _ in polymorphs])
                    fig, axes = plt.subplots(ncols=1, nrows=len(polymorphs), sharex=True)
                    for ax, e in zip(axes, polymorphs):
                        # p = xrdcal.get_pattern(e.structure)
                        ax.stem(*e.data['xrd'], use_line_collection=True, markerfmt=" ", linefmt=e.color,
                                label=e.entry_id)
                        ax.legend(loc=1)
                    plt.show()

                color = polymorphs[0].color
                leader = polymorphs[0].entry_id
                for e in polymorphs[1:]:
                    if e.color == color:
                        for oe in self.all_entries:
                            if oe.leader == e.entry_id:
                                oe.leader = leader
                    else:
                        color = e.color
                        leader = e.entry_id

    def merge_by_polymorph(self, bin_number, gaussian_filter, R_cutoff, plot=False):
        names = set([_.name for _ in self.entries])
        name_dict = {name: [_ for _ in self.entries if _.name == name] for name in names}

        def smooth_hist(q, amp, bins):
            hist, bin_edges = np.histogram(q, bins=bins, weights=amp)
            smoothed = gaussian_filter1d(hist, gaussian_filter)
            return smoothed

        for name, polymorphs in name_dict.items():
            if len(polymorphs) > 1:
                bins = np.linspace(min([_.data['xrd'][0][0] for _ in polymorphs]) - 0.01,
                                   max([_.data['xrd'][0][-1] for _ in polymorphs]) + 0.01, bin_number)
                xrd_match_tri = []
                for i, j in combinations(range(len(polymorphs)), 2):
                    e1, e2 = polymorphs[i], polymorphs[j]
                    s1, s2 = e1.structure, e2.structure
                    if 0.9 < (s1.volume / len(s1)) / (s2.volume / len(s2)) < 1.1:
                        ratio = (s1.volume / len(s1) / (s2.volume / len(s2))) ** (1 / 3)
                        q, amp = e1.data['xrd'][0], e1.data['xrd'][1]
                        smooth_xrds_i = smooth_hist(q, amp, bins)
                        q, amp = e2.data['xrd'][0], e2.data['xrd'][1]
                        smooth_xrds_j = smooth_hist(q / ratio, amp, bins)
                        smooth_xrds_i = smooth_xrds_i / np.max(smooth_xrds_i) * 100
                        smooth_xrds_j = smooth_xrds_j / np.max(smooth_xrds_j) * 100
                        abs_diff = np.abs(smooth_xrds_i - smooth_xrds_j)
                        R = np.sqrt(np.sum(abs_diff ** 2) / max(np.sum(smooth_xrds_i ** 2), np.sum(smooth_xrds_j ** 2)))
                        match = 0 if R < R_cutoff else 1
                    else:
                        match = 1
                    xrd_match_tri.append(match)
                links = linkage(xrd_match_tri)
                clusters = fcluster(links, 0.5, 'distance')

                for e, c in zip(polymorphs, clusters):
                    e.color = f'C{c - 1}'
                polymorphs.sort(key=lambda x: (x.color, x.rank, x.entry_id), reverse=True)

                if plot:
                    print(name, len(polymorphs), len(set(clusters)), [_.entry_id for _ in polymorphs])
                    fig, axes = plt.subplots(ncols=1, nrows=len(polymorphs), sharex=True)
                    for ax, e in zip(axes, polymorphs):
                        ax.stem(*e.data['xrd'], use_line_collection=True, markerfmt=" ", linefmt=e.color,
                                label=e.entry_id)
                        ax.legend(loc=1)
                    plt.show()

                color = polymorphs[0].color
                leader = polymorphs[0].entry_id
                for e in polymorphs[1:]:
                    if e.color == color:
                        for oe in self.all_entries:
                            if oe.leader == e.entry_id:
                                oe.leader = leader
                    else:
                        color = e.color
                        leader = e.entry_id

    def merge_by_xrd(self, bin_number, gaussian_filter, R1_cutoff, R2_cutoff, plot=False):
        def smooth_hist(q, amp, bins):
            hist, bin_edges = np.histogram(q, bins=bins, weights=amp)
            smoothed = gaussian_filter1d(hist, gaussian_filter)
            return smoothed

        def merge(xrd_match_lst):
            sts = [set(l) for l in xrd_match_lst]
            i = 0
            while i < len(sts):
                j = i + 1
                while j < len(sts):
                    if len(sts[i].intersection(sts[j])) > 0:
                        sts[i] = sts[i].union(sts[j])
                        sts.pop(j)
                    else:
                        j += 1
                i += 1
            lst = [list(s) for s in sts]
            return lst

        # bins = np.linspace(min([_.data['xrd'][0][0] for _ in self.entries]) - 0.01,
        #                    max([_.data['xrd'][0][-1] for _ in self.entries]) + 0.01, bin_number)
        xrd_match = []

        for i, j in combinations(range(len(self.entries)), 2):
            e1, e2 = self.entries[i], self.entries[j]
            if e1.structure and e2.structure:
                bins = np.linspace(min([_.data['xrd'][0][0] for _ in [e1, e2]]) - 1,
                                   max([_.data['xrd'][0][-1] for _ in [e1, e2]]) + 1, bin_number)
                R_cutoff = R1_cutoff
            else:
                bins = np.linspace(min([_.data['xrd'][0][0] for _ in [e1, e2]]) - 1,
                                   min([_.data['xrd'][0][-1] for _ in [e1, e2]]) + 1, bin_number)
                R_cutoff = R2_cutoff
            q, amp = e1.data['xrd'][0], e1.data['xrd'][1]
            smooth_xrds_i = smooth_hist(q, amp, bins)
            q, amp = e2.data['xrd'][0], e2.data['xrd'][1]
            smooth_xrds_j = smooth_hist(q, amp, bins)
            smooth_xrds_i = smooth_xrds_i / np.max(smooth_xrds_i) * 100
            smooth_xrds_j = smooth_xrds_j / np.max(smooth_xrds_j) * 100

            abs_diff = np.abs(smooth_xrds_i - smooth_xrds_j)
            R = np.sqrt(np.sum(abs_diff ** 2) / max(np.sum(smooth_xrds_i ** 2), np.sum(smooth_xrds_j ** 2)))

            e1_comp = np.array([e1.composition[Element(el)] for el in self.chemsys])
            e1_comp = e1_comp / np.sum(e1_comp)
            e2_comp = np.array([e2.composition[Element(el)] for el in self.chemsys])
            e2_comp = e2_comp / np.sum(e2_comp)
            comp_distance = np.linalg.norm(e1_comp - e2_comp, ord=1)

            if R < R_cutoff and comp_distance < 1:

                xrd_match.append([i, j])
                if plot:
                    # print([R], [e1.name, e1.entry_id], [e2.name, e2.entry_id])
                    plt.plot(bins[0:-1], smooth_xrds_i, label=f'{e1.name}_{e1.entry_id}')
                    plt.plot(bins[0:-1], smooth_xrds_j, label=f'{e2.name}_{e2.entry_id}')
                    plt.title(f'{R}_unshift')
                    plt.legend()
                    plt.savefig(f'data/{e1.entry_id}_{e2.entry_id}.jpg', bbox_inches='tight')
                    plt.show()


            else:
                for shift in arange(-0.5, 0.5, 0.01):
                    q, amp = e1.data['xrd'][0] + shift, e1.data['xrd'][1]
                    smooth_xrds_i = smooth_hist(q, amp, bins)
                    q, amp = e2.data['xrd'][0], e2.data['xrd'][1]
                    smooth_xrds_j = smooth_hist(q, amp, bins)
                    smooth_xrds_i = smooth_xrds_i / np.max(smooth_xrds_i) * 100
                    smooth_xrds_j = smooth_xrds_j / np.max(smooth_xrds_j) * 100

                    abs_diff = np.abs(smooth_xrds_i - smooth_xrds_j)
                    R = np.sqrt(np.sum(abs_diff ** 2) / max(np.sum(smooth_xrds_i ** 2), np.sum(smooth_xrds_j ** 2)))

                    if R < R_cutoff:
                        xrd_match.append([i, j])
                        if plot:
                            # print([R], [e1.name, e1.entry_id], [e2.name, e2.entry_id])
                            plt.title(f'{R}_shift')
                            plt.plot(bins[0:-1], smooth_xrds_i, label=f'{e1.name}_{e1.entry_id}')
                            plt.plot(bins[0:-1], smooth_xrds_j, label=f'{e2.name}_{e2.entry_id}')
                            plt.legend()
                            plt.show()
                        break
        groups_index = merge(xrd_match)
        for i in range(len(groups_index)):
            groups_index[i] = np.sort(groups_index[i]).tolist()

        # groups = groups_index
        # for k in range(len(groups_index)):
        #     for i in range(len(groups_index[k])):
        #         groups[k][i] = self.entries[groups_index[k][i]]

        groups_id = []
        groups = []
        for k in range(len(groups_index)):
            en_id = []
            en = []
            for i in groups_index[k]:
                e = self.entries[i]
                en_id.append(e.entry_id)
                en.append(e)
            groups_id.append(en_id)
            groups.append(en)


        # for m in range(len(groups_index)):
        #     for i in groups_index[m][1:]:
        #         self.all_entries[i].leader = self.all_entries[groups_index[m][0]].entry_id

        for i in range(len(groups)):
            groups[i].sort(key=lambda x: x.rank[0], reverse=True)
            print(i,groups[i][0].rank)
            leader = groups[i][0].entry_id
            for e in groups[i][1:]:
                print(e.rank)
                e.leader = leader


        return groups_index, groups_id

    def merge_by_icsd(self, bin_number, gaussian_filter, R_cutoff, icdd_entries, icsd_entries):
        def smooth_hist(q, amp, bins):
            hist, bin_edges = np.histogram(q, bins=bins, weights=amp)
            smoothed = gaussian_filter1d(hist, gaussian_filter)
            return smoothed

        xrd_match = []
        for i, e1 in enumerate(icdd_entries):
            if e1.structure is None:
                e1_copy = e1
                for e2 in icsd_entries:
                    q, amp = 4 * np.pi / 1.54056 * np.sin(np.radians(e1.data['xrd'][0]) / 2) * 10, e1.data['xrd'][1]
                    bins = np.linspace(q[0] - 1, q[-1] + 1, bin_number)
                    smooth_xrds_i = smooth_hist(q, amp, bins)
                    q, amp = e2.data['xrd'][0], e2.data['xrd'][1]
                    smooth_xrds_j = smooth_hist(q, amp, bins)
                    smooth_xrds_i = smooth_xrds_i / np.max(smooth_xrds_i) * 100
                    smooth_xrds_j = smooth_xrds_j / np.max(smooth_xrds_j) * 100
                    abs_diff = np.abs(smooth_xrds_i - smooth_xrds_j)
                    R = np.sqrt(np.sum(abs_diff ** 2) / max(np.sum(smooth_xrds_i ** 2), np.sum(smooth_xrds_j ** 2)))

                    e1_comp = np.array([e1.composition[Element(el)] for el in self.chemsys])
                    e1_comp = e1_comp / np.sum(e1_comp)
                    e2_comp = np.array([e2.composition[Element(el)] for el in self.chemsys])
                    e2_comp = e2_comp / np.sum(e2_comp)
                    comp_distance = np.linalg.norm(e1_comp - e2_comp, ord=1)

                    if R < R_cutoff and comp_distance < 1.5:
                        # print(self.chemsys)
                        # print(e1.composition, e1_comp)
                        # print(e2.composition, e2_comp)
                        # print(comp_distance)
                        xrd_match.append(e1.entry_id)
                        plt.plot(bins[0:999], smooth_xrds_i, label=f'icdd {e1_copy.name} id: {e1_copy.entry_id}')
                        plt.plot(bins[0:999], smooth_xrds_j, label=f'icsd {e2.name} id: {e2.entry_id}')
                        plt.legend()
                        plt.show()
                        s = self.get_stability(e2)
                        if s is None:
                            if icdd_entries[i].stability is None:
                                icdd_entries[i] = e2
                        else:
                            if icdd_entries[i].stability is None or icdd_entries[i].stability > s:
                                icdd_entries[i] = e2
                    else:
                        for shift in arange(-0.5, 0.5, 0.01):
                            q, amp = e1.data['xrd'][0] + shift, e1.data['xrd'][1]
                            smooth_xrds_i = smooth_hist(q, amp, bins)
                            q, amp = e2.data['xrd'][0], e2.data['xrd'][1]
                            smooth_xrds_j = smooth_hist(q, amp, bins)
                            smooth_xrds_i = smooth_xrds_i / np.max(smooth_xrds_i) * 100
                            smooth_xrds_j = smooth_xrds_j / np.max(smooth_xrds_j) * 100

                            abs_diff = np.abs(smooth_xrds_i - smooth_xrds_j)
                            R = np.sqrt(
                                np.sum(abs_diff ** 2) / max(np.sum(smooth_xrds_i ** 2), np.sum(smooth_xrds_j ** 2)))

                            if R < R_cutoff and comp_distance < 1.5:
                                # print(self.chemsys)
                                # print(e1.composition, e1_comp)
                                # print(e2.composition, e2_comp)
                                # print(comp_distance)
                                xrd_match.append(e1.entry_id)
                                plt.plot(bins[0:-1], smooth_xrds_i, label=f'icdd {e1_copy.name} id: {e1_copy.entry_id}')
                                plt.plot(bins[0:-1], smooth_xrds_j, label=f'icsd {e2.name} id: {e2.entry_id}')
                                plt.legend()
                                plt.show()
                                s = self.get_stability(e2)
                                if s is None:
                                    if icdd_entries[i].stability is None:
                                        icdd_entries[i] = e2
                                else:
                                    if icdd_entries[i].stability is None or icdd_entries[i].stability > s:
                                        icdd_entries[i] = e2
                                break
        s = set(xrd_match)

        return s, icdd_entries

    def check_oxi(self):
        c1 = self.composition[Element(self.chemsys[0])] * np.min(Element(self.chemsys[0]).common_oxidation_states) \
             + self.composition[Element(self.chemsys[1])] * np.min(Element(self.chemsys[1]).common_oxidation_states) \
             + self.composition[Element(self.chemsys[2])] * np.min(Element(self.chemsys[2]).common_oxidation_states) \
             - self.composition[Element('O')] * 2
        c2 = self.composition[Element(self.chemsys[0])] * np.max(Element(self.chemsys[0]).common_oxidation_states) \
             + self.composition[Element(self.chemsys[1])] * np.max(Element(self.chemsys[1]).common_oxidation_states) \
             + self.composition[Element(self.chemsys[2])] * np.max(Element(self.chemsys[2]).common_oxidation_states) \
             - self.composition[Element('O')] * 2

        return c1 * c2 <= 0

    def get_stability(self, entry):
        def oqmd2pymatgen_struct(d):
            sites = []
            lat = Lattice(d['unit_cell'])
            for s in d['sites']:
                sp, _, x, y, z = s.split()
                x, y, z = map(float, [x, y, z])
                site = PeriodicSite(sp, [x, y, z], lat)
                sites.append(site)
            struct = Structure.from_sites(sites)
            return struct

        with qr.QMPYRester() as q:
            kwargs = {
                "composition": entry.name,
            }
            list_of_data = q.get_oqmd_phases(verbose=False, **kwargs)

        sm = StructureMatcher(comparator=ElementComparator(), attempt_supercell=True)

        s1 = entry.structure
        try:
            if len(list_of_data['data'])!=0:
                for i in range(len(list_of_data['data'])):
                    s2 = oqmd2pymatgen_struct(list_of_data['data'][i])
                    if sm.fit(s1, s2):
                        entry.stability = list_of_data['data'][i]['stability']
                        break
        except Exception as e:
            pass


        return entry.stability

    def check_stability(self, S_cutoff):
        def oqmd2pymatgen_struct(d):
            sites = []
            lat = Lattice(d['unit_cell'])
            for s in d['sites']:
                sp, _, x, y, z = s.split()
                x, y, z = map(float, [x, y, z])
                site = PeriodicSite(sp, [x, y, z], lat)
                sites.append(site)
            struct = Structure.from_sites(sites)
            return struct

        oqmd_data = []
        with qr.QMPYRester() as q:
            for i in range(len(self.ordered_entries)):
                kwargs = {
                    "composition": self.ordered_entries[i].name,
                }
                list_of_data = q.get_oqmd_phases(verbose=False, **kwargs)
                oqmd_data.append(list_of_data)
                # while oqmd_data[i] is None:
                #     list_of_data = q.get_oqmd_phases(verbose=False, **kwargs)
                #     oqmd_data[i] = list_of_data

        sm = StructureMatcher(comparator=ElementComparator(), attempt_supercell=True)
        for i in range(len(self.ordered_entries)):
            s1 = self.ordered_entries[i].structure
            try:
                if len(oqmd_data[i]['data']) !=0:
                    for j in range(len(oqmd_data[i]['data'])):
                        s2 = oqmd2pymatgen_struct(oqmd_data[i]['data'][j])
                        # groups = sm.group_structures([s1, s2])
                        if sm.fit(s1, s2):
                            self.ordered_entries[i].stability = oqmd_data[i]['data'][j]['stability']
                            break
            except Exception as e:
                pass
            continue

        null_stability = [_ for _ in self.ordered_entries if _.stability is None]  # should be removed at last

        stable_entires = [_ for _ in self.ordered_entries if _.stability is not None and _.stability < S_cutoff]
        unstable_entries = [_ for _ in self.ordered_entries if _.stability is not None and _.stability > S_cutoff]
        candidates_entires = [_ for _ in self.entries if _.stability is None or _.stability < S_cutoff]
        return null_stability, stable_entires, unstable_entries, candidates_entires
