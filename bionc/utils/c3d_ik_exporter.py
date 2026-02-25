from math import ceil, log10

import ezc3d
import numpy as np

from .c3d_export_utils import add_point_from_dictionary
from ..protocols.biomechanical_model import GenericBiomechanicalModel
from ..protocols.natural_coordinates import NaturalCoordinates


class C3DInverseKinematicsExporter:
    def __init__(self, filename: str, model: GenericBiomechanicalModel):
        """
        Parameters
        ----------
        filename: str
            The file of the original c3d to consider
        model: GenericBiomechanicalModel
            The Biomechanical model considered
        """

        self.filename = filename
        self.c3d_file = ezc3d.c3d(filename)
        self.model = model

    def add_technical_markers(self, Q: NaturalCoordinates, unit: str = "m") -> None:
        """
        This function add the technical markers of the model to the c3d file. This point are the markers that are rigidly associated to the
        segments of the model.

        Parameters
        ----------
        Q : numpy.ndarray | NaturalCoordinates
            The natural coordinates of the model, should be given as a Natural coordinates but if given as a numpy array it will be converted
        unit : str
            The unit of the markers, either "m" or "mm"

        Returns
        -------
        c3d_file : ezc3d.c3d
            The c3d file with natural coordinate points added
        """
        from ..bionc_numpy.natural_coordinates import NaturalCoordinates as NaturalCoordinatesNumpy

        if Q is not isinstance(Q, NaturalCoordinatesNumpy):
            Q = NaturalCoordinatesNumpy(Q)

        model_markers = self.model.markers(Q)

        unit_factor = {"mm": 1000, "m": 1}.get(unit)
        if unit_factor is None:
            raise ValueError("unit must be 'm' or 'mm'")

        dict_to_add = dict()
        # We add the technical markers
        for ind_marker, name_marker in enumerate(self.model.marker_names):
            dict_to_add[f"{name_marker}_optim"] = model_markers[:, ind_marker, :] * unit_factor

        add_point_from_dictionary(self.c3d_file, dict_to_add)

    def add_natural_coordinate(self, Q: NaturalCoordinates, unit: str = "m") -> None:
        """
        This function add the natural coordinate of the model to the c3d file. It add the segment rp,rd,u,w to the c3d file.

        Parameters
        ----------
        Q : numpy.ndarray | NaturalCoordinates
            The natural coordinates of the model, should be given as a Natural coordinates but if given as a numpy array it will be converted
        unit : str
            The unit of the markers, either "m" or "mm"

        Returns
        -------
        c3d_file : ezc3d.c3d
            The c3d file with natural coordinate points added
        """
        from ..bionc_numpy.natural_coordinates import NaturalCoordinates as NaturalCoordinatesNumpy

        if Q is not isinstance(Q, NaturalCoordinatesNumpy):
            Q = NaturalCoordinatesNumpy(Q)

        # Calulation of a reasonable factor for the u and w
        list_factor = []
        for s in range(Q.nb_qi()):
            Qi = Q.vector(s)
            rp_temp = Qi.rp
            rd_temp = Qi.rd
            u_temp = Qi.u
            v_mean = np.mean(np.linalg.norm(rd_temp - rp_temp, axis=0))
            u_mean = np.mean(np.linalg.norm(u_temp, axis=0))
            list_factor.append(ceil(log10(u_mean / v_mean)))

        most_occurence_factor = max(set(list_factor), key=list_factor.count)
        factor = 10**most_occurence_factor

        unit_factor = {"mm": 1000, "m": 1}.get(unit)
        if unit_factor is None:
            raise ValueError("unit must be 'm' or 'mm'")

        dict_to_add = dict()
        # We add the segment rp,rd,u,w to the c3d file
        for s in range(Q.nb_qi()):
            name_segment = self.model.segment_names[s]
            Qi = Q.vector(s)
            rp_temp = Qi.rp
            rd_temp = Qi.rd
            u_temp = Qi.u
            w_temp = Qi.w

            dict_to_add[f"u_{name_segment}"] = (rp_temp + u_temp / factor) * unit_factor
            dict_to_add[f"rp_{name_segment}"] = rp_temp * unit_factor
            dict_to_add[f"rd_{name_segment}"] = rd_temp * unit_factor
            dict_to_add[f"w_{name_segment}"] = (rd_temp + w_temp / factor) * unit_factor

        add_point_from_dictionary(self.c3d_file, dict_to_add)

    def export(self, newfilename=None) -> None:
        """
        This function export the c3d file from the model and the natural coordinate
        """
        if newfilename is None:
            newfilename = self.filename[:-4] + "_ik.c3d"

        self.c3d_file.write(newfilename)
