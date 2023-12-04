import ezc3d
import numpy as np
from bionc.bionc_numpy import NaturalCoordinates
from math import ceil, floor, log10
from export_c3d_from_bionc_model import add_point_from_dictionary

class C3DInverseKinematicsExporter:
    def __init__(self, filename, model):
            self.filename = filename
            self.c3d_file = ezc3d.c3d(filename)
            self.model = model

    def add_technical_markers_to_c3d(self, Q)->None:
        """
        This function add the technical markers of the model to the c3d file. This point are the markers that are rigidly associated to the
        segments of the model.
        Parameters
        ----------
        c3d_file : ezc3d.c3d
            The c3d file to add the points to
        model : BiomechanicalModel
            The biomechanical model from which the data can be exported
        Q : numpy.ndarray | NaturalCoordinates
            The natural coordinates of the model, should be given as a Natural coordinates but if given as a numpy array it will be converted

        Returns
        -------
        c3d_file : ezc3d.c3d
            The c3d file with natural coordinate points added
        """

        if Q is not isinstance(Q, NaturalCoordinates):
            Q = NaturalCoordinates(Q)

        model_markers = self.model.markers(Q)

        dict_to_add = dict()
        # We add the technical markers
        for ind_marker, name_marker in enumerate(self.model.marker_names):
            dict_to_add[f"{name_marker}_optim"] = model_markers[:, ind_marker, :]

        add_point_from_dictionary(self.c3d_file, dict_to_add)

    def add_natural_coordinate_to_c3d(self, Q)->None:
        """
        This function add the natural coordinate of the model to the c3d file. It add the segment rp,rd,u,w to the c3d file.
        Parameters
        ----------
        c3d_file : ezc3d.c3d
            The c3d file to add the points to
        model : BiomechanicalModel
            The biomechanical model from which the data can be exported
        Q : numpy.ndarray | NaturalCoordinates
            The natural coordinates of the model, should be given as a Natural coordinates but if given as a numpy array it will be converted

        Returns
        -------
        c3d_file : ezc3d.c3d
            The c3d file with natural coordinate points added
        """

        if Q is not isinstance(Q, NaturalCoordinates):
            Q = NaturalCoordinates(Q)
        # Calulation of a reasonable factor for the u and w
        list_factor = []
        for s in range(Q.nb_qi()):
            name_segment = self.model.segment_names[s]
            Qi = Q.vector(s)
            rp_temp = Qi.rp
            rd_temp = Qi.rd
            u_temp = Qi.u
            v_mean = np.mean(np.linalg.norm(rd_temp - rp_temp, axis=0))
            u_mean = np.mean(np.linalg.norm(u_temp, axis=0))
            list_factor.append(ceil(log10(u_mean / v_mean)))

        most_occurence_factor = max(set(list_factor), key=list_factor.count)
        factor = 10 ** most_occurence_factor

        dict_to_add = dict()
        # We add the segment rp,rd,u,w to the c3d file
        for s in range(Q.nb_qi()):
            name_segment = self.model.segment_names[s]
            Qi = Q.vector(s)
            rp_temp = Qi.rp
            rd_temp = Qi.rd
            u_temp = Qi.u
            w_temp = Qi.w

            v_mean = np.mean(np.linalg.norm(rd_temp - rp_temp, axis=0))
            u_mean = np.mean(np.linalg.norm(u_temp, axis=0))
            dict_to_add[f"u_{name_segment}"] = rp_temp + u_temp / factor
            dict_to_add[f"rp_{name_segment}"] = rp_temp
            dict_to_add[f"rd_{name_segment}"] = rd_temp
            dict_to_add[f"w_{name_segment}"] = rd_temp + w_temp / factor

        add_point_from_dictionary(self.c3d_file, dict_to_add)

    def export(self,newfilename=None)->None:
            """
            This function export the c3d file from the model and the natural coordinate
            """
            if newfilename is None:
                newfilename = self.filename[:-4] + "_ik.c3d"

            ezc3d.write(newfilename, self.c3d_file)
