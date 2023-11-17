from __future__ import  annotations
import numpy as np
from enum import IntEnum

class EulerAngleSequence(IntEnum):
    EA_121 = 0,
    EA_123 = 1,
    EA_131 = 2,
    EA_132 = 3,
    EA_212 = 4,
    EA_213 = 5,
    EA_232 = 6,
    EA_231 = 7,
    EA_313 = 8,
    EA_312 = 9,
    EA_323 = 10,
    EA_321 = 11,


class EulerAngles:
    _ea_seq : EulerAngleSequence
    def __init__(self, seq: EulerAngleSequence) -> None:
        #self.ea_seq = seq
        self._ea_seq = seq

    #@property
    #def ea_seq(self) -> EulerAngleSequence:
    #    return self._ea_seq

    #@property.setter
    #def ea_seq(self, seq: EulerAngleSequence) -> None:
    #    self._ea_seq = seq

class DCM:
    _dcm: np.ndarray
    def __init__(self, dcm: np.ndarray) -> None:
        self.dcm = dcm

    def dcm_transposed(self) -> "DCM":
        return DCM(dcm= self.dcm.T)

    @property
    def dcm(self) -> np.ndarray:
        return self._dcm

    @dcm.setter
    def dcm(self, dcm: np.ndarray) -> None:
        if type(dcm) != np.ndarray:
            raise TypeError("Invalid input type %s for dcm, should be np array" % type(dcm))

        #TODO checking that dcm is valid

        self._dcm = dcm

    def __repr__(self) -> str:
        dcm_str = f"""DCM=[[{self.dcm[0,0]},{self.dcm[0,1]},{self.dcm[0,2]}],[{self.dcm[1,0]},{self.dcm[1,1]},{self.dcm[1,2]}],[{self.dcm[2,0]},{self.dcm[2,1]},{self.dcm[2,2]}]]"""
        return dcm_str

    def __str__(self) -> str:
        dcm_str = f"""[[{self.dcm[0, 0]},{self.dcm[0, 1]},{self.dcm[0, 2]}],[{self.dcm[1, 0]},{self.dcm[1, 1]},{self.dcm[1, 2]}],[{self.dcm[2, 0]},{self.dcm[2, 1]},{self.dcm[2, 2]}]]"""
        return dcm_str

    def __mul__(self, dcm_2: "DCM") -> "DCM":
        dcm_mul = np.matmul(self.dcm, dcm_2.dcm)
        return DCM(dcm=dcm_mul)

    def __div__(self, dcm_2: "DCM") -> "DCM":
        dcm_mul = self*dcm_2.dcm_transposed()
        return dcm_mul

    def __truediv__(self, dcm_2: "DCM") -> "DCM":
        return self.__div__(dcm_2)


class Quaternion:
    _quaternion: np.ndarray
    def __init__(self, quat: np.ndarray) -> None:
        self.quaternion = quat
        self.normalize()

    def switch_to_short_rot(self) -> None:
        self._quaternion = -self._quaternion


    def normalize(self) -> None:
        self._quaternion = self._quaternion / np.linalg.norm(self._quaternion)

    @property
    def quaternion(self) -> np.ndarray:
        return self._quaternion

    @quaternion.setter
    def quaternion(self, quat: np.ndarray) -> None:
        if not type(quat) == np.ndarray:
            raise TypeError("Type should be nd array, not", type(quat))

        self._quaternion = quat

    def __repr__(self) -> str:
        quat_str = f"Quaternion=[{self._quaternion[0]},{self._quaternion[1]},{self._quaternion[2]},{self._quaternion[3]}]"
        return quat_str

    def __str__(self) -> str:
        quat_str = f"[{self._quaternion[0]},{self._quaternion[1]},{self._quaternion[2]},{self._quaternion[3]}]"
        return quat_str

    def __add__(self, quat_2: "Quaternion") -> "Quaternion":
        self.normalize()
        quat_2.normalize()
        b1 = self.quaternion
        b2 = quat_2._quaternion

        beta2_mat = np.array(
            [[b2[0], -b2[1], -b2[2], -b2[3]], [b2[1], b2[0], b2[3], -b2[2]], [b2[2], -b2[3], b2[0], b2[1]],
             [b2[3], b2[2], -b2[1], b2[0]]])

        beta_added = np.matmul(beta2_mat, np.transpose(b1))
        return Quaternion(quat=beta_added)



class PRV:
    _phi_deg : float
    _e_hat: np.ndarray

    def __init__(self, phi_deg: float, e_hat: np.ndarray) -> None:
        self._phi_deg = phi_deg
        self._e_hat = e_hat

    def short_positive_phi(self) -> float:
        return self._phi_deg

    def long_positive_phi(self) -> float:
        return 360.0 - self._phi_deg


    def short_negative_phi(self) -> float:
        return -self._phi_deg

    def long_negative_phi(self) -> float:
        return -self.long_positive_phi()

    @property
    def phi_deg_e_hat(self) -> (float, np.ndarray):
        return self._phi_deg, self._e_hat

    @phi_deg_e_hat.setter
    def phi_deg_e_hat(self) -> None:
        return self._phi_deg

    def __repr__(self) -> str:
        prv_str = f"Phi={self._phi_deg} deg, e_hat=[{self._e_hat[0]},{self._e_hat[1]},{self._e_hat[2]}]"
        return prv_str

    def __str__(self) -> str:
        prv_str = f"[{self._phi_deg},{self._e_hat[0]},{self._e_hat[1]},{self._e_hat[2]}]"
        return prv_str


class CRP:
    def __init__(self) -> None:
        pass


class MRP:
    def __init__(self) -> None:
        pass

class AttitudeConverter:
    @staticmethod
    def dcm_to_prv(dcm: DCM) -> PRV:
        dcm_mat = dcm.dcm
        phi = np.arccos(0.5 * (dcm_mat[0, 0] + dcm_mat[1, 1] + dcm_mat[2, 2] - 1))
        phi_deg = np.degrees(phi)
        ehat = (0.5 / np.sin(phi)) * np.array([dcm_mat[1, 2] - dcm_mat[2, 1],
                                               dcm_mat[2, 0] - dcm_mat[0, 2],
                                               dcm_mat[0, 1] - dcm_mat[1, 0]])
        prv = PRV(phi_deg=phi_deg, e_hat=ehat)
        return prv

    @staticmethod
    def prv_to_dcm(prv: PRV) -> DCM:
        phi = np.radians(prv.phi_deg_e_hat[0])
        ehat = prv.phi_deg_e_hat[1]
        sig = 1 - np.cos(phi)
        e1, e2, e3 = ehat[0], ehat[1], ehat[2]

        dcm_mat = np.array([
            [sig * e1 ** 2 + np.cos(phi), e1 * e2 * sig + e3 * np.sin(phi), e1 * e3 * sig - e2 * np.sin(phi)],
             [e2 * e1 * sig - e3 * np.sin(phi), sig * e2 ** 2 + np.cos(phi), e2 * e3 * sig + e1 * np.sin(phi)],
             [e3 * e1 * sig + e2 * np.sin(phi), e3 * e2 * sig - e1 * np.sin(phi), sig * e3 ** 2 + np.cos(phi)]])
        dcm = DCM(dcm=dcm_mat)
        return dcm

    @staticmethod
    def quat_to_dcm(quaternion: Quaternion) -> DCM:
        quaternion.normalize()
        quat = quaternion.quaternion
        b0, b1, b2, b3 = quat[0], quat[1], quat[2], quat[3]
        dcm = np.array([[b0 ** 2 + b1 ** 2 - b2 ** 2 - b3 ** 2, 2 * (b1 * b2 + b0 * b3), 2 * (b1 * b3 - b0 * b2)],
                        [2 * (b1 * b2 - b0 * b3), b0 ** 2 - b1 ** 2 + b2 ** 2 - b3 ** 2, 2 * (b2 * b3 + b0 * b1)],
                        [2 * (b1 * b3 + b0 * b2), 2 * (b2 * b3 - b0 * b1), b0 ** 2 - b1 ** 2 - b2 ** 2 + b3 ** 2]])
        dcm_obj = DCM(dcm=dcm)
        return dcm_obj

    @staticmethod
    def dcm_to_quat(dcm_obj: DCM) -> Quaternion:
        """Convert dcm to quaternion using Sheppard's method"""
        dcm = dcm_obj.dcm
        b0_test = 0.25 * (1 + np.trace(dcm))
        b1_test = 0.25 * (1 + 2 * dcm[0, 0] - np.trace(dcm))
        b2_test = 0.25 * (1 + 2 * dcm[1, 1] - np.trace(dcm))
        b3_test = 0.25 * (1 + 2 * dcm[2, 2] - np.trace(dcm))

        beta_test = np.array([b0_test, b1_test, b2_test, b3_test])
        max_ind = np.argmax(beta_test)

        if max_ind == 0:
            b0 = np.sqrt(b0_test)
            b1 = 0.25 * (dcm[1, 2] - dcm[2, 1]) / b0
            b2 = 0.25 * (dcm[2, 0] - dcm[0, 2]) / b0
            b3 = 0.25 * (dcm[0, 1] - dcm[1, 0]) / b0
        elif max_ind == 1:
            b1 = np.sqrt(b1_test)
            b0 = 0.25 * (dcm[1, 2] - dcm[2, 1]) / b1
            b2 = 0.25 * (dcm[0, 1] - dcm[1, 0]) / b1
            b3 = 0.25 * (dcm[2, 0] - dcm[0, 2]) / b1
        elif max_ind == 2:
            b2 = np.sqrt(b2_test)
            b0 = 0.25 * (dcm[2, 0] - dcm[1, 0]) / b2
            b1 = 0.25 * (dcm[0, 1] + dcm[1, 0]) / b2
            b3 = 0.25 * (dcm[1, 2] + dcm[2, 0]) / b2
        elif max_ind == 3:
            b3 = np.sqrt(b3_test)
            b0 = 0.25 * (dcm[0, 1] - dcm[1, 0]) / b3
            b1 = 0.25 * (dcm[2, 0] + dcm[0, 2]) / b3
            b2 = 0.25 * (dcm[1, 2] + dcm[2, 0]) / b3

        quat = np.array([b0, b1, b2, b3])
        quaternion = Quaternion(quat=quat)

        return quaternion
