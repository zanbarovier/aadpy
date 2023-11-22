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


class AttitudeHelperUtils:
    @staticmethod
    def tilde_matrix(vec):
        """
        Creates a tilde matrix from an input vector
        Args:
            vec (array): Numpy array which is the input vector

        Returns:
             tilde_mat (array): Numpy array which is the tilde matrix
        """
        x1 = vec[0]
        x2 = vec[1]
        x3 = vec[2]
        tilde_mat = np.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0]])

        return tilde_mat

    @staticmethod
    def dcm_princ_rot_1(theta: float, rad: bool=False) -> DCM:
        if not rad:
            theta = np.radians(theta)
        dcm = np.array([[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]])
        return DCM(dcm=dcm)

    @staticmethod
    def dcm_princ_rot_2(theta: float, rad: bool=False) -> DCM:
        if not rad:
            theta = np.radians(theta)
        dcm = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
        return DCM(dcm=dcm)

    @staticmethod
    def dcm_princ_rot_3(theta:float, rad:bool=False) -> DCM:
        if not rad:
            theta = np.radians(theta)
        dcm = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        return DCM(dcm=dcm)


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

    @property
    def dcm(self) -> np.ndarray:
        return self._dcm

    @dcm.setter
    def dcm(self, dcm: np.ndarray) -> None:
        if type(dcm) != np.ndarray:
            raise TypeError("Invalid input type %s for dcm, should be np array" % type(dcm))

        #TODO checking that dcm is valid

        self._dcm = dcm


class Quaternion:
    _quaternion: np.ndarray
    def __init__(self, quat: np.ndarray) -> None:
        self.quaternion = quat
        self.normalize()

    def switch_to_short_rot(self) -> None:
        self._quaternion = -self._quaternion


    def normalize(self) -> None:
        self._quaternion = self._quaternion / np.linalg.norm(self._quaternion)

    def __repr__(self) -> str:
        quat_str = f"Quaternion=[{self._quaternion[0]},{self._quaternion[1]},{self._quaternion[2]},{self._quaternion[3]}]"
        return quat_str

    def __str__(self) -> str:
        quat_str = f"[{self._quaternion[0]},{self._quaternion[1]},{self._quaternion[2]},{self._quaternion[3]}]"
        return quat_str

    def __mul__(self, quat_2: "Quaternion") -> "Quaternion":
        self.normalize()
        quat_2.normalize()
        b1 = self.quaternion
        b2 = quat_2._quaternion

        beta2_mat = np.array(
            [[b2[0], -b2[1], -b2[2], -b2[3]], [b2[1], b2[0], b2[3], -b2[2]], [b2[2], -b2[3], b2[0], b2[1]],
             [b2[3], b2[2], -b2[1], b2[0]]])

        beta_added = np.matmul(beta2_mat, np.transpose(b1))
        return Quaternion(quat=beta_added)

    def divide_right(self, quat_div: "Quaternion") -> Quaternion:
        """ Subtracts 2 quaternions , quat1-quat2
                     -  quat 1 is the left hand side (the product of 2 quaternions) : q1 = q3*q2
                         where we are solving for q3 (i.e. solving for the left quaternion,
                         by dividing the right quaternion)"""
        self.normalize()
        quat_div.normalize()

        b1 = self.quaternion
        b2 = quat_div._quaternion

        beta2_mat = np.array(
            [[b2[0], -b2[1], -b2[2], -b2[3]], [b2[1], b2[0], -b2[3], b2[2]], [b2[2], b2[3], b2[0], -b2[1]],
             [b2[3], -b2[2], b2[1], b2[0]]])

        b2_mat_inv = np.linalg.inv(beta2_mat)

        b_final = b2_mat_inv.dot(np.transpose(b1))
        return Quaternion(quat=b_final)

    def divide_left(self, quat_div: "Quaternion") -> Quaternion:
        """ Subtracts 2 quaternions , quat1-quat2
             -  quat 1 is the left hand side (the product of 2 quaternions) : q1 = q2*q3
                 where we are solving for q3 (i.e. solving for the right quaternion,
                 by dividing the left quaternion)"""
        self.normalize()
        quat_div.normalize()

        b1 = self.quaternion
        b2 = quat_div._quaternion

        beta2_mat = np.array(
            [[b2[0], -b2[1], -b2[2], -b2[3]], [b2[1], b2[0], b2[3], -b2[2]], [b2[2], -b2[3], b2[0], b2[1]],
             [b2[3], b2[2], -b2[1], b2[0]]])

        b2_mat_inv = np.linalg.inv(beta2_mat)

        b_final = b2_mat_inv.dot(np.transpose(b1))
        return Quaternion(quat=b_final)

    @property
    def quaternion(self) -> np.ndarray:
        return self._quaternion

    @quaternion.setter
    def quaternion(self, quat: np.ndarray) -> None:
        if not type(quat) == np.ndarray:
            raise TypeError("Type should be nd array, not", type(quat))

        self._quaternion = quat


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
    _mrp: np.ndarray

    def __init__(self, mrp: np.ndarray) -> None:
        self.mrp = mrp

    def __repr__(self) -> str:
        mrp_str = f"MRP=[{self._mrp[0]},{self._mrp[1]},{self._mrp[2]}]"
        return mrp_str

    def __str__(self) -> str:
        mrp_str = f"[{self._mrp[0]},{self._mrp[1]},{self._mrp[2]}]"
        return mrp_str

    def __add__(self, other: "MRP") -> "MRP":
        self.short_rotation()
        other.short_rotation()

        if (self.determinant() - 1) < 0.01 and (other.determinant()) - 1 < 0.01:
            self.shadow_set()

        s2 = self._mrp
        s1 = other.mrp

        sig = ((1 - np.dot(s1, s1)) * s2 + (1 - np.dot(s2, s2)) * s1 - 2 * np.cross(s2, s1)) / (
                1 + np.dot(s1, s1) * np.dot(s2, s2) - 2 * np.dot(s1, s2))

        added_obj = MRP(mrp=sig)
        added_obj.short_rotation()

        return added_obj


    def __sub__(self, other: "MRP") -> "MRP":
        self.short_rotation()
        other.short_rotation()

        if (self.determinant() - 1) < 0.01 and (other.determinant()) - 1 < 0.01:
            self.shadow_set()

        s1 = self._mrp
        s2 = other.mrp

        sig = ((1 - np.dot(s2, s2)) * s1 - (1 - np.dot(s1, s1)) * s2 + 2 * np.cross(s1, s1)) / (
                1 + np.dot(s1, s1) * np.dot(s2, s2) + 2 * np.dot(s1, s2))

        sub_obj = MRP(mrp=sig)
        sub_obj.short_rotation()

        return sub_obj

    def norm(self) -> float:
        return np.linalg.norm(self._mrp)

    def determinant(self) -> float:
        return np.linalg.norm(self._mrp) ** 2

    def shadow_set(self) -> "MRP":
        self.mrp = -self._mrp/(np.power(np.linalg.norm(self._mrp), 2))
        return self

    def inverse_rotation(self) -> "MRP":
        self._mrp *= -1
        return self

    def short_rotation(self) -> "MRP":
        if self.norm() >= 1:
            self.shadow_set()

        return self

    def long_rotation(self) -> "MRP":
        if self.norm() <= 1:
            self.shadow_set()

        return self

    @property
    def mrp(self) -> np.ndarray:
        return self._mrp

    @mrp.setter
    def mrp(self, mrp: np.ndarray) -> None:
        self._mrp = mrp



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


    @staticmethod
    def mrp_to_dcm(mrp: MRP) -> DCM:
        sig = mrp.mrp
        sig_tilde = AttitudeHelperUtils.tilde_matrix(sig)
        dcm = np.identity(3) + (1 / ((1 + np.dot(sig, sig)) ** 2)) * (
                8 * np.dot(sig_tilde, sig_tilde) - 4 * (1 - (np.dot(sig, sig))) * sig_tilde)
        dcm_obj = DCM(dcm=dcm)
        return dcm_obj

    @staticmethod
    def dcm_to_mrp(dcm_obj: DCM) -> MRP:
        dcm = dcm_obj.dcm
        zeta = np.sqrt(np.trace(dcm) + 1)
        sig = (1 / (zeta * (zeta + 2))) * np.array(
            [dcm[1, 2] - dcm[2, 1], dcm[2, 0] - dcm[0, 2], dcm[0, 1] - dcm[1, 0]])
        mrp = MRP(mrp=sig)
        mrp.shadow_set()
        return mrp

    @staticmethod
    def mrp_to_quat(mrp: MRP) -> Quaternion:
        sig = mrp.mrp
        beta0 = (1 - np.dot(sig, sig)) / (1 + np.dot(sig, sig))
        beta1 = 2 * sig[0] / (1 + np.dot(sig, sig))
        beta2 = 2 * sig[1] / (1 + np.dot(sig, sig))
        beta3 = 2 * sig[2] / (1 + np.dot(sig, sig))

        beta = np.array([beta0, beta1, beta2, beta3])
        beta = beta / np.linalg.norm(beta)  # check quat is normalized
        quat = Quaternion(quat=beta)
        return quat

    @staticmethod
    def quat_to_mrp(quat: Quaternion) -> MRP:
        beta = quat.quaternion
        sig = (1 / (1 + beta[0])) * beta[1:]
        mrp = MRP(mrp=sig)
        mrp.shadow_set()
        return mrp

