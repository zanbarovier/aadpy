from aadpy.attitude import Quaternion, AttitudeConverter
import pytest
import numpy as np

class TestQuaternion:
    def test_create_quaternion(self):
        quat = Quaternion(quat=np.array([1.0, 0.0, 0.0, 0.0]))