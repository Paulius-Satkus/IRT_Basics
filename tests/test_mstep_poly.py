"""Unit tests for polytomous M-step solvers."""

import numpy as np
import pytest

from irt.mstep_poly import (
    update_item_pcm,
    update_item_gpcm,
    update_item_grm,
    update_item_rsm,
    update_tau_rsm,
    update_item_nrm,
)
from irt.core_poly import prob_pcm, prob_gpcm, prob_grm, prob_rsm, prob_nrm


class TestUpdateItemPcm:
    def test_converges(self):
        theta = np.linspace(-3, 3, 21)
        b_true = np.array([0.0, 0.5, 1.0])
        p = prob_pcm(theta, b_true, 4)
        N_k = np.ones(21) * 10
        R_k_c = (p * N_k[:, np.newaxis])
        b0 = np.array([0.1, 0.6, 1.1])
        b_hat = update_item_pcm(theta, N_k, R_k_c, b0, (-6, 6), max_iter=50)
        np.testing.assert_allclose(b_hat, b_true, atol=0.3)


class TestUpdateItemGpcm:
    def test_converges(self):
        theta = np.linspace(-2, 2, 15)
        a_true, b_true = 1.2, np.array([-0.5, 0, 0.5])
        p = prob_gpcm(theta, a_true, b_true, 4)
        N_k = np.ones(15) * 20
        R_k_c = p * N_k[:, np.newaxis]
        a_hat, b_hat = update_item_gpcm(
            theta, N_k, R_k_c, 1.0, np.array([0, 0, 0]),
            (0.25, 4), (-6, 6), max_iter=50
        )
        np.testing.assert_allclose(a_hat, a_true, atol=0.5)
        np.testing.assert_allclose(b_hat, b_true, atol=0.5)


class TestUpdateItemGrm:
    def test_converges(self):
        theta = np.linspace(-2, 2, 15)
        a_true, b_true = 1.0, np.array([-1, 0, 1])
        p = prob_grm(theta, a_true, b_true, 4)
        N_k = np.ones(15) * 20
        R_k_c = p * N_k[:, np.newaxis]
        a_hat, b_hat = update_item_grm(
            theta, N_k, R_k_c, 1.0, np.array([-0.5, 0.5, 1.5]),
            (0.25, 4), (-6, 6), max_iter=50
        )
        assert a_hat > 0
        np.testing.assert_array_less(0, np.diff(b_hat))  # b ordered


class TestUpdateItemRsm:
    def test_updates_b(self):
        theta = np.linspace(-2, 2, 11)
        tau = np.array([-0.5, 0, 0.5])
        b_true = 0.0
        p = prob_rsm(theta, b_true, tau, 4)
        N_k = np.ones(11) * 15
        R_k_c = p * N_k[:, np.newaxis]
        b_hat = update_item_rsm(theta, N_k, R_k_c, 0.1, tau, (-6, 6), max_iter=30)
        np.testing.assert_allclose(b_hat, b_true, atol=0.5)


class TestUpdateItemNrm:
    def test_converges(self):
        theta = np.linspace(-2, 2, 11)
        a = np.array([0, 0.5, 1.0])
        c = np.array([0, -0.5, 0.5])
        p = prob_nrm(theta, a, c, 3)
        N_k = np.ones(11) * 10
        R_k_c = p * N_k[:, np.newaxis]
        a0 = np.array([0, 0.3, 0.8])
        c0 = np.array([0, -0.3, 0.3])
        a_hat, c_hat = update_item_nrm(theta, N_k, R_k_c, a0, c0, 3, max_iter=30)
        np.testing.assert_allclose(a_hat[0], 0)
        np.testing.assert_allclose(c_hat[0], 0)
