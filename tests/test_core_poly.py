"""Unit tests for polytomous core probability functions."""

import numpy as np
import pytest

from irt.core_poly import (
    prob_pcm,
    prob_rsm,
    prob_grm,
    prob_gpcm,
    prob_nrm,
    log_prob_pcm,
    log_prob_rsm,
    log_prob_grm,
    log_prob_gpcm,
    log_prob_nrm,
)


class TestProbPcm:
    def test_sums_to_one(self):
        theta = np.array([-2, 0, 2])
        b = np.array([0.0, 0.5, 1.0])
        p = prob_pcm(theta, b, 4)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(3))

    def test_log_prob_equals_log_of_prob(self):
        theta = np.array([-1, 0, 1])
        b = np.array([0.0, 0.5])
        for c in range(3):
            log_p = log_prob_pcm(theta, b, 3, c)
            p = prob_pcm(theta, b, 3)[:, c]
            np.testing.assert_allclose(log_p, np.log(np.clip(p, 1e-12, 1)))

    def test_edge_cases(self):
        p = prob_pcm(np.array([-10]), np.array([0, 0]), 3)
        np.testing.assert_allclose(p.sum(), 1.0)


class TestProbRsm:
    def test_sums_to_one(self):
        theta = np.array([-1, 0, 1])
        p = prob_rsm(theta, 0.0, np.array([-0.5, 0, 0.5]), 4)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(3))


class TestProbGrm:
    def test_sums_to_one(self):
        theta = np.array([-2, 0, 2])
        a, b = 1.0, np.array([-1, 0, 1])
        p = prob_grm(theta, a, b, 4)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(3))

    def test_ordering(self):
        theta = np.array([0.0])
        a, b = 1.0, np.array([-1, 0, 1])
        p = prob_grm(theta, a, b, 4)
        np.testing.assert_array_less(np.diff(p[0]), 0.1)


class TestProbGpcm:
    def test_sums_to_one(self):
        theta = np.array([-1, 0, 1])
        p = prob_gpcm(theta, 1.5, np.array([0, 0.5, 1]), 4)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(3))


class TestProbNrm:
    def test_sums_to_one(self):
        theta = np.array([-1, 0, 1])
        a = np.array([0, 0.5, 1.0])
        c = np.array([0, -0.5, 0.5])
        p = prob_nrm(theta, a, c, 3)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(3))

    def test_identification(self):
        a, c = np.array([0, 0.5, 1]), np.array([0, -0.5, 0.5])
        p = prob_nrm(np.array([0]), a, c, 3)
        np.testing.assert_allclose(p.sum(), 1.0)
