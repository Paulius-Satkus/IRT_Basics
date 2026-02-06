# QA Report

Date: 2026-02-06

## Summary

- Status: PASS
- Total tests: 246
- Passed: 246
- Failures: 0
- Warnings: 4
- Duration: 19.32s

## Command

```
python -m pytest -vv --tb=short
```

## Full Test Log (Verbose)

```
============================= test session starts ==============================
platform darwin -- Python 3.13.5, pytest-8.4.1, pluggy-1.5.0 -- /Users/paulius.satkus/miniconda3/bin/python
cachedir: .pytest_cache
rootdir: /Users/paulius.satkus/Documents/Item Response Theory Python
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.10.0
collecting ... collected 246 items

tests/test_api.py::TestFitRaschMmlEm::test_basic_fit PASSED              [  0%]
tests/test_api.py::TestFitRaschMmlEm::test_converges PASSED              [  0%]
tests/test_api.py::TestFitRaschMmlEm::test_a_is_one_for_rasch PASSED     [  1%]
tests/test_api.py::TestFitRaschMmlEm::test_loglik_increases PASSED       [  1%]
tests/test_api.py::TestFitRaschMmlEm::test_recovers_difficulty_order PASSED [  2%]
tests/test_api.py::TestFit2plMmlEm::test_basic_fit PASSED                [  2%]
tests/test_api.py::TestFit2plMmlEm::test_a_within_bounds PASSED          [  2%]
tests/test_api.py::TestFitJmle::test_basic_fit PASSED                    [  3%]
tests/test_api.py::TestFitJmle::test_2pl_jmle_raises PASSED              [  3%]
tests/test_api.py::TestFit3plMmlEm::test_basic_fit PASSED                [  4%]
tests/test_api.py::TestFit3plMmlEm::test_c_within_bounds PASSED          [  4%]
tests/test_api.py::TestFit3plMmlEm::test_fixed_c PASSED                  [  4%]
tests/test_api.py::TestScoring::test_eap_scoring PASSED                  [  5%]
tests/test_api.py::TestScoring::test_map_scoring PASSED                  [  5%]
tests/test_api.py::TestScoring::test_mle_scoring PASSED                  [  6%]
tests/test_api.py::TestScoring::test_score_correlates_with_raw PASSED    [  6%]
tests/test_api.py::TestScoring::test_score_new_data PASSED               [  6%]
tests/test_api.py::TestValidation::test_invalid_model_raises PASSED      [  7%]
tests/test_api.py::TestValidation::test_invalid_estimator_raises PASSED  [  7%]
tests/test_api.py::TestValidation::test_invalid_data_raises PASSED       [  8%]
tests/test_api.py::TestMissingData::test_handles_nan PASSED              [  8%]
tests/test_api.py::TestTechnicalParameters::test_custom_quadpts PASSED   [  8%]
tests/test_api.py::TestTechnicalParameters::test_invalid_technical_raises PASSED [  9%]
tests/test_api.py::TestReports::test_item_report PASSED                  [  9%]
tests/test_api.py::TestReports::test_score_result_to_dataframe PASSED    [ 10%]
tests/test_core.py::TestClipProb::test_clips_zeros PASSED                [ 10%]
tests/test_core.py::TestClipProb::test_preserves_valid_probs PASSED      [ 10%]
tests/test_core.py::TestClipProb::test_custom_eps PASSED                 [ 11%]
tests/test_core.py::TestBernoulliLoglik::test_basic_computation PASSED   [ 11%]
tests/test_core.py::TestBernoulliLoglik::test_finite_at_extremes PASSED  [ 12%]
tests/test_core.py::TestBernoulliLoglik::test_vectorized PASSED          [ 12%]
tests/test_core.py::TestProb2pl::test_basic_computation PASSED           [ 13%]
tests/test_core.py::TestProb2pl::test_monotone_in_theta PASSED           [ 13%]
tests/test_core.py::TestProb2pl::test_monotone_in_discrimination PASSED  [ 13%]
tests/test_core.py::TestProb2pl::test_difficulty_shift PASSED            [ 14%]
tests/test_core.py::TestProb2pl::test_symmetric PASSED                   [ 14%]
tests/test_core.py::TestProbMatrix2pl::test_shape PASSED                 [ 15%]
tests/test_core.py::TestProbMatrix2pl::test_consistency_with_scalar PASSED [ 15%]
tests/test_core.py::TestLogProb2pl::test_consistency PASSED              [ 15%]
tests/test_core.py::TestLogProb2pl::test_numerical_stability PASSED      [ 16%]
tests/test_core.py::TestCenterDifficulties::test_centers_to_zero PASSED  [ 16%]
tests/test_core.py::TestCenterDifficulties::test_with_fixed_items PASSED [ 17%]
tests/test_core.py::TestCenterDifficulties::test_preserves_differences PASSED [ 17%]
tests/test_core.py::TestInfo2pl::test_maximum_at_difficulty PASSED       [ 17%]
tests/test_core.py::TestInfo2pl::test_max_value_formula PASSED           [ 18%]
tests/test_core.py::TestInfo2pl::test_positive PASSED                    [ 18%]
tests/test_core_poly.py::TestProbPcm::test_sums_to_one PASSED            [ 19%]
tests/test_core_poly.py::TestProbPcm::test_log_prob_equals_log_of_prob PASSED [ 19%]
tests/test_core_poly.py::TestProbPcm::test_edge_cases PASSED             [ 19%]
tests/test_core_poly.py::TestProbRsm::test_sums_to_one PASSED            [ 20%]
tests/test_core_poly.py::TestProbGrm::test_sums_to_one PASSED            [ 20%]
tests/test_core_poly.py::TestProbGrm::test_ordering PASSED               [ 21%]
tests/test_core_poly.py::TestProbGpcm::test_sums_to_one PASSED           [ 21%]
tests/test_core_poly.py::TestProbNrm::test_sums_to_one PASSED            [ 21%]
tests/test_core_poly.py::TestProbNrm::test_identification PASSED         [ 22%]
tests/test_data.py::TestAsBinaryMatrix::test_numpy_array PASSED          [ 22%]
tests/test_data.py::TestAsBinaryMatrix::test_handles_nan PASSED          [ 23%]
tests/test_data.py::TestAsBinaryMatrix::test_invalid_values_raises PASSED [ 23%]
tests/test_data.py::TestAsBinaryMatrix::test_too_few_items_raises PASSED [ 23%]
tests/test_data.py::TestAsBinaryMatrix::test_too_few_persons_raises PASSED [ 24%]
tests/test_data.py::TestAsBinaryMatrix::test_empty_raises PASSED         [ 24%]
tests/test_data.py::TestFilterPeopleMinItems::test_filters_correctly PASSED [ 25%]
tests/test_data.py::TestFilterPeopleMinItems::test_preserves_alignment PASSED [ 25%]
tests/test_data.py::TestFilterItemsMinPeople::test_filters_correctly PASSED [ 26%]
tests/test_data.py::TestComputeItemPvalues::test_basic_computation PASSED [ 26%]
tests/test_data.py::TestComputeItemPvalues::test_with_missing PASSED     [ 26%]
tests/test_data.py::TestComputeItemPvalues::test_clips_extreme PASSED    [ 27%]
tests/test_data.py::TestComputePersonScores::test_basic_computation PASSED [ 27%]
tests/test_data.py::TestComputePersonScores::test_with_missing PASSED    [ 28%]
tests/test_data.py::TestIdentifyExtremeScores::test_identifies_perfect PASSED [ 28%]
tests/test_data.py::TestIdentifyExtremeScores::test_with_missing PASSED  [ 28%]
tests/test_data_poly.py::TestAsPolytomousMatrix::test_infers_categories PASSED [ 29%]
tests/test_data_poly.py::TestAsPolytomousMatrix::test_explicit_categories PASSED [ 29%]
tests/test_data_poly.py::TestAsPolytomousMatrix::test_rejects_invalid PASSED [ 30%]
tests/test_data_poly.py::TestAsPolytomousMatrix::test_dataframe PASSED   [ 30%]
tests/test_data_poly.py::TestAsPolytomousMatrix::test_too_few_items_raises PASSED [ 30%]
tests/test_data_poly.py::TestComputeItemCategoryProportions::test_correct_proportions PASSED [ 31%]
tests/test_diagnostics.py::TestComputeResiduals::test_output_shapes PASSED [ 31%]
tests/test_diagnostics.py::TestComputeResiduals::test_expected_in_range PASSED [ 32%]
tests/test_diagnostics.py::TestComputeResiduals::test_residual_bounded PASSED [ 32%]
tests/test_diagnostics.py::TestComputeResiduals::test_mean_residual_near_zero PASSED [ 32%]
tests/test_diagnostics.py::TestInfitOutfit::test_item_fit_shapes PASSED  [ 33%]
tests/test_diagnostics.py::TestInfitOutfit::test_person_fit_shapes PASSED [ 33%]
tests/test_diagnostics.py::TestInfitOutfit::test_fit_positive PASSED     [ 34%]
tests/test_diagnostics.py::TestInfitOutfit::test_good_fit_near_one PASSED [ 34%]
tests/test_diagnostics.py::TestInfitOutfit::test_dataframe_output PASSED [ 34%]
tests/test_diagnostics.py::TestPointBiserial::test_shape PASSED          [ 35%]
tests/test_diagnostics.py::TestPointBiserial::test_positive_for_good_items PASSED [ 35%]
tests/test_diagnostics.py::TestPointBiserial::test_bounded PASSED        [ 36%]
tests/test_diagnostics.py::TestItemDiscrimination::test_shape PASSED     [ 36%]
tests/test_diagnostics.py::TestItemDiscrimination::test_positive PASSED  [ 36%]
tests/test_diagnostics.py::TestReliability::test_reliability_bounded PASSED [ 37%]
tests/test_diagnostics.py::TestReliability::test_separation_positive PASSED [ 37%]
tests/test_diagnostics.py::TestChiSquare::test_shapes PASSED             [ 38%]
tests/test_diagnostics.py::TestChiSquare::test_p_values_bounded PASSED   [ 38%]
tests/test_diagnostics.py::TestInformationCriteria::test_aic_bic_ordering PASSED [ 39%]
tests/test_diagnostics.py::TestInformationCriteria::test_2pl_more_params PASSED [ 39%]
tests/test_diagnostics.py::TestResidualDiagnostics::test_q3_shape_and_diag PASSED [ 39%]
tests/test_diagnostics.py::TestResidualDiagnostics::test_q3_summary PASSED [ 40%]
tests/test_diagnostics.py::TestResidualDiagnostics::test_srmsr_non_negative PASSED [ 40%]
tests/test_diagnostics.py::TestFitIndices::test_cfi_tli_bounds PASSED    [ 41%]
tests/test_diagnostics.py::TestFitIndices::test_model_fit_summary_includes_new_stats PASSED [ 41%]
tests/test_diagnostics.py::TestLikelihoodRatioTest::test_positive_chi_square PASSED [ 41%]
tests/test_diagnostics.py::TestLikelihoodRatioTest::test_p_value_bounded PASSED [ 42%]
tests/test_diagnostics.py::TestModelFitSummary::test_returns_dict PASSED [ 42%]
tests/test_diagnostics.py::TestItemFitTable::test_returns_dataframe PASSED [ 43%]
tests/test_ltm_validation.py::TestLSATValidationLtm::test_rasch_difficulty_vs_ltm PASSED [ 43%]
tests/test_ltm_validation.py::TestLSATValidationLtm::test_rasch_loglik_vs_ltm PASSED [ 43%]
tests/test_ltm_validation.py::TestLSATValidationLtm::test_2pl_parameters_vs_ltm PASSED [ 44%]
tests/test_ltm_validation.py::TestLSATValidationLtm::test_eap_scores_vs_ltm PASSED [ 44%]
tests/test_ltm_validation.py::TestLSATValidationLtm::test_map_scores_vs_ltm PASSED [ 45%]
tests/test_ltm_validation.py::TestQuadratureComparison::test_ltm_quadrature_settings PASSED [ 45%]
tests/test_mirt_validation.py::TestLSAT7Validation::test_rasch_difficulty_recovery PASSED [ 45%]
tests/test_mirt_validation.py::TestLSAT7Validation::test_rasch_loglik PASSED [ 46%]
tests/test_mirt_validation.py::TestLSAT7Validation::test_2pl_parameter_recovery PASSED [ 46%]
tests/test_mirt_validation.py::TestLSAT7Validation::test_eap_scores PASSED [ 47%]
tests/test_mirt_validation.py::TestSimulatedDataValidation::test_rasch_known_params PASSED [ 47%]
tests/test_mirt_validation.py::TestSimulatedDataValidation::test_2pl_known_params PASSED [ 47%]
tests/test_mirt_validation.py::TestScoringValidation::test_eap_vs_mirt PASSED [ 48%]
tests/test_mirt_validation.py::TestScoringValidation::test_map_vs_mirt PASSED [ 48%]
tests/test_mirt_validation.py::TestTechnicalSettings::test_quadpts_effect PASSED [ 49%]
tests/test_mml_em.py::TestComputeLoglikNk::test_shape PASSED             [ 49%]
tests/test_mml_em.py::TestComputeLoglikNk::test_finite_values PASSED     [ 50%]
tests/test_mml_em.py::TestComputeLoglikNk::test_handles_missing PASSED   [ 50%]
tests/test_mml_em.py::TestComputeLoglikNk::test_monotone_in_correct_direction PASSED [ 50%]
tests/test_mml_em.py::TestPosteriorWeights::test_rows_sum_to_one PASSED  [ 51%]
tests/test_mml_em.py::TestPosteriorWeights::test_non_negative PASSED     [ 51%]
tests/test_mml_em.py::TestPosteriorWeights::test_no_nan PASSED           [ 52%]
tests/test_mml_em.py::TestSufficientStats::test_item_stats_shape PASSED  [ 52%]
tests/test_mml_em.py::TestSufficientStats::test_all_items_shape PASSED   [ 52%]
tests/test_mml_em.py::TestSufficientStats::test_R_leq_N PASSED           [ 53%]
tests/test_mml_em.py::TestSufficientStats::test_all_zeros_gives_R_zero PASSED [ 53%]
tests/test_mml_em.py::TestInitializeParamsMml::test_rasch_a_is_one PASSED [ 54%]
tests/test_mml_em.py::TestInitializeParamsMml::test_b_within_bounds PASSED [ 54%]
tests/test_mml_em.py::TestInitializeParamsMml::test_uses_start_values PASSED [ 54%]
tests/test_mstep.py::TestRaschQGradHess::test_gradient_at_optimum_near_zero PASSED [ 55%]
tests/test_mstep.py::TestRaschQGradHess::test_hessian_negative PASSED    [ 55%]
tests/test_mstep.py::TestRaschQGradHess::test_Q_finite PASSED            [ 56%]
tests/test_mstep.py::TestUpdateItemRaschNewton::test_converges PASSED    [ 56%]
tests/test_mstep.py::TestUpdateItemRaschNewton::test_respects_bounds PASSED [ 56%]
tests/test_mstep.py::TestTwoplQAndGrad::test_gradient_shape PASSED       [ 57%]
tests/test_mstep.py::TestTwoplQAndGrad::test_finite_values PASSED        [ 57%]
tests/test_mstep.py::TestUpdateItem2plLbfgsb::test_returns_finite PASSED [ 58%]
tests/test_mstep.py::TestUpdateItem2plLbfgsb::test_improves_Q PASSED     [ 58%]
tests/test_mstep.py::TestUpdateItem2plLbfgsb::test_respects_bounds PASSED [ 58%]
tests/test_mstep_poly.py::TestUpdateItemPcm::test_converges PASSED       [ 59%]
tests/test_mstep_poly.py::TestUpdateItemGpcm::test_converges PASSED      [ 59%]
tests/test_mstep_poly.py::TestUpdateItemGrm::test_converges PASSED       [ 60%]
tests/test_mstep_poly.py::TestUpdateItemRsm::test_updates_b PASSED       [ 60%]
tests/test_mstep_poly.py::TestUpdateItemNrm::test_converges PASSED       [ 60%]
tests/test_plotting.py::TestPlotICC::test_single_item PASSED             [ 61%]
tests/test_plotting.py::TestPlotICC::test_multiple_items PASSED          [ 61%]
tests/test_plotting.py::TestPlotICC::test_custom_range PASSED            [ 62%]
tests/test_plotting.py::TestPlotICCEmpirical::test_basic_plot PASSED     [ 62%]
tests/test_plotting.py::TestPlotAllICCs::test_grid_layout PASSED         [ 63%]
tests/test_plotting.py::TestPlotIIF::test_single_item PASSED             [ 63%]
tests/test_plotting.py::TestPlotIIF::test_multiple_items PASSED          [ 63%]
tests/test_plotting.py::TestPlotTIF::test_basic_plot PASSED              [ 64%]
tests/test_plotting.py::TestPlotTIF::test_with_se PASSED                 [ 64%]
tests/test_plotting.py::TestPlotWrightMap::test_basic_plot PASSED        [ 65%]
tests/test_plotting.py::TestPlotItemFit::test_basic_plot PASSED          [ 65%]
tests/test_plotting.py::TestPlotItemFitBars::test_basic_plot PASSED      [ 65%]
tests/test_plotting.py::TestPlotPersonFit::test_basic_plot PASSED        [ 66%]
tests/test_plotting.py::TestPlotAbilityDistribution::test_basic_plot PASSED [ 66%]
tests/test_plotting.py::TestPlotSEByTheta::test_basic_plot PASSED        [ 67%]
tests/test_plotting.py::TestPlotSEByTheta::test_with_theoretical PASSED  [ 67%]
tests/test_plotting.py::TestPlotResiduals::test_basic_plot PASSED        [ 67%]
tests/test_plotting.py::TestPlotResiduals::test_single_item PASSED       [ 68%]
tests/test_plotting.py::TestFitResultPlotMethods::test_plot_icc PASSED   [ 68%]
tests/test_plotting.py::TestFitResultPlotMethods::test_plot_icc_empirical PASSED [ 69%]
tests/test_plotting.py::TestFitResultPlotMethods::test_plot_tif PASSED   [ 69%]
tests/test_plotting.py::TestFitResultPlotMethods::test_plot_wright_map PASSED [ 69%]
tests/test_plotting.py::TestFitResultPlotMethods::test_plot_item_fit PASSED [ 70%]
tests/test_plotting.py::TestFitResultPlotMethods::test_plot_ability_distribution PASSED [ 70%]
tests/test_plotting.py::TestFitResultDiagnosticMethods::test_item_fit PASSED [ 71%]
tests/test_plotting.py::TestFitResultDiagnosticMethods::test_person_fit PASSED [ 71%]
tests/test_plotting.py::TestFitResultDiagnosticMethods::test_model_fit PASSED [ 71%]
tests/test_polytomous_diagnostics.py::TestPolytomousDiagnostics::test_item_fit_returns_dataframe PASSED [ 72%]
tests/test_polytomous_diagnostics.py::TestPolytomousDiagnostics::test_person_fit_works PASSED [ 72%]
tests/test_polytomous_diagnostics.py::TestPolytomousDiagnostics::test_model_fit_returns_dict PASSED [ 73%]
tests/test_polytomous_fit.py::TestPolytomousFit::test_pcm_converges PASSED [ 73%]
tests/test_polytomous_fit.py::TestPolytomousFit::test_gpcm_converges PASSED [ 73%]
tests/test_polytomous_fit.py::TestPolytomousFit::test_grm_converges PASSED [ 74%]
tests/test_polytomous_fit.py::TestPolytomousFit::test_rsm_converges PASSED [ 74%]
tests/test_polytomous_fit.py::TestPolytomousFit::test_nrm_fits PASSED    [ 75%]
tests/test_polytomous_fit.py::TestPolytomousFit::test_eap_map_mle_finite PASSED [ 75%]
tests/test_polytomous_fit.py::TestPolytomousFit::test_fit_accepts_model_names PASSED [ 76%]
tests/test_polytomous_fit.py::TestPolytomousFit::test_fit_result_params_shape PASSED [ 76%]
tests/test_polytomous_fit.py::TestPolytomousFit::test_binary_unchanged PASSED [ 76%]
tests/test_polytomous_plotting.py::TestPolytomousPlotting::test_plot_ccc_does_not_raise PASSED [ 77%]
tests/test_polytomous_plotting.py::TestPolytomousPlotting::test_plot_icc_polytomous PASSED [ 77%]
tests/test_polytomous_validation.py::TestScienceValidationLtm::test_grm_params_vs_ltm PASSED [ 78%]
tests/test_polytomous_validation.py::TestScienceValidationLtm::test_grm_loglik_vs_ltm PASSED [ 78%]
tests/test_polytomous_validation.py::TestScienceValidationLtm::test_gpcm_params_vs_ltm PASSED [ 78%]
tests/test_polytomous_validation.py::TestScienceValidationLtm::test_gpcm_loglik_vs_ltm PASSED [ 79%]
tests/test_polytomous_validation.py::TestScienceValidationLtm::test_pcm_params_vs_ltm PASSED [ 79%]
tests/test_polytomous_validation.py::TestScienceValidationLtm::test_pcm_loglik_vs_ltm PASSED [ 80%]
tests/test_polytomous_validation.py::TestScienceValidationLtm::test_grm_eap_vs_ltm PASSED [ 80%]
tests/test_polytomous_validation.py::TestScienceValidationMirt::test_grm_params_vs_mirt PASSED [ 80%]
tests/test_polytomous_validation.py::TestScienceValidationMirt::test_grm_loglik_vs_mirt PASSED [ 81%]
tests/test_polytomous_validation.py::TestScienceValidationMirt::test_gpcm_params_vs_mirt PASSED [ 81%]
tests/test_polytomous_validation.py::TestScienceValidationMirt::test_gpcm_loglik_vs_mirt PASSED [ 82%]
tests/test_polytomous_validation.py::TestScienceValidationMirt::test_pcm_params_vs_mirt PASSED [ 82%]
tests/test_polytomous_validation.py::TestScienceValidationMirt::test_pcm_loglik_vs_mirt PASSED [ 82%]
tests/test_polytomous_validation.py::TestScienceValidationMirt::test_rsm_converges_and_loglik_reasonable PASSED [ 83%]
tests/test_priors.py::TestPriorDerivatives::test_normal PASSED           [ 83%]
tests/test_priors.py::TestPriorDerivatives::test_lognormal PASSED        [ 84%]
tests/test_priors.py::TestPriorDerivatives::test_gamma PASSED            [ 84%]
tests/test_priors.py::TestPriorDerivatives::test_beta PASSED             [ 84%]
tests/test_priors.py::TestParseItemPriors::test_dict_grouped_and_per_item PASSED [ 85%]
tests/test_priors.py::TestParseItemPriors::test_dataframe PASSED         [ 85%]
tests/test_priors.py::TestParseItemPriors::test_rasch_ignores_a_priors PASSED [ 86%]
tests/test_priors.py::TestParseItemPriors::test_bounds_validation PASSED [ 86%]
tests/test_priors.py::TestParseItemPriors::test_3pl_c_gamma_prior PASSED [ 86%]
tests/test_priors.py::TestMstepPriors::test_rasch_grad_includes_prior PASSED [ 87%]
tests/test_priors.py::TestMstepPriors::test_rasch_update_shrinks_to_prior PASSED [ 87%]
tests/test_priors.py::TestMstepPriors::test_twopl_grad_includes_priors PASSED [ 88%]
tests/test_priors.py::TestFitWithPriors::test_rasch_prior_shrinks_difficulty PASSED [ 88%]
tests/test_quadrature.py::TestMakeThetaGrid::test_basic_grid PASSED      [ 89%]
tests/test_quadrature.py::TestMakeThetaGrid::test_default_parameters PASSED [ 89%]
tests/test_quadrature.py::TestMakeThetaGrid::test_too_few_points_raises PASSED [ 89%]
tests/test_quadrature.py::TestMakeThetaGrid::test_invalid_bounds_raises PASSED [ 90%]
tests/test_quadrature.py::TestPriorLogpdf::test_normal_prior PASSED      [ 90%]
tests/test_quadrature.py::TestPriorLogpdf::test_uniform_prior PASSED     [ 91%]
tests/test_quadrature.py::TestPriorLogpdf::test_invalid_kind_raises PASSED [ 91%]
tests/test_quadrature.py::TestPriorLogpdf::test_negative_sd_raises PASSED [ 91%]
tests/test_quadrature.py::TestPriorLogpdf::test_prior_shape_matches_theta PASSED [ 92%]
tests/test_quadrature.py::TestGaussHermiteGrid::test_weights_sum_to_one PASSED [ 92%]
tests/test_quadrature.py::TestGaussHermiteGrid::test_nodes_centered PASSED [ 93%]
tests/test_quadrature.py::TestGaussHermiteGrid::test_custom_mean_sd PASSED [ 93%]
tests/test_quadrature.py::TestComputeIntegrationWeights::test_weights_sum_to_one PASSED [ 93%]
tests/test_quadrature.py::TestComputeIntegrationWeights::test_weights_positive PASSED [ 94%]
tests/test_quadrature.py::TestComputeIntegrationWeights::test_shape_matches_theta PASSED [ 94%]
tests/test_scoring.py::TestScoreEapFromW::test_eap_is_posterior_mean PASSED [ 95%]
tests/test_scoring.py::TestScoreEapFromW::test_eap_within_grid PASSED    [ 95%]
tests/test_scoring.py::TestScoreEapFromW::test_se_positive PASSED        [ 95%]
tests/test_scoring.py::TestScoreMapFromW::test_map_equals_argmax PASSED  [ 96%]
tests/test_scoring.py::TestScoreMapFromW::test_map_on_grid PASSED        [ 96%]
tests/test_scoring.py::TestScoreMleNewton::test_returns_correct_shape PASSED [ 97%]
tests/test_scoring.py::TestScoreMleNewton::test_extreme_scores_at_bounds PASSED [ 97%]
tests/test_scoring.py::TestScoreMleNewton::test_3pl_scoring_runs PASSED  [ 97%]
tests/test_scoring.py::TestScoreMleNewton::test_moderate_scores_within_bounds PASSED [ 98%]
tests/test_scoring.py::TestScoreMapNewton::test_extreme_scores_finite PASSED [ 98%]
tests/test_scoring.py::TestScoreMapNewton::test_shrinkage_toward_prior PASSED [ 99%]
tests/test_scoring.py::TestExpectedScore::test_increases_with_theta PASSED [ 99%]
tests/test_scoring.py::TestExpectedScore::test_bounded_by_item_count PASSED [100%]

=============================== warnings summary ===============================
tests/test_api.py::TestFitJmle::test_basic_fit
  /Users/paulius.satkus/Documents/Item Response Theory Python/irt/api.py:286: RuntimeWarning: 1 persons have extreme scores. Using Bayesian prior for regularization.
    result = fit_jmle(

tests/test_data.py::TestFilterPeopleMinItems::test_filters_correctly
  /Users/paulius.satkus/Documents/Item Response Theory Python/tests/test_data.py:83: UserWarning: Filtered 2 persons (66.7%) with fewer than 2 observed items.
    X_f, mask_f, names_f, keep = filter_people_min_items(

tests/test_data.py::TestFilterPeopleMinItems::test_preserves_alignment
  /Users/paulius.satkus/Documents/Item Response Theory Python/tests/test_data.py:101: UserWarning: Filtered 1 persons (33.3%) with fewer than 1 observed items.
    X_f, mask_f, names_f, _ = filter_people_min_items(

tests/test_data.py::TestFilterItemsMinPeople::test_filters_correctly
  /Users/paulius.satkus/Documents/Item Response Theory Python/tests/test_data.py:123: UserWarning: Filtered 1 items (50.0%) with fewer than 2 observed persons.
    X_f, mask_f, names_f, keep = filter_items_min_people(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 246 passed, 4 warnings in 19.32s =======================

```

## Results

- All tests passed, including:
  - API integration tests (binary + polytomous)
  - Core math utilities (binary + polytomous)
  - Data handling
  - Quadrature
  - MML-EM and M-step solvers
  - Scoring
  - Plotting
  - External validation against R `mirt` and `ltm`

## Warnings (Expected in tests)

- JMLE extreme scores warning emitted during JMLE test
- Data filtering warnings emitted in tests that intentionally drop rows/cols
- Polytomous non-convergence warnings on some random-data tests (NRM, etc.)

## Notes

- Binary: `mirt` refs via `tests/generate_mirt_reference.R`, `ltm` refs via `tests/generate_ltm_reference.R`
- Polytomous: `make refs` or `Rscript tests/generate_polytomous_reference.R` (requires R + mirt)
- Run `python scripts/run_tests_with_qa_report.py` or `make test` for full suite + QA report
