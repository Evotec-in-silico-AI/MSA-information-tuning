import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from figure_common import operational_bits_site_threshold


@dataclass(frozen=True)
class SourceTableSpec:
    """Defines one explicit source-to-final table copy."""
    final_workbook: str
    final_sheet: str
    source_workbook: str
    source_sheet: str
    column_map: dict
    required: bool = True


output_folder = Path("supplementary_data_files")

output_workbooks = {
    "data1": "Supplementary_Data_1_Cohort_and_MSA_Design.xlsx",
    "data2": "Supplementary_Data_2_MSA_SNR_Reference_Point.xlsx",
    "data3": "Supplementary_Data_3_Topology_and_Latent_Geometry.xlsx",
    "data4": "Supplementary_Data_4_Residue_and_Local_Confidence_Summaries.xlsx",
    "data5": "Supplementary_Data_5_Folding_Benchmark.xlsx",
    "data6": "Supplementary_Data_6_Robustness_Analyses.xlsx",
}

source_workbook_candidates = {
    "data_summary.xlsx": [Path("data_summary.xlsx")],
    "cohort.xlsx": [Path("cohort.xlsx")],
    "figure1_and_supplementary_figure1_source_data.xlsx": [
        Path("figure1_and_supplementary_figure1_source_data.xlsx"),
        Path("figure1_and_supplementary_figure1_outputs") / "figure1_and_supplementary_figure1_source_data.xlsx",
    ],
    "figure2_and_supplementary_figure2_source_data.xlsx": [
        Path("figure2_and_supplementary_figure2_source_data.xlsx"),
        Path("figure2_and_supplementary_figure2_outputs") / "figure2_and_supplementary_figure2_source_data.xlsx",
    ],
    "figure3_source_data.xlsx": [
        Path("figure3_source_data.xlsx"),
        Path("figure3_outputs") / "figure3_source_data.xlsx",
    ],
    "figure4_source_data.xlsx": [
        Path("figure4_source_data.xlsx"),
        Path("figure4_outputs") / "figure4_source_data.xlsx",
    ],
    "figure5_and_supplementary_figure3_source_data.xlsx": [
        Path("figure5_and_supplementary_figure3_source_data.xlsx"),
        Path("figure5_and_supplementary_figure3_outputs") / "figure5_and_supplementary_figure3_source_data.xlsx",
    ],
    "figure6_source_data.xlsx": [
        Path("figure6_source_data.xlsx"),
        Path("figure6_and_supplementary_figure4_outputs") / "figure6_source_data.xlsx",
    ],
    "figure6_and_supplementary_figure4_mediation_source_data.xlsx": [
        Path("figure6_and_supplementary_figure4_mediation_source_data.xlsx"),
        Path("figure6_and_supplementary_figure4_outputs") / "figure6_and_supplementary_figure4_mediation_source_data.xlsx",
    ],
    "figure7_and_supplementary_figure5_source_data.xlsx": [
        Path("figure7_and_supplementary_figure5_source_data.xlsx"),
        Path("figure7_and_supplementary_figure5_outputs") / "figure7_and_supplementary_figure5_source_data.xlsx",
    ],
    "supplementary_figure5cd_source_data.xlsx": [
        Path("supplementary_figure5cd_source_data.xlsx"),
        Path("supplementary_figure5cd_outputs") / "supplementary_figure5cd_source_data.xlsx",
    ],
    "paired_sensitivity_analysis.xlsx": [
        Path("paired_sensitivity_analysis.xlsx"),
        Path("sensitivity_analyses") / "paired_sensitivity_analysis.xlsx",
    ],
    "meff_depth_sensitivity.xlsx": [
        Path("meff_depth_sensitivity.xlsx"),
        Path("sensitivity_analyses") / "meff_depth_sensitivity.xlsx",
    ],
    "msa_snr_metric_sensitivity.xlsx": [
        Path("msa_snr_metric_sensitivity.xlsx"),
        Path("sensitivity_analyses") / "msa_snr_metric_sensitivity.xlsx",
    ],
    "threshold_boundary_sensitivity.xlsx": [
        Path("threshold_boundary_sensitivity.xlsx"),
        Path("sensitivity_analyses") / "threshold_boundary_sensitivity.xlsx",
    ],
}

banned_final_column_names = {
    "outer_label",
    "outer_label_base",
    "condition4",
    "target_stratum",
    "snr_used",
    "bits_site_min",
    "p10_used",
    "null_random",
    "pair_id",
    "run_root",
    "candidate_seed",
    "path_index",
    "sample_group",
    "condition_match_key",
}

condition_value_map = {
    "below targeted": "below-threshold targeted",
    "above targeted": "above-threshold targeted",
    "below matched control": "below-threshold matched control",
    "above matched control": "above-threshold matched control",
    "targeted": "information-targeted MSA",
    "null_random": "depth-matched random-row control",
    "below": "below threshold",
    "above": "above threshold",
    "low_target": "below-threshold targeted",
    "high_target": "above-threshold targeted",
}

metric_value_map = {
    "snr_used": "MSA-SNR",
    "max_plddt": "maximum pLDDT",
    "max_tm": "maximum TM-score",
    "max_novelty": "maximum generated-sequence novelty",
    "max_score_plddt": "maximum pLDDT-novelty frontier score",
    "max_score_tm": "maximum TM-score-novelty frontier score",
    "meff_theta0p70": "Meff (theta=0.70)",
    "n_sequences": "retained sequence count",
}

source_specs = [
    SourceTableSpec(
        "data1",
        "protein_cohort",
        "data_summary.xlsx",
        "Sheet1",
        {
            "accession": "protein_accession",
            "name": "protein_name",
            "species": "species",
            "EC": "ec_number",
            "residues": "length_residues",
            "fig1_taxon_group": "taxonomic_group",
            "fig1_functional_group": "functional_group",
        },
    ),
    SourceTableSpec(
        "data1",
        "protein_length_summary",
        "figure1_and_supplementary_figure1_source_data.xlsx",
        "figure1c_length_summary",
        {"metric": "metric", "value": "value"},
    ),
    SourceTableSpec(
        "data2",
        "msa_snr_peak_locations",
        "figure2_and_supplementary_figure2_source_data.xlsx",
        "figure2e_f_peak_by_protein",
        {
            "protein_name": "protein_accession",
            "bps_at_max_snr": "minimum_bits_per_site_at_maximum_msa_snr",
            "snr_at_max": "maximum_msa_snr",
            "softmax_weight": "peak_weight",
            "n_msa_designs_at_maximum": "n_msa_designs_at_maximum",
        },
    ),
    SourceTableSpec(
        "data3",
        "msa_topology_metric_definitions",
        "figure3_source_data.xlsx",
        "figure3_metric_dictionary",
        {
            "panel": "figure_panel",
            "metric": "metric",
            "metric_label": "metric_label",
            "scale": "scale",
            "role": "role",
        },
    ),
    SourceTableSpec(
        "data3",
        "msa_topology_regime_tests",
        "figure3_source_data.xlsx",
        "figure3c_k_unpaired_welch",
        {
            "panel": "figure_panel",
            "metric": "metric",
            "metric_label": "metric_label",
            "contrast": "contrast",
            "analysis_unit": "analysis_unit",
            "group_a": "group_a",
            "n_a": "n_a",
            "mean_a": "mean_a",
            "sd_a": "sd_a",
            "group_b": "group_b",
            "n_b": "n_b",
            "mean_b": "mean_b",
            "sd_b": "sd_b",
            "p_value": "p_value",
            "q_value_bh": "q_value_bh",
            "effect_name": "effect_size_name",
            "effect": "effect_size",
            "reported_in_figure": "reported_in_figure",
        },
    ),
    SourceTableSpec(
        "data3",
        "msa_topology_corr_summary",
        "figure3_source_data.xlsx",
        "figure3b_correlation_summary",
        {
            "regime": "information_regime",
            "n_features": "n_features",
            "n_feature_pairs": "n_feature_pairs",
            "mean_abs_spearman_rho": "mean_absolute_spearman_rho",
            "median_abs_spearman_rho": "median_absolute_spearman_rho",
        },
    ),
    SourceTableSpec(
        "data3",
        "latent_metric_definitions",
        "figure4_source_data.xlsx",
        "figure4_metric_dictionary",
        {
            "panel": "figure_panel",
            "metric": "metric",
            "metric_label": "metric_label",
            "scale": "scale",
            "role": "role",
        },
    ),
    SourceTableSpec(
        "data3",
        "latent_regime_tests",
        "figure4_source_data.xlsx",
        "figure4c_j_unpaired_welch",
        {
            "panel": "figure_panel",
            "metric": "metric",
            "metric_label": "metric_label",
            "contrast": "contrast",
            "analysis_unit": "analysis_unit",
            "group_a": "group_a",
            "n_a": "n_a",
            "mean_a": "mean_a",
            "sd_a": "sd_a",
            "group_b": "group_b",
            "n_b": "n_b",
            "mean_b": "mean_b",
            "sd_b": "sd_b",
            "p_value": "p_value",
            "q_value_bh": "q_value_bh",
            "effect_name": "effect_size_name",
            "effect": "effect_size",
            "reported_in_figure": "reported_in_figure",
        },
    ),
    SourceTableSpec(
        "data3",
        "latent_correlation_summary",
        "figure4_source_data.xlsx",
        "figure4b_correlation_summary",
        {
            "regime": "information_regime",
            "n_features": "n_features",
            "n_feature_pairs": "n_feature_pairs",
            "mean_abs_spearman_rho": "mean_absolute_spearman_rho",
            "median_abs_spearman_rho": "median_absolute_spearman_rho",
        },
    ),
    SourceTableSpec(
        "data4",
        "residue_mutation_by_msa_info",
        "figure5_and_supplementary_figure3_source_data.xlsx",
        "figure5a_edit_probability",
        {
            "condition4": "analysis_group",
            "bits_bin_mid": "residue_bits_per_site_bin_midpoint",
            "protein_median_mutation_rate": "protein_median_mutation_probability",
            "proteins": "n_proteins",
        },
    ),
    SourceTableSpec(
        "data4",
        "targeted_control_residue_delta",
        "figure5_and_supplementary_figure3_source_data.xlsx",
        "figure5f_effect_sizes",
        {
            "pair_id": "protein_accession",
            "stratum": "information_regime",
            "targeted_condition": "targeted_group",
            "control_condition": "matched_control_group",
            "metric": "metric",
            "metric_label": "metric_label",
            "delta_targeted_minus_control": "targeted_minus_matched_control_delta",
        },
    ),
    SourceTableSpec(
        "data4",
        "local_info_flow_summary",
        "figure6_source_data.xlsx",
        "figure6b_local_flow_summary",
        {
            "comparison": "comparison",
            "below_condition": "below_threshold_group",
            "above_condition": "above_threshold_group",
            "outcome_label": "local_readout",
            "n_below_proteins": "n_below_threshold_proteins",
            "n_above_proteins": "n_above_threshold_proteins",
            "observed_mean_rho_below": "mean_spearman_rho_below_threshold",
            "observed_mean_rho_above": "mean_spearman_rho_above_threshold",
            "observed_delta_rho": "delta_spearman_rho_below_minus_above",
            "bootstrap_ci_lo": "bootstrap_ci_low",
            "bootstrap_ci_hi": "bootstrap_ci_high",
            "bootstrap_p_value": "bootstrap_p_value",
        },
    ),
    SourceTableSpec(
        "data4",
        "mediation_paths_summary",
        "figure6_and_supplementary_figure4_mediation_source_data.xlsx",
        "full_chain_viable",
        {
            "regime_display": "information_regime",
            "terminal_response_display": "terminal_readout",
            "path_label": "path_label",
            "n": "n_blocks",
            "serial_indirect": "serial_indirect_effect",
            "ci_serial_lo": "ci_low",
            "ci_serial_hi": "ci_high",
            "cv_pass_folds": "cross_validation_pass_folds",
            "cv_total_folds": "cross_validation_total_folds",
        },
    ),
    SourceTableSpec(
        "data5",
        "benchmark_input_files",
        "figure7_and_supplementary_figure5_source_data.xlsx",
        "input_inventory",
        {
            "source_kind": "source_type",
            "target_label": "target_label",
            "target_percent": "target_fraction_percent",
            "path": "input_file_label",
            "exists": "file_present",
        },
    ),
    SourceTableSpec(
        "data5",
        "benchmark_qc_summary",
        "figure7_and_supplementary_figure5_source_data.xlsx",
        "stage_summary",
        {
            "source_kind": "source_type",
            "target_label": "target_label",
            "target_percent": "target_fraction_percent",
            "raw_rows_loaded": "raw_rows_loaded",
            "valid_rows": "valid_rows",
            "excluded_rows": "excluded_rows",
            "excluded_status_not_ok": "excluded_status_not_ok",
            "excluded_all_zero_structural_metrics": "excluded_all_zero_structural_metrics",
            "excluded_missing_summary_metrics": "excluded_missing_summary_metrics",
        },
    ),
    SourceTableSpec(
        "data5",
        "common_benchmark_targets",
        "figure7_and_supplementary_figure5_source_data.xlsx",
        "common_target_matched_members",
        {
            "pdb_id": "pdb_id",
            "required_target_count": "required_target_count",
            "present_target_count": "present_target_count",
            "common_target_matched_set_member": "included_in_common_target_matched_set",
        },
    ),
    SourceTableSpec(
        "data5",
        "paired_benchmark_summary",
        "figure7_and_supplementary_figure5_source_data.xlsx",
        "paired_rows_common_target",
        {
            "pdb_id": "pdb_id",
            "target_percent": "target_fraction_percent",
            "jack_mean_tm_score_qc": "jackhmmer_tm_score",
            "tuned_mean_tm_score_qc": "tuned_msa_tm_score",
            "delta_tm_score": "delta_tm_score_tuned_minus_jackhmmer",
            "jack_mean_rmsd_qc": "jackhmmer_rmsd",
            "tuned_mean_rmsd_qc": "tuned_msa_rmsd",
            "delta_rmsd": "delta_rmsd_jackhmmer_minus_tuned",
            "jack_mean_ptm_qc": "jackhmmer_ptm",
            "tuned_mean_ptm_qc": "tuned_msa_ptm",
            "delta_ptm": "delta_ptm_tuned_minus_jackhmmer",
            "jack_depth": "jackhmmer_depth",
            "tuned_depth": "tuned_msa_depth",
            "depth_fraction_of_jackhmmer": "depth_fraction_of_jackhmmer",
            "tuned_meff_theta_0p70": "tuned_msa_meff_theta_0p70",
            "achieved_fraction_below_2bps": "achieved_fraction_below_0p972_bits_per_site",
        },
    ),
    SourceTableSpec(
        "data6",
        "paired_overlap_tests",
        "paired_sensitivity_analysis.xlsx",
        "paired_tests",
        {
            "threshold_bits_site_min": "minimum_bits_per_site_threshold",
            "figure": "figure",
            "panel": "figure_panel",
            "metric": "metric",
            "metric_label": "metric_label",
            "n_paired_proteins": "n_paired_proteins",
            "mean_below": "mean_below_threshold",
            "sd_below": "sd_below_threshold",
            "mean_above": "mean_above_threshold",
            "sd_above": "sd_above_threshold",
            "mean_above_minus_below": "mean_above_minus_below",
            "paired_t_stat_above_minus_below": "paired_t_statistic_above_minus_below",
            "paired_t_p_value_two_sided": "p_value",
            "q_value_all_metrics": "q_value_all_metrics",
            "q_value_within_figure": "q_value_within_figure",
            "direction": "direction",
        },
    ),
    SourceTableSpec(
        "data6",
        "paired_overlap_proteins",
        "paired_sensitivity_analysis.xlsx",
        "protein_overlap",
        {
            "threshold_bits_site_min": "minimum_bits_per_site_threshold",
            "protein_name": "protein_accession",
            "has_below_targeted": "has_below_threshold_targeted_msa",
            "has_above_targeted": "has_above_threshold_targeted_msa",
            "included_in_paired_overlap": "included_in_paired_overlap",
            "conditions_present": "conditions_present",
        },
    ),
    SourceTableSpec(
        "data6",
        "threshold_protein_counts",
        "threshold_boundary_sensitivity.xlsx",
        "protein_counts_by_threshold",
        {
            "threshold_bits_site_min": "minimum_bits_per_site_threshold",
            "below_targeted_protein_count": "below_threshold_targeted_proteins",
            "above_targeted_protein_count": "above_threshold_targeted_proteins",
            "below_matched_control_protein_count": "below_threshold_matched_control_proteins",
            "above_matched_control_protein_count": "above_threshold_matched_control_proteins",
        },
    ),
    SourceTableSpec(
        "data6",
        "threshold_sensitivity_tests",
        "threshold_boundary_sensitivity.xlsx",
        "unpaired_sensitivity",
        {
            "threshold_bits_site_min": "minimum_bits_per_site_threshold",
            "figure": "figure",
            "panel": "figure_panel",
            "metric": "metric",
            "metric_label": "metric_label",
            "contrast": "contrast",
            "analysis_unit": "analysis_unit",
            "group_a": "group_a",
            "n_a": "n_a",
            "mean_a": "mean_a",
            "sd_a": "sd_a",
            "group_b": "group_b",
            "n_b": "n_b",
            "mean_b": "mean_b",
            "sd_b": "sd_b",
            "mean_a_minus_b": "mean_group_a_minus_group_b",
            "p_value": "p_value",
            "q_value_bh": "q_value_bh",
            "effect_name": "effect_size_name",
            "effect": "effect_size",
            "direction": "direction",
        },
    ),
    SourceTableSpec(
        "data6",
        "threshold_sign_stability",
        "threshold_boundary_sensitivity.xlsx",
        "sign_stability_summary",
        {
            "figure": "figure",
            "panel": "figure_panel",
            "metric": "metric",
            "metric_label": "metric_label",
            "contrast": "contrast",
            "primary_threshold": "primary_threshold_bits_per_site",
            "primary_direction": "primary_direction",
            "n_thresholds_tested": "n_thresholds_tested",
            "n_matching_primary_direction": "n_matching_primary_direction",
            "fraction_matching_primary_direction": "fraction_matching_primary_direction",
            "n_thresholds_q_lt_0p05": "n_thresholds_q_value_below_0p05",
            "fraction_thresholds_q_lt_0p05": "fraction_thresholds_q_value_below_0p05",
            "min_p_value": "minimum_p_value",
            "max_p_value": "maximum_p_value",
            "min_q_value_bh": "minimum_q_value_bh",
            "max_q_value_bh": "maximum_q_value_bh",
        },
    ),
    SourceTableSpec(
        "data6",
        "scale_control_corrs",
        "meff_depth_sensitivity.xlsx",
        "scale_correlations",
        {
            "analysis_group": "analysis_group",
            "variable_a": "variable_a",
            "variable_b": "variable_b",
            "n_msa": "n_msa_designs",
            "spearman_rho": "spearman_rho",
            "p_value": "p_value",
        },
    ),
    SourceTableSpec(
        "data6",
        "scale_adjusted_msa_snr",
        "meff_depth_sensitivity.xlsx",
        "partial_spearman_sensitivity",
        {
            "analysis_group": "analysis_group",
            "focal_predictor": "focal_metric",
            "readout": "readout",
            "covariates": "adjustment",
            "n_msa": "n_msa_designs",
            "partial_spearman_rho": "partial_spearman_rho",
            "p_value": "p_value",
        },
    ),
    SourceTableSpec(
        "data6",
        "threshold_scale_balance",
        "meff_depth_sensitivity.xlsx",
        "threshold_scale_balance",
        {
            "threshold_bits_site_min": "minimum_bits_per_site_threshold",
            "condition4": "analysis_group",
            "n_proteins": "n_proteins",
            "median_meff_theta0p70": "median_meff_theta0p70",
            "median_n_sequences": "median_retained_sequence_count",
            "median_primary_msa_snr": "median_msa_snr",
            "median_bits_site_min": "median_minimum_bits_per_site",
        },
    ),
    SourceTableSpec(
        "data6",
        "msa_snr_metric_sensitivity",
        "msa_snr_metric_sensitivity.xlsx",
        "structural_correlations",
        {
            "analysis_group": "analysis_group",
            "metric": "metric",
            "readout": "readout",
            "n_msa": "n_msa_designs",
            "spearman_rho": "spearman_rho",
            "p_value": "p_value",
        },
    ),
    SourceTableSpec(
        "data6",
        "protein_block_subsampling",
        "msa_snr_metric_sensitivity.xlsx",
        "protein_block_subsample_summary",
        {
            "analysis_group": "analysis_group",
            "metric": "metric",
            "readout": "readout",
            "full_spearman_rho": "full_cohort_spearman_rho",
            "n_replicates": "n_subsamples",
            "protein_fraction_retained": "protein_fraction_retained",
            "median_subsample_rho": "median_subsample_spearman_rho",
            "q05_subsample_rho": "subsample_spearman_rho_q05",
            "q95_subsample_rho": "subsample_spearman_rho_q95",
            "fraction_same_sign_as_full": "fraction_same_sign_as_full_cohort",
        },
    ),
]

readme_rows_by_workbook = {
    "data1": [
        ("description", "Protein cohort and aggregated MSA design summaries."),
        ("analysis_unit", "Protein or aggregated MSA-design group, depending on sheet."),
        ("minimum_information_threshold", "Operational threshold: 0.972093947 bits/site, described as approximately 1 bit/site."),
        ("matched_control_definition", "Depth-matched random-row controls inherit below-threshold or above-threshold status from the paired targeted MSA."),
    ],
    "data2": [
        ("description", "MSA-SNR reference-point summaries."),
        ("analysis_unit", "MSA-level correlations or one peak location per protein."),
        ("minimum_information_threshold", "Operational threshold: 0.972093947 bits/site, described as approximately 1 bit/site."),
    ],
    "data3": [
        ("description", "Protein-level MSA topology and generated-sequence latent-geometry summaries."),
        ("analysis_unit", "Protein-level condition means unless otherwise noted."),
        ("statistical_test", "Two-sided Welch tests with Benjamini-Hochberg q-values."),
    ],
    "data4": [
        ("description", "Residue-level, local-confidence, and mediation-style summary tables."),
        ("analysis_unit", "Protein-normalized residue summaries, protein-level deltas, local-flow summaries, or retained mediation-style pathways."),
        ("mediation_interpretation", "Mediation-style paths are reproducible covariance summaries, not causal mediation claims."),
    ],
    "data5": [
        ("description", "Independent folding benchmark target membership, QC, paired outcomes, and local transition summaries."),
        ("delta_direction", "Positive TM-score, pTM, and RMSD deltas indicate better tuned-MSA performance relative to JackHMMER; RMSD uses JackHMMER minus tuned MSA."),
        ("common_target_set", "Benchmark paired summaries use the common target matched set."),
        ("operational_information_cutoff", "The historical Figure 7 cutoff corresponds to 0.972093947 bits/site and is described as approximately 1 bit/site."),
    ],
    "data6": [
        ("description", "Robustness and sensitivity analyses supporting the main regime claims."),
        ("minimum_information_threshold", "Primary operational threshold: 0.972093947 bits/site; threshold sensitivity uses 0.8, 0.9, 0.972093947, 1.0, 1.1, and 1.2 bits/site."),
        ("protein_block_subsampling", "Subsampling summaries are descriptive stability summaries, not bootstrap confidence intervals."),
    ],
}


class SupplementaryDataError(Exception):
    """Signals a fail-fast packaging error."""


def main() -> None:
    """Builds the curated supplementary data workbook package."""
    resolved_sources = resolve_source_workbooks()
    output_folder.mkdir(exist_ok=True)
    workbook_tables = build_all_tables(resolved_sources)
    validate_all_tables(workbook_tables)
    inventory_rows = write_workbooks(workbook_tables, resolved_sources)
    write_inventory(inventory_rows)


def resolve_source_workbooks() -> dict:
    """Resolves required source workbook paths from explicit candidate locations."""
    resolved_sources = {}
    for workbook_name, candidate_paths in source_workbook_candidates.items():
        existing_paths = [candidate_path for candidate_path in candidate_paths if candidate_path.exists()]
        if not existing_paths:
            raise SupplementaryDataError(f"Missing required source workbook: {workbook_name}")
        resolved_sources[workbook_name] = existing_paths[0]
    return resolved_sources


def build_all_tables(resolved_sources: dict) -> dict:
    """Builds all final tables from the explicit manifest."""
    workbook_tables = {workbook_key: [("readme", build_readme(workbook_key))] for workbook_key in output_workbooks}
    add_manifest_tables(workbook_tables, resolved_sources)
    add_custom_tables(workbook_tables, resolved_sources)
    return workbook_tables


def add_manifest_tables(workbook_tables: dict, resolved_sources: dict) -> None:
    """Adds direct source-sheet selections to the final table collection."""
    for source_spec in source_specs:
        source_df = read_source_sheet(resolved_sources, source_spec.source_workbook, source_spec.source_sheet)
        final_df = select_and_rename_columns(source_df, source_spec.column_map, source_spec)
        final_df = clean_dataframe_values(final_df)
        workbook_tables[source_spec.final_workbook].append((source_spec.final_sheet, final_df))


def add_custom_tables(workbook_tables: dict, resolved_sources: dict) -> None:
    """Adds curated tables that require combining or aggregating sources."""
    workbook_tables["data1"].append(("cohort_composition", build_cohort_composition(resolved_sources)))
    workbook_tables["data1"].append(("msa_design_summary", build_msa_design_summary(resolved_sources)))
    workbook_tables["data2"].append(("msa_snr_readout_correlations", build_msa_snr_readout_correlations(resolved_sources)))
    workbook_tables["data4"].append(("local_trace_example_metadata", build_local_trace_example_metadata(resolved_sources)))
    workbook_tables["data5"].append(("benchmark_delta_summaries", build_benchmark_delta_summaries(resolved_sources)))
    workbook_tables["data5"].append(("benchmark_paired_tests", build_benchmark_paired_tests(resolved_sources)))
    workbook_tables["data5"].append(("local_benchmark_transitions", build_local_benchmark_transition_summary(resolved_sources)))
    workbook_tables["data5"].append(("local_benchmark_dictionaries", build_local_benchmark_dictionaries(resolved_sources)))


def build_readme(workbook_key: str) -> pd.DataFrame:
    """Builds a manuscript-facing readme table for one final workbook."""
    rows = readme_rows_by_workbook[workbook_key]
    return pd.DataFrame(rows, columns=["item", "description"])


def read_source_sheet(resolved_sources: dict, workbook_name: str, sheet_name: str) -> pd.DataFrame:
    """Reads one required source sheet with explicit workbook and sheet identifiers."""
    workbook_path = resolved_sources[workbook_name]
    sheet_names = pd.ExcelFile(workbook_path, engine="openpyxl").sheet_names
    if sheet_name not in sheet_names:
        available_sheets = ", ".join(sheet_names)
        raise SupplementaryDataError(f"Missing sheet {sheet_name} in {workbook_name}. Available sheets: {available_sheets}")
    return pd.read_excel(workbook_path, sheet_name=sheet_name, engine="openpyxl")


def select_and_rename_columns(source_df: pd.DataFrame, column_map: dict, source_spec: SourceTableSpec) -> pd.DataFrame:
    """Selects and renames explicit columns from a source table."""
    missing_columns = [column for column in column_map if column not in source_df.columns]
    if missing_columns:
        raise SupplementaryDataError(
            f"Missing columns in {source_spec.source_workbook}:{source_spec.source_sheet}: {missing_columns}"
        )
    final_df = source_df[list(column_map.keys())].rename(columns=column_map).copy()
    return final_df


def clean_dataframe_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Applies reader-facing value translations to string-valued cells."""
    cleaned_df = dataframe.copy()
    for column_name in cleaned_df.columns:
        if cleaned_df[column_name].dtype == object:
            cleaned_df[column_name] = cleaned_df[column_name].map(clean_cell_value)
    return cleaned_df


def clean_cell_value(value):
    """Converts known internal values to manuscript-facing values."""
    if not isinstance(value, str):
        return value
    if value in condition_value_map:
        return condition_value_map[value]
    if value in metric_value_map:
        return metric_value_map[value]
    if value == "target_below_vs_above":
        return "targeted below-threshold versus above-threshold"
    if value == "matched_control_below_vs_above":
        return "matched-control below-threshold versus above-threshold"
    if value == "all":
        return "all MSAs"
    return value


def build_cohort_composition(resolved_sources: dict) -> pd.DataFrame:
    """Builds one tidy cohort composition table from Figure 1 count sheets."""
    taxon_df = read_source_sheet(
        resolved_sources,
        "figure1_and_supplementary_figure1_source_data.xlsx",
        "figure1a_taxon_composition",
    )
    function_df = read_source_sheet(
        resolved_sources,
        "figure1_and_supplementary_figure1_source_data.xlsx",
        "figure1b_function_composition",
    )
    validate_columns(taxon_df, ["taxon_group", "count", "fraction"], "figure1a_taxon_composition")
    validate_columns(function_df, ["functional_group", "count", "fraction"], "figure1b_function_composition")
    taxon_final = taxon_df.rename(columns={"taxon_group": "category"})[["category", "count", "fraction"]].copy()
    taxon_final.insert(0, "composition_type", "taxonomic group")
    function_final = function_df.rename(columns={"functional_group": "category"})[["category", "count", "fraction"]].copy()
    function_final.insert(0, "composition_type", "functional group")
    return pd.concat([taxon_final, function_final], ignore_index=True)


def build_msa_design_summary(resolved_sources: dict) -> pd.DataFrame:
    """Builds an aggregated MSA design summary from the cohort workbook."""
    source_df = read_source_sheet(resolved_sources, "figure5_and_supplementary_figure3_source_data.xlsx", "sample_summary")
    required_columns = ["protein_name", "outer_label", "control_type", "target_stratum", "condition4", "sample_id"]
    validate_columns(source_df, required_columns, "figure5 sample_summary")
    working_df = source_df[required_columns].copy()
    working_df["analysis_group"] = working_df["condition4"].map(clean_cell_value)
    working_df["control_type_label"] = working_df["control_type"].map(clean_cell_value)
    working_df["information_regime"] = working_df["target_stratum"].map(clean_information_regime_value)
    grouped_df = (
        working_df.groupby(["analysis_group", "control_type_label", "information_regime"], dropna=False)
        .agg(
            n_proteins=("protein_name", "nunique"),
            n_msa_designs=("outer_label", "nunique"),
            n_generated_samples=("sample_id", "nunique"),
        )
        .reset_index()
    )
    grouped_df = grouped_df.rename(columns={"control_type_label": "control_type"})
    grouped_df["minimum_bits_per_site_threshold"] = operational_bits_site_threshold
    return grouped_df[
        [
            "analysis_group",
            "control_type",
            "information_regime",
            "n_proteins",
            "n_msa_designs",
            "n_generated_samples",
            "minimum_bits_per_site_threshold",
        ]
    ]


def build_analysis_group_label(row: pd.Series) -> str:
    """Builds a manuscript-facing analysis group label for an MSA design row."""
    regime = clean_information_regime_value(row["target_stratum"])
    control_type = row["control_type"]
    if control_type == "targeted":
        if regime == "below threshold":
            return "below-threshold targeted"
        if regime == "above threshold":
            return "above-threshold targeted"
    if control_type == "null_random":
        if regime == "below threshold":
            return "below-threshold matched control"
        if regime == "above threshold":
            return "above-threshold matched control"
    raise SupplementaryDataError(f"Unexpected control type/regime combination: {control_type}, {regime}")


def clean_information_regime_value(value) -> str:
    """Converts source regime values to reader-facing threshold labels."""
    if value == "below":
        return "below threshold"
    if value == "above":
        return "above threshold"
    if value in ["below threshold", "above threshold"]:
        return value
    raise SupplementaryDataError(f"Unexpected information regime value: {value}")


def build_msa_snr_readout_correlations(resolved_sources: dict) -> pd.DataFrame:
    """Builds compact MSA-SNR readout correlation summaries from Figure 2 stats."""
    source_df = read_source_sheet(
        resolved_sources,
        "figure2_and_supplementary_figure2_source_data.xlsx",
        "figure2_snr_spearman_stats",
    )
    validate_columns(source_df, ["analysis_group", "x_column", "y_column", "n_msa", "spearman_rho", "p_value"], "figure2_snr_spearman_stats")
    filtered_df = source_df[source_df["x_column"] == "snr_used"].copy()
    if filtered_df.empty:
        raise SupplementaryDataError("No MSA-SNR rows found in figure2_snr_spearman_stats")
    filtered_df = filtered_df.rename(
        columns={
            "x_column": "msa_metric",
            "y_column": "readout",
            "n_msa": "n_msa_designs",
        }
    )
    final_df = filtered_df[["analysis_group", "msa_metric", "readout", "n_msa_designs", "spearman_rho", "p_value"]].copy()
    return clean_dataframe_values(final_df)


def build_local_trace_example_metadata(resolved_sources: dict) -> pd.DataFrame:
    """Builds compact Figure 6 example metadata without anchor sequence export."""
    selected_df = read_source_sheet(resolved_sources, "figure6_source_data.xlsx", "figure6a_hero_selected_samples")
    provenance_df = read_source_sheet(resolved_sources, "figure6_source_data.xlsx", "figure6a_hero_selection_provena")
    validate_columns(
        selected_df,
        ["side", "protein_name", "outer_label", "sample_id", "novelty", "bits_site_min", "selection_metric_value", "plddt_mean_display", "seq_length"],
        "figure6a_hero_selected_samples",
    )
    validate_columns(provenance_df, ["selection_metric_name"], "figure6a_hero_selection_provena")
    selection_metric_name = str(provenance_df["selection_metric_name"].iloc[0])
    final_df = selected_df[
        ["side", "protein_name", "outer_label", "sample_id", "novelty", "bits_site_min", "selection_metric_value", "plddt_mean_display", "seq_length"]
    ].rename(
        columns={
            "side": "example_side",
            "protein_name": "protein_accession",
            "outer_label": "msa_design_label",
            "sample_id": "sample_identifier",
            "novelty": "novelty_percent",
            "bits_site_min": "minimum_bits_per_site",
            "plddt_mean_display": "mean_plddt",
            "seq_length": "sequence_length",
        }
    )
    final_df["selection_rule"] = f"Highest novelty within each selected targeted MSA; MSA pair selected by largest absolute difference in {selection_metric_name}."
    return clean_dataframe_values(final_df)


def build_benchmark_delta_summaries(resolved_sources: dict) -> pd.DataFrame:
    """Combines benchmark delta summary tables into one reader-facing table."""
    common_df = read_source_sheet(resolved_sources, "figure7_and_supplementary_figure5_source_data.xlsx", "delta_summary_common_target")
    gated_df = read_source_sheet(resolved_sources, "figure7_and_supplementary_figure5_source_data.xlsx", "delta_summary_ptm_gated_common")
    required_columns = ["summary_label", "target_percent", "metric", "n_pairs", "mean_delta", "mean_delta_ci_low", "mean_delta_ci_high", "mean_delta_ci_level"]
    validate_columns(common_df, required_columns, "delta_summary_common_target")
    validate_columns(gated_df, required_columns, "delta_summary_ptm_gated_common")
    combined_df = pd.concat([common_df[required_columns], gated_df[required_columns]], ignore_index=True)
    return combined_df.rename(
        columns={
            "summary_label": "analysis_set",
            "target_percent": "target_fraction_percent",
            "mean_delta_ci_low": "ci_low",
            "mean_delta_ci_high": "ci_high",
            "mean_delta_ci_level": "ci_level",
        }
    )


def build_benchmark_paired_tests(resolved_sources: dict) -> pd.DataFrame:
    """Combines benchmark paired test tables into one reader-facing table."""
    common_df = read_source_sheet(resolved_sources, "figure7_and_supplementary_figure5_source_data.xlsx", "paired_stats_common_target")
    gated_df = read_source_sheet(resolved_sources, "figure7_and_supplementary_figure5_source_data.xlsx", "paired_stats_ptm_gated_common")
    required_columns = ["summary_label", "target_percent", "metric", "n_pairs", "mean_delta", "paired_t_p_value", "paired_wilcoxon_p_value"]
    validate_columns(common_df, required_columns, "paired_stats_common_target")
    validate_columns(gated_df, required_columns, "paired_stats_ptm_gated_common")
    combined_df = pd.concat([common_df[required_columns], gated_df[required_columns]], ignore_index=True)
    return combined_df.rename(
        columns={
            "summary_label": "analysis_set",
            "target_percent": "target_fraction_percent",
            "paired_t_p_value": "paired_t_test_p_value",
            "paired_wilcoxon_p_value": "paired_wilcoxon_p_value",
        }
    )


def build_local_benchmark_transition_summary(resolved_sources: dict) -> pd.DataFrame:
    """Builds a compact local benchmark transition summary from Supplementary Figure 5c-d sources."""
    source_c_df = read_source_sheet(resolved_sources, "supplementary_figure5cd_source_data.xlsx", "suppfig5c_source")
    source_d_df = read_source_sheet(resolved_sources, "supplementary_figure5cd_source_data.xlsx", "suppfig5d_source")
    required_columns = [
        "gate",
        "target_range",
        "target_range_label",
        "transition_class",
        "transition_label",
        "delta_error_band",
        "probability",
        "n_pdbs",
        "n_pairs_band_grid",
        "n_residues_band",
        "n_residues_total_band_grid",
        "sequence_window_radius",
        "sequence_window_width",
    ]
    validate_columns(source_c_df, required_columns, "suppfig5c_source")
    validate_columns(source_d_df, required_columns, "suppfig5d_source")
    combined_df = pd.concat([source_c_df[required_columns], source_d_df[required_columns]], ignore_index=True)
    return combined_df.rename(
        columns={
            "gate": "confidence_gate",
            "transition_class": "information_transition",
            "transition_label": "information_transition_label",
            "delta_error_band": "local_error_change_band",
            "n_pairs_band_grid": "n_paired_models",
            "n_residues_band": "n_residue_windows_in_band",
            "n_residues_total_band_grid": "n_residue_windows_total",
        }
    )


def build_local_benchmark_dictionaries(resolved_sources: dict) -> pd.DataFrame:
    """Combines local benchmark dictionaries into one reader-facing dictionary table."""
    transition_df = read_source_sheet(resolved_sources, "supplementary_figure5cd_source_data.xlsx", "information_transition_dict")
    band_df = read_source_sheet(resolved_sources, "supplementary_figure5cd_source_data.xlsx", "delta_error_band_dictionary")
    validate_columns(transition_df, ["transition_class", "transition_label", "description"], "information_transition_dict")
    validate_columns(band_df, ["delta_error_band", "interpretation"], "delta_error_band_dictionary")
    transition_final = transition_df.rename(
        columns={
            "transition_class": "term",
            "transition_label": "label",
            "description": "description",
        }
    )[["term", "label", "description"]]
    transition_final.insert(0, "dictionary", "information transition")
    band_final = band_df.rename(
        columns={
            "delta_error_band": "term",
            "interpretation": "description",
        }
    )[["term", "description"]]
    band_final.insert(1, "label", band_final["term"])
    band_final.insert(0, "dictionary", "local error change band")
    return pd.concat([transition_final, band_final], ignore_index=True)


def validate_columns(dataframe: pd.DataFrame, required_columns: list, context: str) -> None:
    """Validates required source columns for one transformation context."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise SupplementaryDataError(f"Missing columns for {context}: {missing_columns}")


def validate_all_tables(workbook_tables: dict) -> None:
    """Validates final sheets before writing any final workbook."""
    for workbook_key, tables in workbook_tables.items():
        sheet_names = [sheet_name for sheet_name, dataframe in tables]
        duplicate_sheet_names = sorted({sheet_name for sheet_name in sheet_names if sheet_names.count(sheet_name) > 1})
        if duplicate_sheet_names:
            raise SupplementaryDataError(f"Duplicate final sheets in {workbook_key}: {duplicate_sheet_names}")
        for sheet_name, dataframe in tables:
            validate_sheet_name(sheet_name, workbook_key)
            validate_nonempty_table(dataframe, workbook_key, sheet_name)
            validate_final_columns(dataframe, workbook_key, sheet_name)


def validate_sheet_name(sheet_name: str, workbook_key: str) -> None:
    """Validates Excel sheet-name constraints."""
    if len(sheet_name) > 31:
        raise SupplementaryDataError(f"Sheet name exceeds Excel limit in {workbook_key}: {sheet_name}")
    invalid_characters = set("[]:*?/\\")
    if any(character in invalid_characters for character in sheet_name):
        raise SupplementaryDataError(f"Sheet name contains invalid Excel characters in {workbook_key}: {sheet_name}")


def validate_nonempty_table(dataframe: pd.DataFrame, workbook_key: str, sheet_name: str) -> None:
    """Validates that required final tables are not empty."""
    if dataframe.empty:
        raise SupplementaryDataError(f"Final table is empty: {workbook_key}:{sheet_name}")


def validate_final_columns(dataframe: pd.DataFrame, workbook_key: str, sheet_name: str) -> None:
    """Validates reader-facing final column names."""
    duplicate_columns = sorted({column for column in dataframe.columns if list(dataframe.columns).count(column) > 1})
    if duplicate_columns:
        raise SupplementaryDataError(f"Duplicate final columns in {workbook_key}:{sheet_name}: {duplicate_columns}")
    banned_columns_present = sorted(set(dataframe.columns).intersection(banned_final_column_names))
    if banned_columns_present:
        raise SupplementaryDataError(f"Banned internal columns in {workbook_key}:{sheet_name}: {banned_columns_present}")


def write_workbooks(workbook_tables: dict, resolved_sources: dict) -> list:
    """Writes final supplementary workbooks and returns inventory rows."""
    inventory_rows = []
    for workbook_key, file_name in output_workbooks.items():
        output_path = output_folder / file_name
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, dataframe in workbook_tables[workbook_key]:
                dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
                inventory_rows.append(
                    {
                        "final_workbook": file_name,
                        "final_sheet": sheet_name,
                        "n_rows": len(dataframe),
                        "n_columns": len(dataframe.columns),
                        "columns": "; ".join(str(column) for column in dataframe.columns),
                    }
                )
    inventory_rows.extend(build_source_inventory_rows(resolved_sources))
    return inventory_rows


def build_source_inventory_rows(resolved_sources: dict) -> list:
    """Builds source workbook inventory rows."""
    rows = []
    for workbook_name, workbook_path in resolved_sources.items():
        rows.append(
            {
                "final_workbook": "source_inventory",
                "final_sheet": workbook_name,
                "n_rows": "",
                "n_columns": "",
                "columns": str(workbook_path),
            }
        )
    return rows


def write_inventory(inventory_rows: list) -> None:
    """Writes package inventory workbook."""
    inventory_df = pd.DataFrame(inventory_rows)
    output_path = output_folder / "Supplementary_Data_File_Inventory.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        inventory_df.to_excel(writer, sheet_name="inventory", index=False)


if __name__ == "__main__":
    main()
