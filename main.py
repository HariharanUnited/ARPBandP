import json


from utils import *
from visualizer_utils import *
from visualizer import *
from pricing_network_builder import *
from pricing_labeling import *
from data_utils import *
from validator import validate_solution
from aircraft_recovery_cg import CGConfig, solve_root_cg_from_config
from aircraft_recovery_bp import *

def main():
    core_name = "73Y"
    input_dir = Path("data") / "inputs" / core_name
    output_dir = Path("data") / "outputs" / core_name
    pricing_dir = output_dir / "pricing_networks"
    input_dir.mkdir(parents=True, exist_ok=True)
    pricing_dir.mkdir(parents=True, exist_ok=True)

    flights_raw_csv = input_dir / f"flights_{core_name}.csv"
    oos_raw_csv = input_dir / f"oos_tails_{core_name}.csv"
    flights_clean_csv = input_dir / f"flights_{core_name}_clean.csv"
    flights_clean_json = input_dir / f"flights_{core_name}_clean.json"

    flights_csv_to_clean_csv(flights_raw_csv, flights_clean_csv)
    flights_csv_to_json(flights_clean_csv, flights_clean_json)

    metadata_path = input_dir / "metadata.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata.json must be a JSON object: {metadata_path}")

    IN_MINS = False
    if IN_MINS:
        delta_mx = 60 * 24
        min_slack = -60 * 24
    else:
        delta_mx = 3600 * 24
        min_slack = -3600 * 24

    # START_TIME must use the same units as the flights data (minutes vs seconds).
    start_from_meta = metadata.get("start_time")
    if start_from_meta is not None:
        try:
            START_TIME = int(float(start_from_meta))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"metadata.json start_time must be a number: {start_from_meta}"
            ) from exc
    else:
        raise ValueError("metadata.json must contain a start_time field")
    

    flights_out, oos_out = csv_creator_starttime_dependent(
        flights_clean_csv,
        oos_raw_csv,
        start_time=START_TIME,
        output_dir=input_dir,
    )

    flights_json = Path(flights_out).with_suffix(".json")
    flights_csv_to_json(flights_out, flights_json)

    rel_json, _ = preprocess_Flights_for_recovery_withrelstarttime(
        flights_json_path=flights_json,
        oos_csv_path=oos_out,
        start_time=START_TIME,
        output_json_path=flights_json.with_name(f"{flights_json.stem}_rel.json"),
    )

    final_solution_csv = output_dir / (
        "cg_final_solution_mins.csv" if IN_MINS else "cg_final_solution_secs.csv"
    )
    integral_solution_csv = output_dir / (
        "cg_final_solution_mins_integral_root_CG.csv"
        if IN_MINS
        else "cg_final_solution_secs_integral_root_CG.csv"
    )

    # config = CGConfig(
    #     input_json_path=str(rel_json),
    #     oos_csv_path=str(oos_out),
    #     output_dir=str(pricing_dir),
    #     max_cols_per_tail=10,
    #     use_multi_objective=False,
    #     final_full_column_printing_log=True,
    #     back_arc_implementation=True,
    #     min_slack=min_slack,
    #     final_solution_csv_path=str(final_solution_csv),
    #     final_solution_csv_value_tol=1e-6,
    #     final_solution_csv_include_route_meta=False,
    #     final_integral_solution_csv_path=str(integral_solution_csv),
    #     delta_mx=delta_mx,
    #     delta_mx_pruning=True,
    #     init_via_comb_artvar_seedroute=False,
    #     injection_on_first_flight=True,
    #     include_cancellation=False,
    #     path_track_logging=False,
    #     alternate_impute=False,
    #     activate_swap_reaschability_pricing_restriction=True,
    #     reachability_logger=True,
    #     CG_plain_root_integralizer=True,
    # )
    # model, pricer = solve_root_cg_from_config(config)

    bp_config = BPConfig(
        input_json_path=str(rel_json),
        oos_csv_path=str(oos_out),
        output_dir=str(pricing_dir),
        max_cols_per_tail=10,
        use_multi_objective=False,
        back_arc_implementation=True,
        min_slack=min_slack,
        delta_mx=delta_mx,
        delta_mx_pruning=True,
        init_via_comb_artvar_seedroute=False,
        injection_on_first_flight=True,
        include_cancellation=False,
        path_track_logging=False,
        alternate_impute=False,
        activate_swap_reaschability_pricing_restriction=True,
        reachability_logger=True,
        enable_global_dedup=True,
        branching_logger=True,
        route_var_binary=True,
        use_custom_branching=True,
        bp_debug=BPDebugConfig(
            log_pricer_state=False,
            log_node_pricing=True,
            log_candidate_paths=False,
            log_duplicates=False,
            log_pricing_summary=False,
            log_node_info=False,
            log_incumbent=False,
            log_branch_path=False,
            log_constraints=False,
            log_bounds=False,
            log_prune=False,
            log_branching=False,
            validate_paths=False,
        ),
    )
    model, pricer = solve_branch_and_price_from_config(bp_config)

    etr_by_tail = read_oos_etr_by_tail(oos_out)
    model_to_validate = getattr(pricer, "integral_model", None) or model
    validate_solution(
        model_to_validate,
        pricer.bundle,
        etr_by_tail=etr_by_tail,
        enable_validator_logging=False,
        delta_max_default=delta_mx,
        value_tol=1e-6,
    )




if __name__ == "__main__":
    main()
