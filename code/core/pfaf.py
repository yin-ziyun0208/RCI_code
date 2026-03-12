"""Pfafstetter-code generation for packaged basin folders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import pyogrio

from .basin_io import BasinPaths, iter_global_basin_paths

COMPLETE_BASIN_DIGITS = {"2", "4", "6", "8"}


def build_prefixes_by_level(
    full_codes: Iterable[str],
    max_level: Optional[int] = None,
) -> Dict[int, List[str]]:
    """Build the unique Pfaf prefix set at each level from the full code table."""
    normalized_codes = sorted({str(code) for code in full_codes if str(code)})
    if not normalized_codes:
        return {}

    max_code_length = max(len(code) for code in normalized_codes)
    if max_level is not None:
        max_code_length = min(max_code_length, max_level)

    return {
        level: sorted({code[:level] for code in normalized_codes if len(code) >= level})
        for level in range(1, max_code_length + 1)
    }


def compute_closed_prefix_status(
    full_codes: Iterable[str],
    max_level: Optional[int] = None,
) -> Dict[str, bool]:
    """
    Derive the recursive "closed basin" status of each Pfaf prefix.

    This uses the full set of reach-level Pfaf codes rather than a target-level
    subset. Shorter sibling prefixes must remain visible; otherwise open odd
    prefixes can be misclassified as independent basins at deeper levels.
    """
    prefixes_by_level = build_prefixes_by_level(full_codes, max_level=max_level)
    closed_status: Dict[str, bool] = {"": True}

    for level in sorted(prefixes_by_level):
        sibling_groups: Dict[str, List[str]] = {}
        for prefix in prefixes_by_level[level]:
            sibling_groups.setdefault(prefix[:-1], []).append(prefix)

        for prefix in prefixes_by_level[level]:
            parent_prefix = prefix[:-1]
            digit = prefix[-1]
            parent_closed = closed_status.get(parent_prefix, False)
            sibling_digits = [int(item[-1]) for item in sibling_groups[parent_prefix]]
            odd_digits = [value for value in sibling_digits if value % 2 == 1]
            max_odd_digit = max(odd_digits) if odd_digits else None
            is_closed = digit in COMPLETE_BASIN_DIGITS or (
                parent_closed and max_odd_digit is not None and int(digit) == max_odd_digit
            )
            closed_status[prefix] = is_closed

    return closed_status


def summarize_unit_topology(
    unit_comids: Set[int],
    upstream_map: Dict[int, List[int]],
    downstream_map: Dict[int, Optional[int]],
) -> Dict[str, object]:
    """Summarize outlet count and external inflow for one reach set."""
    outlets = sorted(
        comid for comid in unit_comids if downstream_map.get(comid) not in unit_comids
    )
    external_inflows = []
    external_inflow_count = 0

    for comid in sorted(unit_comids):
        external_upstreams = [
            upstream for upstream in upstream_map.get(comid, []) if upstream not in unit_comids
        ]
        if not external_upstreams:
            continue
        external_inflow_count += len(external_upstreams)
        if len(external_inflows) < 10:
            external_inflows.append(
                {
                    "comid": comid,
                    "external_upstreams": external_upstreams,
                    "downstream_comid": downstream_map.get(comid),
                }
            )

    return {
        "num_reaches": len(unit_comids),
        "outlet_count": len(outlets),
        "outlets": outlets,
        "external_inflow_count": external_inflow_count,
        "external_inflows": external_inflows,
    }


@dataclass(frozen=True)
class TributaryCandidate:
    root_comid: int
    parent_mainstem_comid: int
    parent_mainstem_index: int
    uparea: float


class PfafstetterCoder:
    """Generate reach-level Pfafstetter codes from a closed river-network unit."""

    def __init__(
        self,
        river_df: pd.DataFrame,
        comid_col: str = "COMID",
        uparea_col: str = "uparea",
        down_col: str = "NextDownID",
        up_cols: Optional[List[str]] = None,
        max_level: Optional[int] = None,
        min_unit_reaches: int = 3,
    ) -> None:
        if up_cols is None:
            up_cols = ["up1", "up2", "up3", "up4"]

        self.comid_col = comid_col
        self.uparea_col = uparea_col
        self.down_col = down_col
        self.up_cols = up_cols
        self.max_level = max_level
        self.min_unit_reaches = min_unit_reaches

        required_cols = [self.comid_col, self.uparea_col, self.down_col, *self.up_cols]
        missing_cols = [col for col in required_cols if col not in river_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required river-network columns: {missing_cols}")

        self.river_df = pd.DataFrame(river_df.loc[:, required_cols]).copy()
        self.river_df[self.comid_col] = self.river_df[self.comid_col].astype(int)
        self.river_df[self.uparea_col] = pd.to_numeric(
            self.river_df[self.uparea_col], errors="coerce"
        ).fillna(0.0)
        self.river_df[self.down_col] = pd.to_numeric(
            self.river_df[self.down_col], errors="coerce"
        ).fillna(0).astype(int)
        for up_col in self.up_cols:
            self.river_df[up_col] = pd.to_numeric(
                self.river_df[up_col], errors="coerce"
            ).fillna(0).astype(int)

        self.comids: Set[int] = set(self.river_df[self.comid_col])
        self.uparea_map: Dict[int, float] = dict(
            zip(self.river_df[self.comid_col], self.river_df[self.uparea_col])
        )
        self.downstream_map: Dict[int, Optional[int]] = {}
        self.upstream_map: Dict[int, List[int]] = {}

        topology_cols = [self.comid_col, self.down_col, *self.up_cols]
        for row in self.river_df[topology_cols].itertuples(index=False, name=None):
            comid = int(row[0])
            next_down = int(row[1])
            self.downstream_map[comid] = next_down if next_down in self.comids else None

            upstreams = []
            for upstream_id in row[2:]:
                upstream_id = int(upstream_id)
                if upstream_id > 0 and upstream_id in self.comids:
                    upstreams.append(upstream_id)
            self.upstream_map[comid] = upstreams

        self.codes: Dict[int, str] = {}
        self.subdivision_records: List[Dict[str, object]] = []
        self.terminal_records: List[Dict[str, object]] = []
        self._upstream_closure_cache: Dict[int, frozenset[int]] = {}

    def find_basin_outlet(self) -> int:
        outside = [
            comid for comid, downstream in self.downstream_map.items() if downstream is None
        ]
        if outside:
            return max(outside, key=lambda comid: self.uparea_map[comid])
        return max(self.comids, key=lambda comid: self.uparea_map[comid])

    def find_unit_outlet(self, unit_comids: Set[int]) -> int:
        outlets = [
            comid
            for comid in unit_comids
            if self.downstream_map.get(comid) not in unit_comids
        ]
        if len(outlets) != 1:
            raise ValueError(
                f"Expected exactly one outlet for unit, found {len(outlets)}: {sorted(outlets)[:10]}"
            )
        return outlets[0]

    def upstream_closure(self, comid: int) -> frozenset[int]:
        if comid in self._upstream_closure_cache:
            return self._upstream_closure_cache[comid]

        stack: List[Tuple[int, bool]] = [(comid, False)]
        processing: Set[int] = set()

        while stack:
            current, expanded = stack.pop()
            if current in self._upstream_closure_cache:
                continue
            if not expanded:
                if current in processing:
                    raise ValueError(f"Cycle detected while building upstream closure for COMID {current}")
                processing.add(current)
                stack.append((current, True))
                for upstream in self.upstream_map.get(current, []):
                    if upstream not in self._upstream_closure_cache:
                        stack.append((upstream, False))
                continue

            closure = {current}
            for upstream in self.upstream_map.get(current, []):
                closure.update(self._upstream_closure_cache[upstream])
            self._upstream_closure_cache[current] = frozenset(closure)
            processing.remove(current)

        return self._upstream_closure_cache[comid]

    def trace_mainstem(self, unit_comids: Set[int], outlet_comid: int) -> List[int]:
        down_to_up = [outlet_comid]
        current = outlet_comid
        while True:
            candidates = [
                upstream
                for upstream in self.upstream_map.get(current, [])
                if upstream in unit_comids
            ]
            if not candidates:
                break
            current = max(candidates, key=lambda comid: self.uparea_map.get(comid, -1.0))
            down_to_up.append(current)
        return down_to_up[::-1]

    def collect_tributary_candidates(
        self,
        unit_comids: Set[int],
        mainstem: List[int],
    ) -> List[TributaryCandidate]:
        candidates: List[TributaryCandidate] = []
        for mainstem_index, mainstem_comid in enumerate(mainstem):
            upstream_mainstem = mainstem[mainstem_index - 1] if mainstem_index > 0 else None
            for upstream in self.upstream_map.get(mainstem_comid, []):
                if upstream not in unit_comids:
                    continue
                if upstream_mainstem is not None and upstream == upstream_mainstem:
                    continue
                candidates.append(
                    TributaryCandidate(
                        root_comid=upstream,
                        parent_mainstem_comid=mainstem_comid,
                        parent_mainstem_index=mainstem_index,
                        uparea=self.uparea_map.get(upstream, 0.0),
                    )
                )
        return candidates

    def select_major_tributaries(
        self,
        candidates: List[TributaryCandidate],
    ) -> List[TributaryCandidate]:
        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda item: (-item.uparea, -item.parent_mainstem_index, item.root_comid),
        )[:4]
        return sorted(
            ranked,
            key=lambda item: (-item.parent_mainstem_index, -item.uparea, item.root_comid),
        )

    def first_mainstem_anchor(
        self,
        comid: int,
        unit_comids: Set[int],
        mainstem_set: Set[int],
    ) -> int:
        current = comid
        visited: Set[int] = set()
        while current not in mainstem_set:
            if current in visited:
                raise ValueError(f"Cycle detected while tracing downstream from COMID {comid}")
            visited.add(current)
            next_down = self.downstream_map.get(current)
            if next_down is None or next_down not in unit_comids:
                raise ValueError(
                    f"Reach {comid} leaves the current unit before intersecting the main stem"
                )
            current = next_down
        return current

    def partition_unit(
        self,
        unit_comids: Set[int],
        mainstem: List[int],
    ) -> Tuple[Dict[int, Set[int]], List[TributaryCandidate]]:
        candidates = self.collect_tributary_candidates(unit_comids, mainstem)
        major_tributaries = self.select_major_tributaries(candidates)
        if not major_tributaries:
            return {}, []

        child_units: Dict[int, Set[int]] = {}
        assigned: Set[int] = set()

        for index, tributary in enumerate(major_tributaries):
            digit = 2 * (index + 1)
            tributary_set = set(self.upstream_closure(tributary.root_comid)) & unit_comids
            child_units[digit] = tributary_set
            assigned.update(tributary_set)

        remaining = unit_comids - assigned
        mainstem_set = set(mainstem)
        mainstem_index = {comid: index for index, comid in enumerate(mainstem)}
        thresholds = [tributary.parent_mainstem_index for tributary in major_tributaries]
        headwater_digit = 2 * len(major_tributaries) + 1

        for comid in remaining:
            anchor_comid = self.first_mainstem_anchor(comid, unit_comids, mainstem_set)
            anchor_index = mainstem_index[anchor_comid]
            if anchor_index >= thresholds[0]:
                digit = 1
            elif anchor_index < thresholds[-1]:
                digit = headwater_digit
            else:
                digit = headwater_digit
                for position in range(1, len(thresholds)):
                    lower = thresholds[position]
                    upper = thresholds[position - 1]
                    if lower <= anchor_index < upper:
                        digit = 2 * position + 1
                        break
            child_units.setdefault(digit, set()).add(comid)

        return child_units, major_tributaries

    def assign_code(self, unit_comids: Iterable[int], code: str) -> None:
        for comid in unit_comids:
            if comid in self.codes and self.codes[comid] != code:
                raise ValueError(
                    f"COMID {comid} assigned conflicting codes: {self.codes[comid]} vs {code}"
                )
            self.codes[comid] = code

    def record_terminal_unit(self, prefix: str, unit_comids: Set[int], reason: str) -> None:
        """Record why recursion stopped for a terminal Pfaf unit."""
        self.terminal_records.append(
            {
                "prefix": prefix or "<root>",
                "level": len(prefix),
                "num_reaches": len(unit_comids),
                "reason": reason,
            }
        )

    def recurse(self, unit_comids: Set[int], prefix: str) -> None:
        current_level = len(prefix)
        if not unit_comids:
            return
        if self.max_level is not None and current_level >= self.max_level:
            self.assign_code(unit_comids, prefix)
            self.record_terminal_unit(prefix, unit_comids, "max_level")
            return
        if len(unit_comids) < self.min_unit_reaches:
            self.assign_code(unit_comids, prefix)
            self.record_terminal_unit(prefix, unit_comids, "below_min_unit_reaches")
            return

        outlet = self.find_unit_outlet(unit_comids)
        mainstem = self.trace_mainstem(unit_comids, outlet)
        child_units, major_tributaries = self.partition_unit(unit_comids, mainstem)
        if not child_units:
            self.assign_code(unit_comids, prefix)
            self.record_terminal_unit(prefix, unit_comids, "no_major_tributaries")
            return

        record = {
            "prefix": prefix or "<root>",
            "level": current_level + 1,
            "outlet_comid": outlet,
            "num_reaches": len(unit_comids),
            "child_digits": sorted(child_units),
            "major_tributaries": [
                {
                    "digit": 2 * (index + 1),
                    "root_comid": tributary.root_comid,
                    "parent_mainstem_comid": tributary.parent_mainstem_comid,
                    "parent_mainstem_index": tributary.parent_mainstem_index,
                    "uparea": tributary.uparea,
                    "num_reaches": len(child_units[2 * (index + 1)]),
                }
                for index, tributary in enumerate(major_tributaries)
            ],
        }
        self.subdivision_records.append(record)

        for digit, child_set in sorted(child_units.items()):
            self.recurse(child_set, f"{prefix}{digit}")

    def validate(self) -> Dict[str, object]:
        errors: List[str] = []
        if self.comids != set(self.codes):
            missing = sorted(self.comids - set(self.codes))
            extra = sorted(set(self.codes) - self.comids)
            if missing:
                errors.append(f"Missing codes for {len(missing)} reaches, first few: {missing[:10]}")
            if extra:
                errors.append(f"Unexpected codes for {len(extra)} reaches, first few: {extra[:10]}")

        prefix_counts: Dict[str, int] = {"": len(self.codes)}
        for code in self.codes.values():
            for level in range(1, len(code) + 1):
                prefix = code[:level]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        for record in self.subdivision_records:
            prefix = "" if record["prefix"] == "<root>" else record["prefix"]
            if prefix_counts.get(prefix, 0) != record["num_reaches"]:
                errors.append(
                    f"Prefix {record['prefix']} expected {record['num_reaches']} reaches, "
                    f"but found {prefix_counts.get(prefix, 0)} in final codes"
                )

        prefix_members: Dict[str, Set[int]] = {"": set(self.codes)}
        for comid, code in self.codes.items():
            for level in range(1, len(code) + 1):
                prefix_members.setdefault(code[:level], set()).add(comid)

        closed_status = compute_closed_prefix_status(self.codes.values())
        for prefix, is_closed in sorted(closed_status.items()):
            if not is_closed:
                continue
            unit_comids = prefix_members.get(prefix)
            if not unit_comids:
                continue
            topology = summarize_unit_topology(unit_comids, self.upstream_map, self.downstream_map)
            if topology["outlet_count"] != 1 or topology["external_inflow_count"] != 0:
                display_prefix = prefix or "<root>"
                errors.append(
                    f"Closed prefix {display_prefix} is not topologically closed: "
                    f"outlet_count={topology['outlet_count']}, "
                    f"external_inflow_count={topology['external_inflow_count']}, "
                    f"sample_external_inflows={topology['external_inflows'][:3]}"
                )

        per_level_counts: Dict[int, int] = {}
        for code in self.codes.values():
            per_level_counts[len(code)] = per_level_counts.get(len(code), 0) + 1

        terminal_reason_distribution: Dict[str, int] = {}
        for record in self.terminal_records:
            reason = str(record["reason"])
            terminal_reason_distribution[reason] = terminal_reason_distribution.get(reason, 0) + 1

        return {
            "num_reaches": len(self.comids),
            "num_codes": len(self.codes),
            "max_code_length": max((len(code) for code in self.codes.values()), default=0),
            "min_code_length": min((len(code) for code in self.codes.values()), default=0),
            "num_subdivided_units": len(self.subdivision_records),
            "num_terminal_units": len(self.terminal_records),
            "terminal_reason_distribution": terminal_reason_distribution,
            "code_length_distribution": per_level_counts,
            "errors": errors,
        }

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, object]]:
        root_outlet = self.find_basin_outlet()
        root_unit = set(self.upstream_closure(root_outlet))
        if root_unit != self.comids:
            missing = sorted(self.comids - root_unit)
            raise ValueError(
                "Input basin is not a single closed root unit. "
                f"Root outlet {root_outlet} covers {len(root_unit)} of {len(self.comids)} reaches; "
                f"reaches outside the root closure include {missing[:10]}"
            )

        root_topology = summarize_unit_topology(root_unit, self.upstream_map, self.downstream_map)
        if root_topology["outlet_count"] != 1 or root_topology["external_inflow_count"] != 0:
            raise ValueError(
                "Input basin is not topologically closed as a root unit: "
                f"outlet_count={root_topology['outlet_count']}, "
                f"external_inflow_count={root_topology['external_inflow_count']}"
            )

        self.recurse(root_unit, "")

        codes_df = pd.DataFrame(
            {
                self.comid_col: list(self.codes.keys()),
                "pfafstetter": list(self.codes.values()),
            }
        ).sort_values(self.comid_col)

        validation = self.validate()
        validation["root_outlet_comid"] = root_outlet
        validation["root_num_reaches"] = len(root_unit)
        validation["stop_rules"] = {
            "max_level": "Stop when the requested maximum Pfaf level is reached.",
            "below_min_unit_reaches": (
                "Stop when a unit contains fewer reaches than min_unit_reaches."
            ),
            "no_major_tributaries": (
                "Stop when no tributary branch remains to define a further Pfaf subdivision."
            ),
        }
        validation["subdivision_records"] = self.subdivision_records
        validation["terminal_records"] = self.terminal_records

        level1_summary = []
        for digit in sorted({code[0] for code in self.codes.values() if code}):
            unit = {
                comid for comid, code in self.codes.items() if len(code) >= 1 and code[0] == digit
            }
            level1_summary.append(
                {
                    "digit": digit,
                    "num_reaches": len(unit),
                    "outlet_comid": self.find_unit_outlet(unit),
                    "max_uparea": max(self.uparea_map[comid] for comid in unit),
                }
            )
        validation["level1_summary"] = level1_summary

        return codes_df, validation


def generate_single_basin_pfaf(
    basin: BasinPaths,
    output_csv: Optional[str | Path] = None,
    report_json: Optional[str | Path] = None,
    max_level: Optional[int] = None,
    min_unit_reaches: int = 3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate Pfaf codes for one packaged basin."""
    output_csv = Path(output_csv) if output_csv is not None else basin.basin_dir / "pfaf_codes.csv"
    report_json = Path(report_json) if report_json is not None else basin.basin_dir / "pfaf_report.json"

    required_cols = ["COMID", "uparea", "NextDownID", "up1", "up2", "up3", "up4"]
    river_df = pyogrio.read_dataframe(
        basin.river_network,
        columns=required_cols,
        read_geometry=False,
    )
    coder = PfafstetterCoder(
        river_df=river_df,
        max_level=max_level,
        min_unit_reaches=min_unit_reaches,
    )
    codes_df, validation = coder.generate()
    if validation.get("errors"):
        raise ValueError(
            "Pfaf validation failed; outputs were not written. "
            f"Errors: {validation['errors'][:5]}"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    codes_df.to_csv(output_csv, index=False)
    with open(report_json, "w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)
    return codes_df, validation


def generate_batch_pfaf(
    global_dir: str = "data/basins/global",
    output_dir: str = "data/basins/global",
    max_level: Optional[int] = None,
    min_unit_reaches: int = 3,
    basin_names: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate Pfaf codes across packaged global basins."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    inventory_rows = []
    failure_rows = []
    basins = list(iter_global_basin_paths(global_dir, basin_names=basin_names))

    print("=" * 72)
    print(f"Generating Pfafstetter codes for {len(basins)} basins")
    print("Input mode: packaged data/basins/global folders")
    print("=" * 72)

    for index, basin in enumerate(basins, start=1):
        print(f"\n[{index}/{len(basins)}] {basin.name}")
        try:
            river_df = pyogrio.read_dataframe(
                basin.river_network,
                columns=["COMID", "uparea", "NextDownID", "up1", "up2", "up3", "up4"],
                read_geometry=False,
            )
            codes_df, validation = generate_single_basin_pfaf(
                basin=basin,
                max_level=max_level,
                min_unit_reaches=min_unit_reaches,
            )
            print(f"  Source reaches: {len(river_df)}")
            print(f"  Coded reaches in primary closed unit: {len(codes_df)}")
            print(f"  Output CSV: {basin.basin_dir / 'pfaf_codes.csv'}")
            print(f"  Validation errors: {len(validation.get('errors', []))}")
            inventory_rows.append(
                {
                    "basin_name": basin.name,
                    "river_file": basin.river_network.name,
                    "source_mode": basin.mode,
                    "num_source_reaches": len(river_df),
                    "num_connected_reaches": validation.get("root_num_reaches", len(codes_df)),
                    "num_nonconnected_reaches_skipped": 0,
                    "num_nonroot_reaches_skipped": len(river_df) - len(codes_df),
                    "num_coded_reaches": len(codes_df),
                    "root_outlet_comid": validation.get("root_outlet_comid"),
                    "validation_error_count": len(validation.get("errors", [])),
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "basin_name": basin.name,
                    "river_file": basin.river_network.name,
                    "error": str(exc),
                }
            )

    inventory_df = pd.DataFrame(inventory_rows)
    failures_df = pd.DataFrame(failure_rows)
    inventory_df.to_csv(output_dir_path / "global_pfaf_inventory.csv", index=False)
    failures_df.to_csv(output_dir_path / "global_pfaf_failures.csv", index=False)
    return inventory_df, failures_df
