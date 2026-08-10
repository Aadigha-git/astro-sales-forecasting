"""Workforce management utilities (staffing from forecasted demand)."""

from workforce.staffing import (
    StaffingAssumptions,
    StaffingCalculator,
    StaffingResult,
    apply_shrinkage,
    compute_workload,
    agents_for_service_level,
    service_level_achieved,
)

__all__ = [
    "StaffingAssumptions",
    "StaffingCalculator",
    "StaffingResult",
    "apply_shrinkage",
    "compute_workload",
    "agents_for_service_level",
    "service_level_achieved",
]
