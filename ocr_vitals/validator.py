"""Physiological range validation for extracted vital signs."""

import logging

from .config import RANGES

logger = logging.getLogger(__name__)


def validate_vitals(vitals: dict) -> tuple:
    """Validate extracted vital signs against physiological ranges.

    Args:
        vitals: Dictionary of extracted vital sign values.

    Returns:
        Tuple of (validation_issues, missing_fields).
        - validation_issues: dict of fields that are out of range
        - missing_fields: list of field names that were not extracted
    """
    validation = {}
    missing_fields = []

    for field, expected_range in RANGES.items():
        value = vitals.get(field)

        if value is None:
            missing_fields.append(field)
            logger.debug("Field '%s' is missing", field)
            continue

        if field == "huyet_ap":
            # Blood pressure has nested validation
            _validate_blood_pressure(value, expected_range, validation, missing_fields)
        else:
            low, high = expected_range
            if not (low <= value <= high):
                validation[field] = {
                    "out_of_range": True,
                    "value": value,
                    "expected": f"{low}-{high}",
                }
                logger.warning(
                    "Field '%s' out of range: %s (expected %s-%s)",
                    field, value, low, high,
                )

    logger.info(
        "Validation complete: %d issues, %d missing fields",
        len(validation), len(missing_fields),
    )
    return validation, missing_fields


def _validate_blood_pressure(bp: dict, ranges: dict, validation: dict, missing_fields: list):
    """Validate blood pressure sub-fields.

    Args:
        bp: Blood pressure dict with tam_thu and tam_truong.
        ranges: Expected ranges for each sub-field.
        validation: Validation issues dict to update.
        missing_fields: Missing fields list to update.
    """
    for sub_field in ("tam_thu", "tam_truong"):
        value = bp.get(sub_field)
        if value is None:
            missing_fields.append(f"huyet_ap.{sub_field}")
            continue

        low, high = ranges[sub_field]
        if not (low <= value <= high):
            key = f"huyet_ap.{sub_field}"
            validation[key] = {
                "out_of_range": True,
                "value": value,
                "expected": f"{low}-{high}",
            }
            logger.warning(
                "Field '%s' out of range: %s (expected %s-%s)",
                key, value, low, high,
            )
