from ecg_annotator.models.state import AnnotationSession, Point
from ecg_annotator.annotations.mapper import map_red_to_green_with_gaps
from ecg_annotator.utils.time import sec_to_hms


def normalize_points(points: list[Point]) -> list[Point]:
    return sorted(set(points), key=lambda point: point[0])


def _raise_if_unmapped(unmapped, lead_name, color, expected_color, content_start_time):
    for point_time in unmapped:
        time_str = sec_to_hms(point_time, content_start_time)
        raise ValueError(
            f"Lead {lead_name}: {color} line at {time_str} has no corresponding {expected_color} line."
        )


def validate_annotation_session(
    session: AnnotationSession,
    content_start_time,
    distance_threshold: float,
) -> None:
    selected_annotation_type = bool(session.annotation_type)
    if not selected_annotation_type and any(
        not lead_state.ignored for lead_state in session.lead_states
    ):
        raise ValueError(
            "Please select at least one of 'Before', 'After', or 'Both' annotation types."
        )

    prev_len = -1
    for lead_state in session.lead_states:
        lead_state.red_points = normalize_points(lead_state.red_points)
        lead_state.green_points = normalize_points(lead_state.green_points)
        lead_state.blue_points = normalize_points(lead_state.blue_points)

        if lead_state.ignored:
            continue

        if not lead_state.red_points or not lead_state.green_points:
            raise ValueError(
                f"Lead {lead_state.lead_name} requires at least one red and one green line or it should be marked with ignore."
            )

        mapping, unmapped_red, unmapped_green = map_red_to_green_with_gaps(
            lead_state.red_points, lead_state.green_points
        )
        _raise_if_unmapped(unmapped_red, lead_state.lead_name, "red", "green", content_start_time)
        _raise_if_unmapped(unmapped_green, lead_state.lead_name, "green", "red", content_start_time)

        _, _, unmapped_blue = map_red_to_green_with_gaps(
            lead_state.green_points, lead_state.blue_points
        )
        _raise_if_unmapped(unmapped_blue, lead_state.lead_name, "blue", "green", content_start_time)

        for red, green in mapping.items():
            dist = green - red
            green_time_str = sec_to_hms(green, content_start_time)
            red_time_str = sec_to_hms(red, content_start_time)

            if dist < 0:
                raise RuntimeError(
                    f"Lead {lead_state.lead_name}: green line at {green_time_str} is *before* red line at {red_time_str}. This should not happen. Contact support."
                )
            if dist >= distance_threshold:
                raise ValueError(
                    f"Lead {lead_state.lead_name}: distance between red ({red_time_str}) and green ({green_time_str}) "
                    f"is too large ({dist:.3f}s)."
                )

        if prev_len != -1 and len(mapping) != prev_len:
            raise ValueError(
                f"Lead {lead_state.lead_name}: inconsistent number of red-green pairs compared to previous leads."
            )

        prev_len = len(mapping)
