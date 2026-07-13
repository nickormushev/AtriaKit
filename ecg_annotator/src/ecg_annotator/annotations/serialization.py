from ecg_annotator.annotations.state import AnnotationSession
from ecg_annotator.annotations.mapper import map_red_to_green_with_gaps


def build_annotation_rows(
    session: AnnotationSession,
    patient_id,
    patient_file,
    time_to_sample,
) -> list[dict]:
    rows_to_add = []
    annotation_type = session.annotation_type or "Unspecified"

    for lead_state in session.lead_states:
        if lead_state.ignored:
            rows_to_add.append(
                {
                    "patient_id": patient_id,
                    "lead": lead_state.lead_name,
                    "p_wave_id": -1,
                    "onset": -1,
                    "offset": -1,
                    "qrs_onset": -1,
                    "ignore": True,
                    "type": "Ignored",
                    "comment": session.comment,
                    "confidence": lead_state.confidence,
                    "file_path": patient_file,
                }
            )
            continue

        map_red_to_green, _, _ = map_red_to_green_with_gaps(
            lead_state.red_points, lead_state.green_points
        )
        map_green_to_blue, _, _ = map_red_to_green_with_gaps(
            lead_state.green_points, lead_state.blue_points
        )

        for p_idx, (onset, offset) in enumerate(map_red_to_green.items()):
            if onset >= offset:
                continue

            qrs_onset = -1
            if offset in map_green_to_blue:
                qrs_onset = time_to_sample(map_green_to_blue[offset])

            rows_to_add.append(
                {
                    "patient_id": patient_id,
                    "lead": lead_state.lead_name,
                    "p_wave_id": p_idx,
                    "onset": time_to_sample(onset),
                    "offset": time_to_sample(offset),
                    "qrs_onset": qrs_onset,
                    "ignore": False,
                    "comment": session.comment,
                    "type": annotation_type,
                    "confidence": lead_state.confidence,
                    "file_path": patient_file,
                }
            )

    return rows_to_add
