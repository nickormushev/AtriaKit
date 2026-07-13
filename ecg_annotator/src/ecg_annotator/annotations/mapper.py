def map_red_to_green_with_gaps(red_points, green_points):
    """
    Match red → green with constraint: no red OR green lies between them.
    Returns:
        mapped_red_to_green, unmapped_red_indices, unmapped_green_indices
    """

    reds = sorted([x for (x, _) in red_points])
    greens = sorted([x for (x, _) in green_points])

    mapped = {}
    r_idx = 0
    g_idx = 0

    unmapped_red = set()
    unmapped_green = set()

    while r_idx < len(reds) and g_idx < len(greens):
        rx = reds[r_idx]
        gx = greens[g_idx]

        if gx <= rx:
            unmapped_green.add(gx)
            g_idx += 1
            continue

        has_red_between = False
        if r_idx + 1 < len(reds):
            next_rx = reds[r_idx + 1]
            if next_rx < gx:
                has_red_between = True

        if has_red_between:
            r_idx += 1
            unmapped_red.add(rx)
            continue

        mapped[rx] = gx
        r_idx += 1
        g_idx += 1

    for i in range(r_idx, len(reds)):
        unmapped_red.add(reds[i])

    for i in range(g_idx, len(greens)):
        unmapped_green.add(greens[i])

    return mapped, unmapped_red, unmapped_green
