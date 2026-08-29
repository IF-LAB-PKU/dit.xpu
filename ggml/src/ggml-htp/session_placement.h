#pragma once

// Returns the session owner for an HTP-only execution ordinal.  The mapping
// is balanced and contiguous: each session owns one interval of ordinals.
static inline int htp_session_owner_for_ordinal(int htp_ordinal,
                                                int htp_node_count,
                                                int session_count) {
    if (session_count <= 1) {
        return 0;
    }
    if (htp_ordinal < 0) {
        return 0;
    }
    if (htp_node_count <= 0 || htp_ordinal >= htp_node_count) {
        return session_count - 1;
    }

    const long long scaled_ordinal = (long long) htp_ordinal * session_count;
    const int owner = (int) (scaled_ordinal / htp_node_count);
    return owner < session_count ? owner : session_count - 1;
}
