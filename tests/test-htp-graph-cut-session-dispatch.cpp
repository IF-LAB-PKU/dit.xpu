#include "../ggml/src/ggml-htp/htp_session_dispatch.h"
#include "../ggml/src/ggml-htp/tensor_session_view.h"

#include <cassert>
#include <utility>
#include <vector>

static std::vector<int> dispatch_owners(int session_count, int htp_nodes) {
    HtpSessionDispatchContext context(session_count);
    context.begin_graph_view(htp_nodes, 1);

    std::vector<int> owners;
    owners.reserve(static_cast<size_t>(htp_nodes));
    for (int ordinal = 0; ordinal < htp_nodes; ++ordinal) {
        const int owner = context.next_htp_owner();
        assert(context.valid_session(owner));
        assert(context.acquire(owner));
        owners.push_back(owner);
    }
    return owners;
}

static std::pair<int, std::vector<std::vector<int>>> dispatch_segments(
    int session_count, const std::vector<int>& segment_htp_nodes,
    int staging_owner_hint) {
    HtpSessionDispatchContext context(session_count);
    std::vector<std::vector<int>> owners;
    owners.reserve(segment_htp_nodes.size());
    int graph_views = 0;
    for (size_t segment = 0; segment < segment_htp_nodes.size(); ++segment) {
        context.begin_graph_view(segment_htp_nodes[segment],
                                 static_cast<size_t>(staging_owner_hint + segment));
        ++graph_views;
        std::vector<int> segment_owners;
        for (int ordinal = 0; ordinal < segment_htp_nodes[segment]; ++ordinal) {
            const int owner = context.next_htp_owner();
            assert(context.valid_session(owner));
            assert(context.acquire(owner));
            segment_owners.push_back(owner);
        }
        owners.push_back(std::move(segment_owners));
    }
    return {graph_views, std::move(owners)};
}

int main() {
    // A graph view is only a restricted node list: it must use exactly the
    // same ordinal placement policy as the equivalent full graph.
    const auto full_graph = dispatch_owners(4, 12);
    const auto graph_cut  = dispatch_owners(4, 12);
    assert(full_graph == graph_cut);
    const std::vector<int> expected = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    assert(full_graph == expected);

    // The graph-cut segment index does not select an owner.  Each segment
    // starts a fresh view and remains bounded by the initialized registry.
    const auto second_segment = dispatch_owners(4, 12);
    assert(second_segment == expected);

    // The resolved SD3.5 plan has two real segments. Staging owner metadata
    // is mapping information only and cannot turn them into hidden windows or
    // select one owner for an entire segment.
    const auto plan_with_owner_zero = dispatch_segments(4, {8, 8}, 0);
    const auto plan_with_owner_three = dispatch_segments(4, {8, 8}, 3);
    assert(plan_with_owner_zero.first == 2);
    assert(plan_with_owner_three.first == 2);
    assert(plan_with_owner_zero.second == plan_with_owner_three.second);
    for (const auto& segment_owners : plan_with_owner_zero.second) {
        assert(segment_owners == std::vector<int>({0, 0, 1, 1, 2, 2, 3, 3}));
    }

    HtpSessionRuntime session0{0, 3};
    HtpSessionRuntime session1{1, 7};
    HtpTensorSessionViews views;
    const HtpTensorIdentity tensor{"cut.output", 99};
    assert(views.ensure_view(tensor, session0, 10, 0, 4096, HtpTensorRole::Output));
    assert(views.ensure_view(tensor, session1, 10, 0, 4096, HtpTensorRole::Input));
    assert(views.publish_output(tensor, session0.id));
    assert(views.prepare_input(tensor, session1.id, 1));
    assert(views.view(tensor, session1.id).producer_epoch == 1);
    assert(views.release_view(tensor, session0.id));
    assert(views.has_view(tensor, session1.id));
    assert(views.release_view(tensor, session1.id));
    assert(!views.has_view(tensor, session1.id));
    return 0;
}
