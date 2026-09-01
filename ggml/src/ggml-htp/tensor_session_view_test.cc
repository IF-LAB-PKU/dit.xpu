#include "tensor_session_view.h"

#include <cassert>
#include <cstdio>

static void require_condition(bool condition, const char * label) {
    if (!condition) {
        std::fprintf(stderr, "tensor-session-view test failed: %s\n", label);
        __builtin_trap();
    }
}

int main() {
    HtpSessionRuntime default_session{0, 3};
    HtpSessionRuntime reserved_session{1, 7};
    HtpSessionRuntime third_session{2, 8};
    HtpTensorSessionViews views;

    const HtpTensorIdentity tensor{"activation", 42};
    assert(views.ensure_view(tensor, default_session, 11, 4096, 1024, HtpTensorRole::Activation));
    assert(views.ensure_view(tensor, reserved_session, 11, 4096, 1024, HtpTensorRole::Activation));
    assert(views.ensure_view(tensor, third_session, 11, 4096, 1024, HtpTensorRole::Activation));
    assert(views.has_view(tensor, default_session.id));
    assert(views.has_view(tensor, reserved_session.id));
    assert(views.view(tensor, default_session.id).domain == 3);
    assert(views.view(tensor, reserved_session.id).domain == 7);
    assert(views.view(tensor, default_session.id).dsp_address !=
           views.view(tensor, reserved_session.id).dsp_address);
    assert(views.view(tensor, third_session.id).domain == 8);
    assert(views.view(tensor, third_session.id).dsp_address !=
           views.view(tensor, reserved_session.id).dsp_address);

    assert(!views.prepare_input(tensor, default_session.id));
    assert(views.publish_output(tensor, reserved_session.id));
    assert(views.prepare_input(tensor, default_session.id));
    assert(views.view(tensor, default_session.id).producer_epoch == 1);
    assert(views.prepare_input(tensor, third_session.id));
    assert(views.view(tensor, third_session.id).producer_epoch == 1);

    // A consumer must reject an explicitly stale producer epoch instead of
    // silently treating an older cache publication as current.
    require_condition(!views.prepare_input(tensor, default_session.id, 0), "stale epoch rejected");
    require_condition(views.prepare_input(tensor, default_session.id, 1), "current epoch accepted");

    // Releasing one session-local view cannot invalidate another session's
    // view of the same canonical backing allocation.
    require_condition(views.release_view(tensor, reserved_session.id), "release reserved view");
    require_condition(!views.has_view(tensor, reserved_session.id), "reserved view removed");
    require_condition(views.has_view(tensor, default_session.id), "default view retained");
    require_condition(views.has_view(tensor, third_session.id), "third view retained");

    assert(!views.prepare_input(tensor, 99));
    assert(!views.publish_output(tensor, 99));
    return 0;
}
