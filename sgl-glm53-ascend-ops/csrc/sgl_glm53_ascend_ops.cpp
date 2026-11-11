// GLM-5.3 Ascend ops — torch namespace registration.
//
// Creates the `sgl_glm53_ascend` torch op namespace. Every op extends it via
// TORCH_LIBRARY_FRAGMENT in its own TU; this is the only non-fragment
// declaration, so the namespace has a neutral home that no op owns.
//
// No op schemas are defined here. When adding a new op, put its schema in its
// own host TU as TORCH_LIBRARY_FRAGMENT.

#include <torch/extension.h>

TORCH_LIBRARY(sgl_glm53_ascend, m) {
    (void)m;
}
