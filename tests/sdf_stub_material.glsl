// Provisional stand-in for workstream F's shaders/material.glsl, used by tests/sdf_harness.py only while that file
// doesn't exist. Covers just the fields gl_sdf reads. Packing (harness only): texel 1 of row `id` = emissive.rgb,
// emissive_strength.
struct Mat {
    vec3 emissive;
    float emissive_strength;
};

Mat mat_fetch(sampler2D table, uint id) {
    vec4 e = texelFetch(table, ivec2(1, int(id)), 0);
    return Mat(e.rgb, e.a);
}
