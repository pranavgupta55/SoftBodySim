// PROVISIONAL stand-in for F's shaders/material.glsl (gl_glass.py includes this only while material.glsl doesn't
// exist; tests/glass_harness.py packs its table the same way, see harness_pack). Field names follow materials.Material
// so the code that uses Mat compiles unchanged against F's real struct.
// Packing: 8 RGBA32F texels per material, one row per material id:
//   0 tint.rgb, tint_strength       1 emissive.rgb, emissive_strength
//   2 specular, roughness, bands, bevel_px                3 profile, kind, ior, reflectivity
//   4 transparency, absorption, pattern, pattern_scale    5 pattern_strength, casts_shadow, lit, 0
struct Mat {
    vec3 tint; float tint_strength;
    vec3 emissive; float emissive_strength;
    float specular; float roughness; int bands; float bevel_px;
    int profile; int kind; float ior; float reflectivity;
    float transparency; float absorption; int pattern; float pattern_scale;
    float pattern_strength; bool casts_shadow; bool lit;
};

Mat mat_fetch(sampler2D table, uint id) {
    int row = int(id);
    vec4 t0 = texelFetch(table, ivec2(0, row), 0);
    vec4 t1 = texelFetch(table, ivec2(1, row), 0);
    vec4 t2 = texelFetch(table, ivec2(2, row), 0);
    vec4 t3 = texelFetch(table, ivec2(3, row), 0);
    vec4 t4 = texelFetch(table, ivec2(4, row), 0);
    vec4 t5 = texelFetch(table, ivec2(5, row), 0);
    Mat m;
    m.tint = t0.rgb; m.tint_strength = t0.a;
    m.emissive = t1.rgb; m.emissive_strength = t1.a;
    m.specular = t2.x; m.roughness = t2.y; m.bands = int(t2.z + 0.5); m.bevel_px = t2.w;
    m.profile = int(t3.x + 0.5); m.kind = int(t3.y + 0.5); m.ior = t3.z; m.reflectivity = t3.w;
    m.transparency = t4.x; m.absorption = t4.y; m.pattern = int(t4.z + 0.5); m.pattern_scale = t4.w;
    m.pattern_strength = t5.x; m.casts_shadow = t5.y > 0.5; m.lit = t5.z > 0.5;
    return m;
}
