// Two jump floods at once, seed pass (gl_sdf.py SdfEffect's translucent fields): jfa_seed.frag for two masks in
// one RGBA16I texture, xy = mask 0's seed, zw = mask 1's. Mask k is solid where u_src[k]'s alpha > 0.5 (sdf_solid
// mode 1). Needs sdf_common.glsl.
uniform sampler2D u_src0;
uniform sampler2D u_src1;
uniform vec2 u_period;  // texels, (0, 0) none

out ivec4 f_seed;

ivec2 seed(sampler2D src, ivec2 p, ivec2 size) {
    bool s = sdf_solid(src, p, 1);
    ivec2 offs[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
    for (int i = 0; i < 4; i++) {
        ivec2 q = p + offs[i];
        if (sdf_wrap(q, size, u_period) && sdf_solid(src, q, 1) != s) {
            return p;
        }
    }
    return ivec2(SDF_NONE);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_src0, 0);
    f_seed = ivec4(seed(u_src0, p, size), seed(u_src1, p, size));
}
