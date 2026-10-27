// Jump flooding, seed pass (gl_sdf.py): every texel on either side of a solid / empty boundary (a 4-neighbour with
// the other state) seeds itself, storing its own texel coordinates; all others store SDF_NONE. Neighbours off a
// non-periodic texture edge are skipped, so a body running off screen gets no boundary there.
// Needs sdf_common.glsl.
uniform sampler2D u_src;
uniform int u_mode;     // sdf_solid's mode
uniform vec2 u_period;  // texels, (0, 0) none

out ivec2 f_seed;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_src, 0);
    bool s = sdf_solid(u_src, p, u_mode);
    bool edge = false;
    ivec2 offs[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
    for (int i = 0; i < 4; i++) {
        ivec2 q = p + offs[i];
        if (sdf_wrap(q, size, u_period) && sdf_solid(u_src, q, u_mode) != s) {
            edge = true;
        }
    }
    f_seed = edge ? p : ivec2(SDF_NONE);
}
