// Caustics (gl_glass.CausticsEffect): the smoothed rim normal (glass_common.rim_normal) of every pixel on the rim of a
// glass / fluid / mirror body (within RIM_BAND px of its edge: edge distance from the G-buffer, 0 on outlines), into
// an RG16F texture the caustics march reads (u_rim) in place of the stair-stepped sdf gradient. Everything else
// writes 0 and exits early (one or two fetches).
out vec2 f_rim;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    f_rim = vec2(0.0);
    uint id = mat_id(c);
    if (id == 0u || edge_d(c) > RIM_BAND) return;
    int k = kind_of(id);
    if (k != KIND_GLASS && k != KIND_FLUID && k != KIND_MIRROR) return;
    f_rim = rim_normal(c);
}
