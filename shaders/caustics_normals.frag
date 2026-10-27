// Caustics (gl_glass.CausticsEffect): the smoothed rim normal (glass_common.rim_normal) of every pixel on the rim of a
// glass / fluid / mirror body (within RIM_BAND px of its edge: edge distance from the G-buffer, 0 on outlines), into
// an RG16F texture the caustics march reads (u_rim) in place of the stair-stepped sdf gradient. Everything else
// writes 0 and exits early (one or two fetches).
//
// Glass / fluid less than 2 px thick (one pixel wide along some direction: a sliver, or a zero-area body that is only
// its outline) has no boundary normal worth refracting on (rim_normal finds no bevel, the sdf gradient of a 1 px line
// points every which way, which fanned the light into a starburst of streaks): it gets vec2(RIM_THIN_MARK, 0), and
// glass_common.hit_normal lets rays cross it unbent.
out vec2 f_rim;

bool thin(ivec2 c, uint id) {
    for (int k = 0; k < 4; k++) {
        ivec2 d = k == 0 ? ivec2(1, 0) : (k == 1 ? ivec2(0, 1) : (k == 2 ? ivec2(1, 1) : ivec2(1, -1)));
        if (mat_id(pixi(c + d)) != id && mat_id(pixi(c - d)) != id) return true;
    }
    return false;
}

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    f_rim = vec2(0.0);
    uint id = mat_id(c);
    if (id == 0u || edge_d(c) > RIM_BAND) return;
    int k = kind_of(id);
    if (k != KIND_GLASS && k != KIND_FLUID && k != KIND_MIRROR) return;
    if (k != KIND_MIRROR && thin(c, id)) {
        f_rim = vec2(RIM_THIN_MARK, 0.0);
        return;
    }
    f_rim = rim_normal(c);
}
