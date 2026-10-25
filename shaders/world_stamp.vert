// One instance per pygame.draw.circle: the circle's pixel mask (drawn once by pygame itself into u_atlas) stamped
// at the truncated centre.
in ivec4 in_pos;    // centre x, y (whole px), mask top-left offset from the centre x, y
in ivec4 in_mask;   // mask x, y in the atlas, mask width, height
in ivec2 in_style;  // packed 0xBBGGRR colour, depth layer

flat out ivec2 v_origin;  // the mask's top-left pixel on the framebuffer
flat out ivec2 v_atlas;
flat out vec3 v_col;

void main() {
    ivec2 lo = in_pos.xy + in_pos.zw;
    gl_Position = to_clip(box_corner(lo, lo + in_mask.zw), in_style.y);
    v_origin = lo;
    v_atlas = in_mask.xy;
    v_col = unpack_rgb(in_style.x);
}
