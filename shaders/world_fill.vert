// One instance per filled polygon (pygame.draw.polygon, width 0): a quad over its pixel bounding box.
in ivec4 in_info;  // first vertex in u_verts, vertex count, packed 0xBBGGRR colour, depth layer
in ivec4 in_box;   // min x, min y, max x, max y of its whole-pixel vertices

flat out ivec4 v_info;
flat out ivec4 v_box;
flat out vec3 v_col;

void main() {
    gl_Position = to_clip(box_corner(in_box.xy, in_box.zw + 1), in_info.w);
    v_info = in_info;
    v_box = in_box;
    v_col = unpack_rgb(in_info.z);
}
