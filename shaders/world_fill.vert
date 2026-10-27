// One instance per filled polygon (pygame.draw.polygon, width 0): a quad over its pixel bounding box.
in ivec4 in_info;  // first vertex in u_verts, vertex count, packed 0xBBGGRR colour, depth layer
in ivec4 in_box;   // min x, min y, max x, max y of its whole-pixel vertices
in ivec2 in_mat;   // material id (lighting on), the body's winding (+1 / -1: which side of each edge is inside)

flat out ivec4 v_info;
flat out ivec4 v_box;
flat out vec3 v_col;
flat out ivec2 v_mat;

void main() {
    gl_Position = to_clip(box_corner(in_box.xy, in_box.zw + 1), in_info.w);
    v_info = in_info;
    v_box = in_box;
    v_col = unpack_rgb(in_info.z);
    v_mat = in_mat;
}
