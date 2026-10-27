// The G-buffer as it stands after the lit pass A (everything but the GLASS / FLUID bodies) copied pixel for pixel
// into gbuf["behind"] / ["behind_nrm"] / ["behind_mat"] (gl_world.GLWorld._copy_behind): what lies under a
// see-through body, opaque bodies included.
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform usampler2D u_mat;

layout(location = 0) out vec4 f_albedo;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out uint f_mat;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    f_albedo = texelFetch(u_albedo, p, 0);
    f_normal = texelFetch(u_normal, p, 0);
    f_mat = texelFetch(u_mat, p, 0).r;
}
