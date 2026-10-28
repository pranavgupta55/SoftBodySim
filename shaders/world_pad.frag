// The lit-render margin (gl_world.GLWorld._margin): the view's G-buffer copied into the middle of the padded
// occluder G-buffer (GLWorld.occ), over the bodies drawn there for the margin around it. Texel p of the padded
// textures is render px p - u_origin; texels outside the view [0, u_clip) keep what the margin draw left.
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform usampler2D u_mat;
uniform ivec2 u_origin;
uniform ivec2 u_view;  // the view's size (GLWorld's u_clip), px

layout(location = 0) out vec4 f_albedo;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out uint f_mat;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) - u_origin;
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, u_view))) discard;
    f_albedo = texelFetch(u_albedo, p, 0);
    f_normal = texelFetch(u_normal, p, 0);
    f_mat = texelFetch(u_mat, p, 0).r;
}
