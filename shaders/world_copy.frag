// A texture copied pixel for pixel into the bound framebuffer (lighting on with no effect modules: the albedo).
uniform sampler2D u_src;

out vec4 f_color;

void main() {
    f_color = texelFetch(u_src, ivec2(gl_FragCoord.xy), 0);
}
