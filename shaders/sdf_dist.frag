// Jump flooding, distance pass (gl_sdf.py): the signed distance in texels from each texel centre to the solid
// boundary, from the nearest seed. Seeds are the texels on both sides of the boundary, which lies half a texel
// past them, hence + 0.5 (boundary texels read +-0.5). Negative inside solids. With no seed at all: +-SDF_FAR.
// Needs sdf_common.glsl.
uniform isampler2D u_seeds;
uniform sampler2D u_src;
uniform int u_mode;     // sdf_solid's mode
uniform vec2 u_period;  // texels, (0, 0) none

out float f_dist;

const float SDF_FAR = 1e4;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 s = texelFetch(u_seeds, p, 0).xy;
    float d = s.x == SDF_NONE ? SDF_FAR : length(sdf_min_image(vec2(s - p), u_period)) + 0.5;
    f_dist = sdf_solid(u_src, p, u_mode) ? -d : d;
}
