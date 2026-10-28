// Two jump floods at once, distance pass (gl_sdf.py SdfEffect's translucent fields): sdf_dist.frag for both seed
// pairs. location 0: mask 0's signed distance (shared["sdf_opaque"]), location 1: mask 1's (shared["sdf_tr"]),
// location 2: mask 1's rgb at its nearest solid texel (the seed, or the solid 4-neighbour of an outside seed;
// white with no seed): every texel gets the transmittance of the translucent body nearest to it
// (shared["sdf_tint"]); location 3: the union's, min(mask 0's, mask 1's) (shared["sdf"] on these frames: exact
// outside both masks; inside one where it touches the other, the distance to that one's own edge). Needs
// sdf_common.glsl.
uniform isampler2D u_seeds;
uniform sampler2D u_src0;
uniform sampler2D u_src1;
uniform vec2 u_period;  // texels, (0, 0) none

layout(location = 0) out float f_dist0;
layout(location = 1) out float f_dist1;
layout(location = 2) out vec4 f_carry;
layout(location = 3) out float f_union;

const float SDF_FAR = 1e4;

float dist(ivec2 s, ivec2 p) {
    return s.x == SDF_NONE ? SDF_FAR : length(sdf_min_image(vec2(s - p), u_period)) + 0.5;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec4 s = texelFetch(u_seeds, p, 0);
    float d0 = dist(s.xy, p), d1 = dist(s.zw, p);
    f_dist0 = sdf_solid(u_src0, p, 1) ? -d0 : d0;
    f_dist1 = sdf_solid(u_src1, p, 1) ? -d1 : d1;
    f_union = min(f_dist0, f_dist1);
    f_carry = vec4(1.0);
    if (s.z != SDF_NONE) {
        ivec2 size = textureSize(u_src1, 0);
        ivec2 offs[5] = ivec2[5](ivec2(0, 0), ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
        for (int i = 0; i < 5; i++) {
            ivec2 q = s.zw + offs[i];
            if (sdf_wrap(q, size, u_period) && sdf_solid(u_src1, q, 1)) {
                f_carry = vec4(texelFetch(u_src1, q, 0).rgb, 1.0);
                break;
            }
        }
    }
}
