// Push-pull inpainting, push (up) pass: a level's own average where it has coverage, the filled coarser level
// (bilinear) where it doesn't, blended by coverage. a = 1 everywhere in the result.
layout(location = 0) out vec4 o_ratio;
layout(location = 1) out vec4 o_col;

uniform sampler2D u_own_ratio;
uniform sampler2D u_own_col;
uniform sampler2D u_coarse_ratio;  // LINEAR
uniform sampler2D u_coarse_col;
uniform bool u_has_coarse;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec4 r = texelFetch(u_own_ratio, c, 0);
    vec4 q = texelFetch(u_own_col, c, 0);
    vec3 cr = vec3(1.0), cc = q.rgb;
    if (u_has_coarse) {
        vec2 uv = (vec2(c) + 0.5) * 0.5 / vec2(textureSize(u_coarse_ratio, 0));
        cr = texture(u_coarse_ratio, uv).rgb;
        cc = texture(u_coarse_col, uv).rgb;
    }
    float wr = clamp(2.0 * r.a, 0.0, 1.0), wc = clamp(2.0 * q.a, 0.0, 1.0);
    o_ratio = vec4(mix(cr, r.rgb, wr), 1.0);
    o_col = vec4(mix(cc, q.rgb, wc), 1.0);
}
