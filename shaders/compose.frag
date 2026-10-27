// Compose (gl_lights.py, effect order 300): the lit scene into GLWorld.tex.
//     out = albedo * (ambient + light + gi + caustics) + spec + emissive
// Unlit pixels (flag bit2 clear: background, grid, trails, lit=False bodies) keep their albedo unchanged. Missing
// inputs are bound to a black texture. With u_levels > 0 the light term is quantised to that many levels per unit
// with a 4x4 Bayer dither (gl_lights.LIGHT_LEVELS).
// Included before it: material.glsl (Mat, mat_fetch).
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform usampler2D u_mat;
uniform sampler2D u_mat_table;
uniform sampler2D u_light;
uniform sampler2D u_spec;
uniform sampler2D u_gi;
uniform sampler2D u_caustics;
uniform vec3 u_ambient;
uniform float u_levels;

out vec4 f_color;

float bayer4(ivec2 p) {
    int x = p.x & 3, y = p.y & 3;
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[y * 4 + x]) + 0.5) / 16.0;
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec4 albedo = texelFetch(u_albedo, px, 0);
    uint flags = uint(texelFetch(u_normal, px, 0).a * 255.0 + 0.5);
    if ((flags & 4u) == 0u) {
        f_color = albedo;
        return;
    }
    vec3 light = u_ambient + texelFetch(u_light, px, 0).rgb + texelFetch(u_gi, px, 0).rgb
               + texelFetch(u_caustics, px, 0).rgb;
    if (u_levels > 0.0) {
        light = floor(light * u_levels + bayer4(px)) / u_levels;
    }
    Mat m = mat_fetch(u_mat_table, texelFetch(u_mat, px, 0).r);
    vec3 col = albedo.rgb * light + texelFetch(u_spec, px, 0).rgb + m.emissive * m.emissive_strength;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
