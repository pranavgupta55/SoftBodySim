// The "behind" layer lit (gl_glass.GlassEffect, before the resolve): what lies under a glass / fluid body
// (gbuf["behind"]: background, grid, trails and every opaque body, F's pass A) lit at its own pixel the way L's
// compose lights the scene:
//     tone(exposure x (behind_albedo * (ambient + light + gi + caustics) + emissive))
// with the light / gi / caustics of that pixel (shared textures; black when missing) and the emissive of the
// behind body (none on its black border, flag bit3). Behind pixels without the lit flag (bit2: past the world clip,
// lit=False) keep their albedo, as compose keeps them. No specular: the light pass computed it for the glass
// surface on top, not for what lies under it. The resolve's env() reads this instead of the push-pull light ratio
// inpainted from the floor around the body (which missed opaque bodies under the glass and took the darkness of
// the body's own cast shadow).
out vec4 f_color;

uniform sampler2D u_behind;
uniform sampler2D u_behind_nrm;
uniform usampler2D u_behind_mat;
uniform sampler2D u_light;
uniform sampler2D u_gi;
uniform sampler2D u_caustics;
uniform vec3 u_ambient;
uniform bool u_tonemap;
uniform float u_knee;
uniform float u_exposure;

// compose.frag's tone(): a hue-preserving shoulder above u_knee plus a +-0.5/255 ordered dither
float bayer4(ivec2 p) {
    int x = p.x & 3, y = p.y & 3;
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[y * 4 + x]) + 0.5) / 16.0;
}

vec3 tone(vec3 col, ivec2 px) {
    col = max(col, vec3(0.0));
    float m = max(col.r, max(col.g, col.b));
    if (m > u_knee) {
        float s = 1.0 - u_knee;
        col *= (u_knee + s * (1.0 - exp(-(m - u_knee) / s))) / m;
    }
    return col + (bayer4(px) - 0.5) / 255.0;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 albedo = texelFetch(u_behind, p, 0);
    uint flags = uint(texelFetch(u_behind_nrm, p, 0).a * 255.0 + 0.5);
    if ((flags & 4u) == 0u || p.x >= u_clip.x || p.y >= u_clip.y) {
        f_color = vec4(albedo.rgb, 1.0);
        return;
    }
    vec3 light = u_ambient + texelFetch(u_light, p, 0).rgb + texelFetch(u_gi, p, 0).rgb
               + texelFetch(u_caustics, p, 0).rgb;
    uint id = texelFetch(u_behind_mat, p, 0).r;
    vec3 emit = vec3(0.0);
    if (id != 0u && (flags & 8u) == 0u) {
        Mat m = mat_fetch(u_table, id);
        emit = m.emissive * m.emissive_strength;
    }
    vec3 col = (albedo.rgb * light + emit) * u_exposure;
    if (u_tonemap) col = tone(col, p);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
