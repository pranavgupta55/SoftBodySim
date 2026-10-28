// Translucent shadows, source pass (gl_sdf.py SdfEffect, only on frames that need them): splits the G-buffer's
// shadow casters (flags bit0 + bit1) into
//   location 0 (RGBA8): the translucent ones (FLAG_TRANSLUCENT, see-through glass / fluid): rgb = the light they
//                       let through (mat_transmittance), a = 1; everything else 0;
//   location 1 (RGBA8): the opaque ones, a = 1.
// Each then gets its own jump flood (sdf_common mode 1): shared["sdf_tr"] (+ the transmittance carried out to the
// nearest translucent body, shared["sdf_tint"]) and shared["sdf_opaque"].
// A translucent body's black border (FLAG_BORDER, its outline) takes the colour of the body's inside next to it
// (white when there is none within BORDER_REACH px): it is the body's edge, not a dark rim. Needs material.glsl.
uniform sampler2D u_normal;   // G-buffer normal (flags in alpha)
uniform sampler2D u_albedo;   // G-buffer albedo
uniform usampler2D u_mat;     // G-buffer material id
uniform sampler2D u_mats;     // material table
// the points view (gl_world _occluder_fill): gbuf["tr"], view-sized, the silhouettes' transmittance (a = 1 on one);
// texel p - u_tr_origin of it is this pass's texel p (the lit-render margin pads the G-buffer read here)
uniform sampler2D u_tr;
uniform bool u_has_tr;
uniform ivec2 u_tr_origin;

const int BORDER_REACH = 3;  // px

layout(location = 0) out vec4 f_tr;
layout(location = 1) out vec4 f_opaque;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int f = int(texelFetch(u_normal, p, 0).a * 255.0 + 0.5);
    f_tr = vec4(0.0);
    f_opaque = vec4(0.0);
    if ((f & (FLAG_OCCLUDER | FLAG_CASTS_SHADOW)) != (FLAG_OCCLUDER | FLAG_CASTS_SHADOW)) return;
    if ((f & FLAG_TRANSLUCENT) == 0) {
        f_opaque = vec4(1.0);
        return;
    }
    if (u_has_tr) {  // the points view: a see-through silhouette's transmittance is in gbuf["tr"]
        ivec2 v = p - u_tr_origin;
        if (all(greaterThanEqual(v, ivec2(0))) && all(lessThan(v, textureSize(u_tr, 0)))) {
            vec4 t = texelFetch(u_tr, v, 0);
            if (t.a > 0.5) {
                f_tr = vec4(t.rgb, 1.0);
                return;
            }
        }
    }
    uint id = texelFetch(u_mat, p, 0).r;
    Mat m = mat_fetch(u_mats, id);
    vec3 col = texelFetch(u_albedo, p, 0).rgb;
    if ((f & FLAG_BORDER) != 0) {
        // the outline: the body's colour from a pixel of its inside nearby (the outline is what the jump flood
        // carries outward, so it must hold the body's tint), else white
        ivec2 size = textureSize(u_normal, 0);
        col = vec3(1.0);
        float best = 1e9;
        for (int dy = -BORDER_REACH; dy <= BORDER_REACH; dy++) {
            for (int dx = -BORDER_REACH; dx <= BORDER_REACH; dx++) {
                ivec2 q = p + ivec2(dx, dy);
                if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) continue;
                int fq = int(texelFetch(u_normal, q, 0).a * 255.0 + 0.5);
                float dq = float(dx * dx + dy * dy);
                if ((fq & (FLAG_TRANSLUCENT | FLAG_BORDER)) == FLAG_TRANSLUCENT && dq < best
                        && texelFetch(u_mat, q, 0).r == id) {
                    best = dq;
                    col = texelFetch(u_albedo, q, 0).rgb;
                }
            }
        }
    }
    // (a pixel flagged translucent whose material id isn't, e.g. none: it blocks, as an opaque one would)
    vec3 t = mat_translucent(m) ? mat_transmittance(m, col) : vec3(0.0);
    f_tr = vec4(t, 1.0);
}
