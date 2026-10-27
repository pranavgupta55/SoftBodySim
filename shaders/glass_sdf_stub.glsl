// Stand-in for S's sdf_at (shaders/sdf_shadow.glsl), included by gl_glass.py only while that file doesn't exist
// (the harness). Same signature: signed distance in render px at p (negative inside occluders), wrapped by period.
float sdf_at(sampler2D sdf, vec2 p, vec2 period) {
    ivec2 size = textureSize(sdf, 0);
    if (period.x > 0.0) p.x = mod(p.x, period.x);
    if (period.y > 0.0) p.y = mod(p.y, period.y);
    ivec2 c = clamp(ivec2(floor(p)), ivec2(0), size - 1);
    return texelFetch(sdf, c, 0).r;
}
