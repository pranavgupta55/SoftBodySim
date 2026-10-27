// gl_lights.py's stand-in for sdf_shadow.glsl when there is no shared["sdf"] (gl_sdf absent): everything is lit.
float sdf_shadow(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period) {
    return 1.0;
}
