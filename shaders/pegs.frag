#version 330 core
// Draws the native-resolution peg grid overlay (peg_layer.py) over the composited frame while a hook is being placed.
// The mask is one channel at window resolution (1.0 on a dot); each mask px covers u_scale fb px (> 1 on a high-DPI
// drawable), fetched whole (nearest) like hud.frag. Blended "over" with alpha = mask * u_color.a. In the composite
// mode the editor's UI layer masks it out (no dots over the hint box, message or cursor).

out vec4 f_color;

uniform sampler2D u_pegs;    // R8 mask; texture row 0 is the top row
uniform sampler2D u_ui;      // the UI layer (screenUI), only read when u_ui_mask is set
uniform float u_fb_height;   // framebuffer height, to flip gl_FragCoord (bottom-left origin) to top-left
uniform float u_scale;       // framebuffer px per mask px
uniform float u_ui_scale;    // framebuffer px per UI surface px
uniform bool u_ui_mask;
uniform vec4 u_color;        // dot colour and opacity

void main() {
    vec2 p = vec2(gl_FragCoord.x, u_fb_height - gl_FragCoord.y);
    ivec2 c = ivec2(floor(p / u_scale));
    if (any(lessThan(c, ivec2(0))) || any(greaterThanEqual(c, textureSize(u_pegs, 0)))) {
        discard;
    }
    float a = texelFetch(u_pegs, c, 0).r * u_color.a;
    if (u_ui_mask) {
        ivec2 q = ivec2(floor(p / u_ui_scale));
        if (all(lessThan(q, textureSize(u_ui, 0)))) {
            a *= 1.0 - texelFetch(u_ui, q, 0).a;
        }
    }
    if (a <= 0.0) {
        discard;
    }
    f_color = vec4(u_color.rgb, a);
}
