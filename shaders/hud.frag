#version 330 core
// Draws the native-resolution HUD layer (hud_layer.py: the info and compute cards) over the composited frame.
// Rendered with the viewport set to the layer's rect and alpha blending on, so only those pixels are touched.
// Like composite.frag it works in framebuffer pixels from the top-left and fetches whole texels (nearest; the layer
// is 1:1 with the framebuffer unless the drawable is high-DPI, then each layer px covers u_scale fb px).

out vec4 f_color;

uniform sampler2D u_hud;    // the layer; texture row 0 is its top row
uniform float u_fb_height;  // framebuffer height, to flip gl_FragCoord (bottom-left origin) to top-left
uniform vec2 u_origin;      // the layer's top-left, framebuffer px
uniform float u_scale;      // framebuffer px per layer px

void main() {
    vec2 p = vec2(gl_FragCoord.x, u_fb_height - gl_FragCoord.y) - u_origin;
    ivec2 c = ivec2(floor(p / u_scale));
    if (any(lessThan(c, ivec2(0))) || any(greaterThanEqual(c, textureSize(u_hud, 0)))) {
        discard;
    }
    f_color = texelFetch(u_hud, c, 0);
}
