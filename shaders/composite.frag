#version 330 core
// Composites the three low-res pygame layers and upscales them in one pass (instead of three
// pygame.transform.scale calls and blits). Works in framebuffer pixels measured from the top-left, like pygame.
//
// Upscaling is nearest-neighbour via texelFetch on integer texel coordinates, so every screen pixel copies exactly
// one surface pixel (no filtering, no half-texel rounding issues). Texture row 0 is the surface's top row.

out vec4 f_color;

uniform sampler2D u_world;  // the GPU world layer (gl_world.py: background, grid, trails, bodies), if u_world_on
uniform bool u_world_on;
uniform sampler2D u_game;   // screen2: the world (opaque background + bodies), or with u_world_on what pygame still
                            // draws over it (debug overlays, the editor's previews) on a transparent surface
uniform sampler2D u_ghost;  // screenT: translucent editor preview layer
uniform sampler2D u_ui;     // screenUI: panels, cards, cursor (never shaken)

uniform float u_fb_height;    // framebuffer height, to flip gl_FragCoord (bottom-left origin) to top-left
uniform vec2 u_origin;        // top-left corner of the scaled layers, framebuffer px
uniform float u_scale;        // framebuffer px per surface px
uniform vec2 u_shake;         // screen shake offset of the world layers, framebuffer px
uniform float u_ghost_alpha;  // screenT's surface alpha (set_alpha(100) -> 100 / 255)
uniform vec3 u_background;    // shown where no layer covers the screen (the background colour)

vec4 fetch(sampler2D tex, vec2 p) {
    ivec2 c = ivec2(floor(p / u_scale));
    ivec2 size = textureSize(tex, 0);
    if (any(lessThan(c, ivec2(0))) || any(greaterThanEqual(c, size))) {
        return vec4(0.0);
    }
    return texelFetch(tex, c, 0);
}

void main() {
    vec2 p = vec2(gl_FragCoord.x, u_fb_height - gl_FragCoord.y) - u_origin;
    vec3 col = u_background;
    // Same "over" blend pygame uses when blitting a per-pixel-alpha surface: dst = src * a + dst * (1 - a)
    if (u_world_on) {
        // the surfaces are the window size / SCALE_DOWN rounded down, so an odd window leaves a strip under one
        // surface px at the right / bottom: it takes the world's edge px (not u_background, which on a lit world
        // can be far brighter than the lit scene: a light line along the bottom)
        vec2 edge = vec2(textureSize(u_world, 0)) * u_scale;
        vec2 q = p - u_shake;
        q = mix(q, min(q, edge - 0.5), lessThan(q, edge + u_scale));
        vec4 world = fetch(u_world, q);
        col = mix(col, world.rgb, world.a);
    }
    vec4 game = fetch(u_game, p - u_shake);
    col = mix(col, game.rgb, game.a);
    vec4 ghost = fetch(u_ghost, p - u_shake);
    col = mix(col, ghost.rgb, ghost.a * u_ghost_alpha);
    vec4 ui = fetch(u_ui, p);
    col = mix(col, ui.rgb, ui.a);
    f_color = vec4(col, 1.0);
}
