struct StyleUniform {
    intensity_factor: f32,
    thickness: f32,
}

struct Rgb {
    value: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}


@group(0) @binding(2) var<uniform> u_style: StyleUniform;
@group(1) @binding(0) var<storage, read> s_colormap: array<Rgb>;
@group(2) @binding(0) var t_intensity: texture_2d<f32>;


@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    // https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
    // https://gpuweb.github.io/gpuweb/#coordinate-systems
    // | vertex_index | uv     | clip_position  |
    // | ============ | ====== | ============== |
    // | 0            | (0, 0) | (-1, +1, 0, 1) |
    // | 1            | (0, 2) | (-1, -3, 0, 1) |
    // | 2            | (2, 0) | (+3, +1, 0, 1) |
    // Emits a triangle covering the full screen.
    // The vertices are arranged in counterclockwise orientation.
    let u = f32(vertex_index & 2);
    let v = f32((vertex_index << 1) & 2);
    return VertexOutput(
        vec4(2.0 * u - 1.0, 1.0 - 2.0 * v, 0.0, 1.0),
        vec2(u, v),
    );
}


@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let intensity = textureLoad(t_intensity, vec2<u32>(in.uv * vec2<f32>(textureDimensions(t_intensity))), 0).r;
    let rescaled_intensity = clamp(intensity * u_style.intensity_factor, 0.0, 1.0);
    let rgb = s_colormap[u32(rescaled_intensity * 255.0)];
    return vec4(rgb.value, 1.0);
}
