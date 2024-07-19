struct StyleUniform {
    color: vec3<f32>,
    opacity: f32,
    thickness: f32,
};

struct Vertex {
    @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};


@group(0) @binding(2) var<uniform> u_style: StyleUniform;
@group(1) @binding(0) var t_intensity: texture_2d<f32>;


@vertex
fn vs_main(
    in: Vertex,
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
    let u = f32(in.vertex_index & 2);
    let v = f32((in.vertex_index << 1) & 2);
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
    return clamp(intensity, 0.0, 1.0) * u_style.opacity * vec4(u_style.color, 1.0);
}
