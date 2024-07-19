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
    // vertex_index -> uv
    // 0 -> (0, 0)
    // 1 -> (2, 0)
    // 2 -> (0, 2)
    // This triangle covers the screen with uv spanned (0 - 1, 0 - 1).
    let uv = vec2(f32((in.vertex_index << 1) & 2), f32(in.vertex_index & 2));
    return VertexOutput(
        vec4(uv * 2.0 - 1.0, 0.0, 1.0),
        uv,
    );
}


@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let intensity = textureLoad(t_intensity, vec2<u32>(in.uv * vec2<f32>(textureDimensions(t_intensity))), 0).r;
    return clamp(intensity, 0.0, 1.0) * u_style.opacity * vec4(u_style.color, 1.0);
}
