// Modified from an example provided in https://github.com/gfx-rs/wgpu/

mod colormap;

use std::borrow::Cow;
use std::sync::Arc;

use encase::{ShaderSize, ShaderType};


#[derive(ShaderType)]
struct CameraUniform {
    projection_matrix: glam::Mat4,
    view_matrix: glam::Mat4,
}


#[derive(ShaderType)]
struct ModelUniform {
    model_matrix: glam::Mat4,
}


#[derive(ShaderType)]
struct StyleUniform {
    intensity_factor: f32,
    thickness: f32,
}


#[derive(ShaderType)]
struct Rgb {
    value: glam::Vec3,
}


#[derive(ShaderType)]
struct QuadraticBezier {
    position_0: glam::Vec3,
    position_1: glam::Vec3,
    position_2: glam::Vec3,
}


#[derive(ShaderType)]
struct Vertex {
    position_0: glam::Vec3,
    position_1: glam::Vec3,
    position_2: glam::Vec3,
    position: glam::Vec3,
}


struct State<'window> {
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    bounding_geometry_pipeline: wgpu::ComputePipeline,
    intensity_pipeline: wgpu::RenderPipeline,
    frame_pipeline: wgpu::RenderPipeline,
    intensity_texture: wgpu::Texture,
    stencil_texture: wgpu::Texture,
    camera_uniform_buffer: wgpu::Buffer,
    model_uniform_buffer: wgpu::Buffer,
    style_uniform_buffer: wgpu::Buffer,
    quadratic_bezier_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    static_storage_bind_group: wgpu::BindGroup,
    compute_storage_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
}

impl<'window> State<'window> {
    async fn new(window: Arc<winit::window::Window>, curve_len: usize) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: [
                        wgpu::Features::BUFFER_BINDING_ARRAY,
                        wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                        wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    ].iter().copied().reduce(std::ops::BitOr::bitor).unwrap(),
                    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let window_size = window.inner_size();
        let config = {
            let surface_capabilities = surface.get_capabilities(&adapter);
            // Shader code in this tutorial assumes an Srgb surface texture. Using a different
            // one will result all the colors comming out darker. If you want to support non
            // Srgb surfaces, you'll need to account for that when drawing to the frame.
            let surface_format = surface_capabilities
                .formats
                .iter()
                .copied()
                .find(|format| format.is_srgb())
                .unwrap_or(surface_capabilities.formats[0]);
            wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: window_size.width,
                height: window_size.height,
                present_mode: surface_capabilities.present_modes[0],
                alpha_mode: surface_capabilities.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            }
        };
        surface.configure(&device, &config);

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // var<uniform> u_camera: CameraUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(CameraUniform::SHADER_SIZE),
                    },
                    count: None,
                },
                // var<uniform> u_model: ModelUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(ModelUniform::SHADER_SIZE),
                    },
                    count: None,
                },
                // var<uniform> u_style: StyleUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(StyleUniform::SHADER_SIZE),
                    },
                    count: None,
                },
            ],
        });
        let static_storage_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // var<storage, read> s_colormap: array<Rgb>
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Rgb::SHADER_SIZE),
                    },
                    count: None,
                },
            ],
        });
        let compute_storage_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // var<storage, read> s_in: array<QuadraticBezier>
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: Some(QuadraticBezier::SHADER_SIZE),
                    },
                    count: None,
                },
                // var<storage, read_write> s_vertices: array<Vertex>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: Some(Vertex::SHADER_SIZE),
                    },
                    count: None,
                },
                // var<storage, read_write> s_indices: array<u32>
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(wgpu::VertexFormat::Uint32.size()),
                    },
                    count: None,
                },
            ],
        });
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // var t_intensity: texture_2d<f32>;
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: false
                        },
                    },
                    count: None,
                },
            ],
        });

        let bounding_geometry_pipeline = {
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("quadratic_bezier_bounding_geometry.wgsl"))),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &uniform_bind_group_layout,
                    &compute_storage_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "cs_main",
                compilation_options: Default::default(),
            })
        };
        let intensity_pipeline = {
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("quadratic_bezier_intensity.wgsl"))),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &uniform_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vs_main",
                    compilation_options: Default::default(),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: Vertex::SHADER_SIZE.get(),
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    offset: Vertex::METADATA.offset(0),
                                    shader_location: 0,
                                    format: wgpu::VertexFormat::Float32x3,
                                },
                                wgpu::VertexAttribute {
                                    offset: Vertex::METADATA.offset(1),
                                    shader_location: 1,
                                    format: wgpu::VertexFormat::Float32x3,
                                },
                                wgpu::VertexAttribute {
                                    offset: Vertex::METADATA.offset(2),
                                    shader_location: 2,
                                    format: wgpu::VertexFormat::Float32x3,
                                },
                                wgpu::VertexAttribute {
                                    offset: Vertex::METADATA.offset(3),
                                    shader_location: 3,
                                    format: wgpu::VertexFormat::Float32x3,
                                },
                            ],
                        },
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: "fs_main",
                    compilation_options: Default::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R32Float,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                    // or Features::POLYGON_MODE_POINT
                    polygon_mode: wgpu::PolygonMode::Fill,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Stencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState {
                        front: wgpu::StencilFaceState {
                            compare: wgpu::CompareFunction::Always,
                            fail_op: wgpu::StencilOperation::Keep,
                            depth_fail_op: wgpu::StencilOperation::Keep,
                            pass_op: wgpu::StencilOperation::Replace,
                        },
                        back: wgpu::StencilFaceState::IGNORE,
                        read_mask: !0,
                        write_mask: !0,
                    },
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
        };
        let frame_pipeline = {
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("quadratic_bezier_frame.wgsl"))),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &uniform_bind_group_layout,
                    &static_storage_bind_group_layout,
                    &texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vs_main",
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: "fs_main",
                    compilation_options: Default::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: config.format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Stencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState {
                        front: wgpu::StencilFaceState {
                            compare: wgpu::CompareFunction::Equal,
                            fail_op: wgpu::StencilOperation::Keep,
                            depth_fail_op: wgpu::StencilOperation::Keep,
                            pass_op: wgpu::StencilOperation::Keep,
                        },
                        back: wgpu::StencilFaceState::IGNORE,
                        read_mask: !0,
                        write_mask: !0,
                    },
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
        };

        let intensity_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: window_size.width,
                height: window_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: window_size.width,
                height: window_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Stencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: CameraUniform::SHADER_SIZE.into(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,  // TODO: MAP_WRITE ?
            mapped_at_creation: false,
        });
        let model_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ModelUniform::SHADER_SIZE.into(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let style_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: StyleUniform::SHADER_SIZE.into(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let quadratic_bezier_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: curve_len as u64 * QuadraticBezier::SHADER_SIZE.get(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: curve_len as u64 * 10 * Vertex::SHADER_SIZE.get(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: curve_len as u64 * 20 * wgpu::VertexFormat::Uint32.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });
        let colormap_buffer = {
            let colormap_data = colormap::VIRIDIS_COLORMAP.map(|value| Rgb {
                value: glam::Vec3::from_array(value)
            });
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: colormap_data.len() as u64 * Rgb::SHADER_SIZE.get(),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            });
            let mut colormap_cpu_buffer = encase::UniformBuffer::new(Vec::<u8>::new());
            colormap_cpu_buffer.write(&colormap_data).unwrap();
            buffer.slice(..).get_mapped_range_mut().copy_from_slice(colormap_cpu_buffer.as_ref());
            buffer.unmap();
            buffer
        };
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: model_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: style_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let static_storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &static_storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: colormap_buffer.as_entire_binding(),
                },
            ],
        });
        let compute_storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: quadratic_bezier_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: index_buffer.as_entire_binding(),
                },
            ],
        });
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&intensity_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                },
            ],
        });

        Self {
            window,
            surface,
            device,
            queue,
            bounding_geometry_pipeline,
            intensity_pipeline,
            frame_pipeline,
            intensity_texture,
            stencil_texture,
            camera_uniform_buffer,
            model_uniform_buffer,
            style_uniform_buffer,
            quadratic_bezier_buffer,
            vertex_buffer,
            index_buffer,
            uniform_bind_group,
            static_storage_bind_group,
            compute_storage_bind_group,
            texture_bind_group,
        }
    }

    fn render(
        &mut self,
        camera_uniform: &CameraUniform,
        model_uniform: &ModelUniform,
        style_uniform: &StyleUniform,
        curve_data: &[QuadraticBezier],
    ) {
        {
            let mut camera_uniform_cpu_buffer = encase::UniformBuffer::new(Vec::<u8>::new());
            camera_uniform_cpu_buffer.write(&camera_uniform).unwrap();
            self.queue.write_buffer(&self.camera_uniform_buffer, 0, camera_uniform_cpu_buffer.as_ref());
            let mut model_uniform_cpu_buffer = encase::UniformBuffer::new(Vec::<u8>::new());
            model_uniform_cpu_buffer.write(&model_uniform).unwrap();
            self.queue.write_buffer(&self.model_uniform_buffer, 0, model_uniform_cpu_buffer.as_ref());
            let mut style_uniform_cpu_buffer = encase::UniformBuffer::new(Vec::<u8>::new());
            style_uniform_cpu_buffer.write(&style_uniform).unwrap();
            self.queue.write_buffer(&self.style_uniform_buffer, 0, style_uniform_cpu_buffer.as_ref());
            let mut quadratic_bezier_cpu_buffer = encase::StorageBuffer::new(Vec::<u8>::new());
            quadratic_bezier_cpu_buffer.write(curve_data).unwrap();
            self.queue.write_buffer(&self.quadratic_bezier_buffer, 0, quadratic_bezier_cpu_buffer.as_ref());
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        let frame = self.surface.get_current_texture().unwrap();
        let intensity_view = self.intensity_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let stencil_view = self.stencil_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut bounding_geometry_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            bounding_geometry_pass.set_pipeline(&self.bounding_geometry_pipeline);
            bounding_geometry_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            bounding_geometry_pass.set_bind_group(1, &self.compute_storage_bind_group, &[]);
            bounding_geometry_pass.dispatch_workgroups((curve_data.len() as u32 + 63) / 64, 1, 1);
        }

        {
            let mut intensity_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &intensity_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &stencil_view,
                    depth_ops: None,
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            intensity_pass.set_pipeline(&self.intensity_pipeline);
            intensity_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            intensity_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            intensity_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            intensity_pass.set_stencil_reference(1);
            intensity_pass.draw_indexed(0..(20 * curve_data.len() as u32), 0, 0..1);
        }

        {
            let mut frame_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &stencil_view,
                    depth_ops: None,
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            frame_pass.set_pipeline(&self.frame_pipeline);
            frame_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            frame_pass.set_bind_group(1, &self.static_storage_bind_group, &[]);
            frame_pass.set_bind_group(2, &self.texture_bind_group, &[]);
            frame_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            frame_pass.set_stencil_reference(1);
            frame_pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}


struct App {
    window_size: winit::dpi::PhysicalSize<u32>,
    state: Option<State<'static>>,
    curve_data: Vec<QuadraticBezier>,
    uniform_data_getter: Box<dyn Fn(f32) -> (CameraUniform, ModelUniform, StyleUniform)>,
    initial_timestamp: std::time::Instant,
}

impl App {
    fn new(
        window_size: winit::dpi::PhysicalSize<u32>,
        curve_data: Vec<QuadraticBezier>,
        uniform_data_getter: Box<dyn Fn(f32) -> (CameraUniform, ModelUniform, StyleUniform)>,
    ) -> Self {
        Self {
            window_size,
            state: None,
            curve_data,
            uniform_data_getter,
            initial_timestamp: std::time::Instant::now(),
        }
    }

    fn run(&mut self) {
        let event_loop = winit::event_loop::EventLoop::new().unwrap();
        let _ = event_loop.run_app(self);
    }
}

impl winit::application::ApplicationHandler for App {
    fn resumed(
        &mut self,
        active_event_loop: &winit::event_loop::ActiveEventLoop,
    ) {
        if self.state.is_some() {
            return;
        }
        let state = pollster::block_on(State::new(
            Arc::new(active_event_loop.create_window(winit::window::Window::default_attributes().with_inner_size(self.window_size)).unwrap()),
            self.curve_data.len(),
        ));
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        active_event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        match event {
            winit::event::WindowEvent::RedrawRequested => {
                let seconds = self.initial_timestamp.elapsed().as_secs_f32();
                let (camera_uniform, model_uniform, style_uniform) = (self.uniform_data_getter)(seconds);
                state.render(
                    &camera_uniform,
                    &model_uniform,
                    &style_uniform,
                    self.curve_data.as_slice(),
                )
            },
            winit::event::WindowEvent::CloseRequested => {
                active_event_loop.exit()
            },
            _ => {}
        };
    }

    fn about_to_wait(
        &mut self,
        _: &winit::event_loop::ActiveEventLoop,
    ) {
        let state = self.state.as_mut().unwrap();
        state.window.request_redraw();
    }
}


fn linspace(start: f32, end: f32, n_step: usize) -> Vec<f32> {
    assert!(n_step != 0);
    let step = (end - start) / n_step as f32;
    (0..=n_step).map(|index| {
        start + index as f32 * step
    }).collect()
}

fn circle_segments(
    origin: glam::Vec3,
    radius: glam::Vec3,
    normal: glam::Vec3,
) -> Vec<QuadraticBezier> {
    const N_SEGMENTS: usize = 16;
    let transform = glam::Affine3A::from_cols(
        radius.into(),
        normal.normalize().cross(radius).into(),
        glam::Vec3::ZERO.into(),
        origin.into(),
    );
    linspace(0.0, std::f32::consts::TAU, N_SEGMENTS).as_slice().windows(2).map(|window| {
        let theta_0 = window[0];
        let theta_2 = window[1];
        let theta_1 = (theta_0 + theta_2) / 2.0;
        let d_theta = (theta_2 - theta_0) / 2.0;
        let (sin_theta_0, cos_theta_0) = theta_0.sin_cos();
        let (sin_theta_2, cos_theta_2) = theta_2.sin_cos();
        let (sin_theta_1, cos_theta_1) = theta_1.sin_cos();
        QuadraticBezier {
            position_0: transform.transform_point3(cos_theta_0 * glam::Vec3::X + sin_theta_0 * glam::Vec3::Y),
            position_1: transform.transform_point3((cos_theta_1 * glam::Vec3::X + sin_theta_1 * glam::Vec3::Y) / d_theta.cos()),
            position_2: transform.transform_point3(cos_theta_2 * glam::Vec3::X + sin_theta_2 * glam::Vec3::Y),
        }
    }).collect()
}


#[allow(unused)]
fn example_curve() -> App {
    App::new(
        winit::dpi::PhysicalSize {
            width: 1600,
            height: 900,
        },
        vec![
            QuadraticBezier {
                position_0: glam::Vec3::new(0.0, 0.0, 0.0),
                position_1: glam::Vec3::new(1.0, 0.0, 0.0),
                position_2: glam::Vec3::new(2.0, 1.0, 0.0),
            },
        ],
        Box::new(|_| {
            let camera_uniform = CameraUniform {
                projection_matrix: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 100.0),
                view_matrix: glam::Mat4::look_at_rh(2.0 * glam::Vec3::Z, glam::Vec3::ZERO, glam::Vec3::Y),
            };
            let model_uniform = ModelUniform {
                model_matrix: glam::Mat4::IDENTITY,
            };
            let style_uniform = StyleUniform {
                intensity_factor: 1.0,
                thickness: 0.2,
            };
            (camera_uniform, model_uniform, style_uniform)
        }),
    )
}


#[allow(unused)]
fn example_rotating_curve() -> App {
    App::new(
        winit::dpi::PhysicalSize {
            width: 1600,
            height: 900,
        },
        vec![
            QuadraticBezier {
                position_0: glam::Vec3::new(0.0, 0.0, 0.0),
                position_1: glam::Vec3::new(1.0, 0.0, 0.0),
                position_2: glam::Vec3::new(2.0, 1.0, 0.0),
            },
            QuadraticBezier {
                position_0: glam::Vec3::new(0.0, 0.0, 0.0),
                position_1: glam::Vec3::new(-1.0, 0.0, 0.0),
                position_2: glam::Vec3::new(-2.0, -1.0, 0.0),
            },
        ],
        Box::new(|seconds| {
            let camera_uniform = CameraUniform {
                projection_matrix: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 100.0),
                view_matrix: glam::Mat4::look_at_rh(3.0 * glam::Vec3::Z, glam::Vec3::ZERO, glam::Vec3::Y),
            };
            let model_uniform = ModelUniform {
                model_matrix: glam::Mat4::from_quat(glam::Quat::from_rotation_y(seconds)),
            };
            let style_uniform = StyleUniform {
                intensity_factor: 1.0,
                thickness: 0.2,
            };
            (camera_uniform, model_uniform, style_uniform)
        }),
    )
}


#[allow(unused)]
fn example_circle() -> App {
    App::new(
        winit::dpi::PhysicalSize {
            width: 1600,
            height: 900,
        },
        circle_segments(glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Z),
        Box::new(|seconds| {
            let camera_uniform = CameraUniform {
                projection_matrix: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 100.0),
                view_matrix: glam::Mat4::look_at_rh(glam::Vec3::Z * 3.0, glam::Vec3::ZERO, glam::Vec3::Y),
            };
            let model_uniform = ModelUniform {
                model_matrix: glam::Mat4::IDENTITY,
            };
            let style_uniform = StyleUniform {
                intensity_factor: 1.0,
                thickness: 0.2,
            };
            (camera_uniform, model_uniform, style_uniform)
        }),
    )
}


#[allow(unused)]
fn example_sphere_mesh() -> App {
    const N_LONGITUDE: usize = 6;
    const N_LATITUDE: usize = 7;
    App::new(
        winit::dpi::PhysicalSize {
            width: 1600,
            height: 900,
        },
        linspace(0.0, std::f32::consts::PI, N_LONGITUDE).iter().skip(1).map(|longitude| {
            let (sin_longitude, cos_longitude) = longitude.sin_cos();
            circle_segments(glam::Vec3::ZERO, glam::Vec3::Y, sin_longitude * glam::Vec3::X + cos_longitude * glam::Vec3::Z)
        }).flatten().chain(
            linspace(0.0, std::f32::consts::PI, N_LATITUDE).iter().map(|latitude| {
                let (sin_latitude, cos_latitude) = latitude.sin_cos();
                circle_segments(cos_latitude * glam::Vec3::Y, sin_latitude * glam::Vec3::X, glam::Vec3::Y)
            }).flatten()
        ).collect(),
        Box::new(|seconds| {
            let camera_uniform = CameraUniform {
                projection_matrix: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 100.0).into(),
                view_matrix: glam::Mat4::look_at_rh(glam::Vec3::Z * 1.5, glam::Vec3::ZERO, glam::Vec3::Y).into(),
            };
            let model_uniform = ModelUniform {
                model_matrix: glam::Mat4::from_quat(glam::Quat::from_rotation_y(seconds) * glam::Quat::from_rotation_x(0.75 * seconds)).into(),
            };
            let style_uniform = StyleUniform {
                intensity_factor: 0.5,
                thickness: 0.02,
            };
            (camera_uniform, model_uniform, style_uniform)
        }),
    )
}


fn main() {
    env_logger::init();
    let mut app = example_sphere_mesh();
    app.run();
}
