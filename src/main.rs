use std::borrow::Cow;
use std::sync::Arc;

use encase::{ShaderSize, ShaderType};


#[derive(ShaderType)]
struct CameraUniform {
    projection_matrix: mint::ColumnMatrix4<f32>,
    view_matrix: mint::ColumnMatrix4<f32>,
}


#[derive(encase::ShaderType)]
struct ModelUniform {
    model_matrix: mint::ColumnMatrix4<f32>,
}


#[derive(encase::ShaderType)]
struct StyleUniform {
    color: mint::Vector3<f32>,
    opacity: f32,
    thickness: f32,
}


#[derive(encase::ShaderType)]
struct QuadraticBezier {
    position_0: mint::Vector3<f32>,
    position_1: mint::Vector3<f32>,
    position_2: mint::Vector3<f32>,
}


#[derive(encase::ShaderType)]
struct Vertex {
    position_0: mint::Vector3<f32>,
    position_1: mint::Vector3<f32>,
    position_2: mint::Vector3<f32>,
    position: mint::Vector3<f32>,
}


struct State<'window> {
    curve_data: Vec<QuadraticBezier>,
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'window>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    bounding_geometry_pipeline: wgpu::ComputePipeline,
    intensity_pipeline: wgpu::RenderPipeline,
    rendering_pipeline: wgpu::RenderPipeline,
    camera_uniform_buffer: wgpu::Buffer,
    model_uniform_buffer: wgpu::Buffer,
    style_uniform_buffer: wgpu::Buffer,
    quadratic_bezier_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    intensity_texture: wgpu::Texture,
    uniform_bind_group: wgpu::BindGroup,
    storage_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    initial_timestamp: std::time::Instant,
}

impl<'window> State<'window> {
    async fn new(window: Arc<winit::window::Window>) -> Self {
        let curve_data = vec![
            QuadraticBezier {
                position_0: [0.0, 0.0, 0.0].into(),
                position_1: [1.0, 0.0, 0.0].into(),
                position_2: [2.0, 1.0, 0.0].into(),
            },
            QuadraticBezier {
                position_0: [0.0, 0.0, 0.0].into(),
                position_1: [-1.0, 0.0, 0.0].into(),
                position_2: [-2.0, -1.0, 0.0].into(),
            },
            //QuadraticBezier {
            //    position_0: [-2.0, 1.0, 0.0].into(),
            //    position_1: [0.0, -1.0, 0.0].into(),
            //    position_2: [2.0, 1.0, 0.0].into(),
            //},
        ];

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
                    required_features: wgpu::Features::BUFFER_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
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
                .find(|f| f.is_srgb())
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
        let storage_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    &storage_bind_group_layout,
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
                            format: wgpu::TextureFormat::R32Float,//config.format,
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
        };
        let rendering_pipeline = {
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("quadratic_bezier_rendering.wgsl"))),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &uniform_bind_group_layout,
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
        };

        let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: CameraUniform::SHADER_SIZE.into(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
            size: curve_data.len() as u64 * QuadraticBezier::SHADER_SIZE.get(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: curve_data.len() as u64 * 10 * Vertex::SHADER_SIZE.get(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: curve_data.len() as u64 * 20 * wgpu::VertexFormat::Uint32.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });
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
        let storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &storage_bind_group_layout,
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

        let initial_timestamp = std::time::Instant::now();

        Self {
            curve_data,
            window,
            surface,
            config,
            device,
            queue,
            bounding_geometry_pipeline,
            intensity_pipeline,
            rendering_pipeline,
            camera_uniform_buffer,
            model_uniform_buffer,
            style_uniform_buffer,
            quadratic_bezier_buffer,
            vertex_buffer,
            index_buffer,
            intensity_texture,
            uniform_bind_group,
            storage_bind_group,
            texture_bind_group,
            initial_timestamp,
        }
    }

    fn render(&mut self) {
        {
            let seconds = self.initial_timestamp.elapsed().as_secs_f32();
            let aspect_ratio = self.config.width as f32 / self.config.height as f32;
            let camera_uniform = CameraUniform {
                projection_matrix: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.1, 100.0).into(),
                view_matrix: glam::Mat4::look_at_rh(glam::Vec3::Z * 3.0, glam::Vec3::ZERO, glam::Vec3::Y).into(),
            };
            let model_uniform = ModelUniform {
                model_matrix: glam::Mat4::from_quat(glam::Quat::from_rotation_y(seconds)).into(),
            };
            let style_uniform = StyleUniform {
                color: [1.0, 1.0, 1.0].into(),
                opacity: 1.0,
                thickness: 0.2,
            };
            // TODO
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
            quadratic_bezier_cpu_buffer.write(&self.curve_data).unwrap();
            self.queue.write_buffer(&self.quadratic_bezier_buffer, 0, quadratic_bezier_cpu_buffer.as_ref());
        }

        let frame = self.surface.get_current_texture().unwrap();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });

        {
            let mut bounding_geometry_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            bounding_geometry_pass.set_pipeline(&self.bounding_geometry_pipeline);
            bounding_geometry_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            bounding_geometry_pass.set_bind_group(1, &self.storage_bind_group, &[]);
            bounding_geometry_pass.dispatch_workgroups((self.curve_data.len() as u32 + 63) / 64, 1, 1);
        }

        {
            let view = self.intensity_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut intensity_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            intensity_pass.set_pipeline(&self.intensity_pipeline);
            intensity_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            intensity_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            intensity_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            intensity_pass.draw_indexed(0..(20 * self.curve_data.len() as u32), 0, 0..1);
        }

        {
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut rendering_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rendering_pass.set_pipeline(&self.rendering_pipeline);
            rendering_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            rendering_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            rendering_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rendering_pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            // Reconfigure the surface with the new size
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            // On macos the window needs to be redrawn manually after resizing
            self.window.request_redraw();
        }
    }
}

struct App<'window> {
    state: Option<State<'window>>,
}

impl<'window> winit::application::ApplicationHandler for App<'window> {
    fn resumed(&mut self, active_event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let state = pollster::block_on(State::new(Arc::new(active_event_loop.create_window(winit::window::Window::default_attributes()).unwrap())));
        self.state = Some(state);
    }

    fn window_event(&mut self, active_event_loop: &winit::event_loop::ActiveEventLoop, _: winit::window::WindowId, event: winit::event::WindowEvent) {
        let state = self.state.as_mut().unwrap();
        match event {
            winit::event::WindowEvent::Resized(new_size) => {
                state.resize(new_size)
            }
            winit::event::WindowEvent::RedrawRequested => {
                state.render()
            }
            winit::event::WindowEvent::CloseRequested => active_event_loop.exit(),
            _ => {}
        };
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        let state = self.state.as_mut().unwrap();
        state.window.request_redraw();
    }
}


pub fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let _ = event_loop.run_app(&mut App {
        state: None
    });
}
