#![allow(dead_code)]

mod common; #[allow(unused_imports)] pub use common::*;
mod teapot; #[allow(unused_imports)] pub use teapot::*;
mod vector; #[allow(unused_imports)] pub use vector::*;

use std::sync::Arc;

use winit::{application::ApplicationHandler, dpi::{PhysicalPosition, PhysicalSize}, event::{DeviceEvent, DeviceId, KeyEvent, MouseButton, WindowEvent}, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId}};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;



#[cfg(target_arch = "wasm32")]
pub mod canvas {
    use wasm_bindgen::UnwrapThrowExt;
    use wasm_bindgen::JsCast;
    
    const CANVAS_ID: &str = "canvas";

    pub fn get_canvas() -> web_sys::HtmlCanvasElement {
        let window = web_sys::window().expect_throw("No window!");
        let document = window.document().expect_throw("No document!");
        let canvas = document.get_element_by_id(CANVAS_ID).expect_throw("No canvas!");
        canvas.unchecked_into()
    }
}



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: Vec3<f32>,
    color: Vec3<f32>,
}

impl Vertex {
    const ATTRIBUTES: &[wgpu::VertexAttribute] = &[
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x3,
        },
    ];
    
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: Self::ATTRIBUTES,
        }
    }
}

type Index = u32;


const VERTICES: &[Vertex] = &[
    Vertex { position: Vector([-0.1,  0.5,  0.0]), color: Vector([1.0, 0.0, 0.0]) },
    Vertex { position: Vector([-0.5,  0.0,  0.0]), color: Vector([0.0, 1.0, 0.0]) },
    Vertex { position: Vector([-0.3, -0.5,  0.0]), color: Vector([0.0, 0.0, 1.0]) },
    Vertex { position: Vector([ 0.4, -0.4,  0.0]), color: Vector([1.0, 0.5, 0.0]) },
    Vertex { position: Vector([ 0.5,  0.2,  0.0]), color: Vector([0.5, 0.0, 1.0]) },
];

const INDICES: &[Index] = &[0, 1, 2, 0, 2, 3, 0, 3, 4];



pub struct TrackedKeys {
    pub w: bool,
    pub s: bool,
    pub a: bool,
    pub d: bool,
    pub space: bool,
    pub shift: bool,
}


pub struct Camera {
    pub position: Vec3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub fov: f32,
    pub aspect_ratio: f32,
    pub near_clip: f32,
    pub far_clip: f32,
}

impl Camera {
    pub fn get_transform(&self) -> Mat4x4<f32> {
        let width_scale = 1.0 / f32::tan(self.fov * std::f32::consts::PI / 180.0 * 0.5);
        scale_axes([-width_scale, width_scale * self.aspect_ratio, 1.0 / (self.far_clip - self.near_clip), 1.0]) * Vector([
            Vector([1.0, 0.0, 0.0, 0.0]),
            Vector([0.0, 1.0, 0.0, 0.0]),
            Vector([0.0, 0.0, 1.0, 1.0]),
            Vector([0.0, 0.0, 0.0, 0.0]),
        ]) * rotate_axes([0, 1], self.roll) * rotate_axes([1, 2], self.pitch) * rotate_axes([0, 2], self.yaw) * translate_3d(-self.position)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexUniforms {
    pub camera_transform: Mat4x4<f32>,
    pub model_transform: Mat4x4<f32>,
}



pub struct WindowState {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub limits: wgpu::Limits,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub is_surface_configured: bool,
    pub render_pipeline: wgpu::RenderPipeline,
    
    pub font_system: glyphon::FontSystem,
    pub swash_cache: glyphon::SwashCache,
    pub text_viewport: glyphon::Viewport,
    pub atlas: glyphon::TextAtlas,
    pub text_renderer: glyphon::TextRenderer,
    pub fps_text_buffer: glyphon::Buffer,
    pub controls_text_buffer: glyphon::Buffer,
    
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    
    pub average_frame_dt: f64,
    
    #[cfg(not(target_arch = "wasm32"))]
    pub previous_frame_time: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    pub previous_frame_time: f64,
    
    pub mouse_position: PhysicalPosition<f64>,
    pub camera: Camera,
    pub cursor_grab: bool,
    pub speed: f64,
    pub sensitivity: f64,
    pub keys: TrackedKeys,
}

impl WindowState {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))] backends: wgpu::Backends::VULKAN,
            #[cfg(    target_arch = "wasm32" )] backends: wgpu::Backends::GL,
            ..Default::default()
        });
        
        let surface = instance.create_surface(window.clone()).unwrap();
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await?;
        
        let limits = adapter.limits();
        
        #[cfg(not(target_arch = "wasm32"))]
        let required_limits = wgpu::Limits::default();
        #[cfg(target_arch = "wasm32")]
        let required_limits = wgpu::Limits::downlevel_webgl2_defaults();
        
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
        }).await?;
        
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,//surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 0,
        };
        
        
        let image_bytes = include_bytes!("../assets/test.png");
        let image = image::load_from_memory(image_bytes).unwrap().to_rgba8();
        let (image_width, image_height) = image.dimensions();
        
        let texture_size = wgpu::Extent3d {
            width: image_width,
            height: image_height,
            depth_or_array_layers: 1,
        };
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("texture"),
            view_formats: &[],
        });
        
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &image,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * image_width),
                rows_per_image: Some(image_height),
            },
            texture_size,
        );
        
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        
        
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: &[0u8; std::mem::size_of::<VertexUniforms>()],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        
        
        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex buffer"),
            contents: bytemuck::cast_slice(teapot::VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index buffer"),
            contents: bytemuck::cast_slice(teapot::INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into())
        });
        
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render pipeline layout"),
            bind_group_layouts: &[
                &uniform_bind_group_layout,
                &bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("test_constant", 0.9)],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("test_constant", 0.9)],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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
            cache: None,
        });
        
        
        let mut font_system = glyphon::FontSystem::new_with_locale_and_db(String::from("en-US"), glyphon::fontdb::Database::new());
        // font_system.db_mut().load_fonts_dir("assets/fonts");
        font_system.db_mut().load_font_data(include_bytes!("../assets/fonts/Luciole-Regular.ttf").to_vec());
        
        let swash_cache = glyphon::SwashCache::new();
        let cache = glyphon::Cache::new(&device);
        let text_viewport = glyphon::Viewport::new(&device, &cache);
        let mut atlas = glyphon::TextAtlas::new(&device, &queue, &cache, surface_format);
        let text_renderer = glyphon::TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);
        
        let mut fps_text_buffer = glyphon::Buffer::new(&mut font_system, glyphon::Metrics { font_size: 24.0, line_height: 24.0 });
        fps_text_buffer.set_size(&mut font_system, Some(300.0), Some(100.0));
        // fps_text_buffer.set_text(&mut font_system, "Text text!", &glyphon::Attrs::new().color(glyphon::Color::rgb(255, 255, 255)).family(glyphon::Family::Name("Luciole")), glyphon::Shaping::Basic);
        fps_text_buffer.shape_until_scroll(&mut font_system, false);
        
        let mut controls_text_buffer = glyphon::Buffer::new(&mut font_system, glyphon::Metrics { font_size: 24.0, line_height: 24.0 });
        controls_text_buffer.set_size(&mut font_system, Some(500.0), Some(100.0));
        controls_text_buffer.set_text(&mut font_system, "Controls: Mouse + WASD", &glyphon::Attrs::new().color(glyphon::Color::rgb(255, 255, 255)).family(glyphon::Family::Name("Luciole")), glyphon::Shaping::Basic);
        controls_text_buffer.shape_until_scroll(&mut font_system, false);
        
        
        Ok(Self {
            window,
            surface,
            limits,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            
            font_system,
            swash_cache,
            text_viewport,
            atlas,
            text_renderer,
            fps_text_buffer,
            controls_text_buffer,
            
            vertex_buffer,
            index_buffer,
            bind_group,
            uniform_buffer,
            uniform_bind_group,
            
            average_frame_dt: 0.0,
            #[cfg(not(target_arch = "wasm32"))]
            previous_frame_time: std::time::Instant::now(),
            #[cfg(target_arch = "wasm32")]
            previous_frame_time: 0.0,
            
            mouse_position: PhysicalPosition { x: 0.0, y: 0.0 },
            camera: Camera {
                position: Vector([0.0, 0.0, -2.0]),
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                fov: 90.0,
                aspect_ratio: size.width as f32 / size.height as f32,
                near_clip: 0.0,
                far_clip: 10.0,
            },
            cursor_grab: false,
            speed: 1.0,
            sensitivity: 0.005,
            keys: TrackedKeys { w: false, s: false, a: false, d: false, space: false, shift: false }
        })
    }
    
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return }
        self.config.width = new_size.width.min(self.limits.max_texture_dimension_2d);
        self.config.height = new_size.height.min(self.limits.max_texture_dimension_2d);
        // Make sure canvas width and height are set in CSS or this call will take control and crash the app in a very silly way!
        self.surface.configure(&self.device, &self.config);
        self.is_surface_configured = true;
        
        self.text_viewport.update(&self.queue, glyphon::Resolution { width: self.config.width, height: self.config.height });
        self.camera.aspect_ratio = self.config.width as f32 / self.config.height as f32;
    }
    
    pub fn update(&mut self) {
        todo!()
    }
    
    pub fn render(&mut self) -> std::result::Result<(), wgpu::SurfaceError> {
        if !self.is_surface_configured { return Ok(()) }
        
        #[cfg(not(target_arch = "wasm32"))]
        let dt = {
            let now = std::time::Instant::now();
            let dt = now.duration_since(self.previous_frame_time).as_secs_f64();
            self.previous_frame_time = now;
            dt
        };
        
        #[cfg(target_arch = "wasm32")]
        let dt = {
            let now = web_sys::window().unwrap().performance().unwrap().now() * 0.001;
            let dt = now - self.previous_frame_time;
            self.previous_frame_time = now;
            dt
        };
        
        
        let contribution_to_average = dt.clamp(0.07, 1.0);
        self.average_frame_dt = (1.0 - contribution_to_average) * self.average_frame_dt + contribution_to_average * dt;
        
        
        use num_traits::ConstZero;
        let mut movement = Vector::<f32, 3>::ZERO;
        
        if self.keys.w { movement += Vector([0.0, 0.0, 1.0]); }
        if self.keys.s { movement -= Vector([0.0, 0.0, 1.0]); }
        if self.keys.a { movement += Vector([1.0, 0.0, 0.0]); }
        if self.keys.d { movement -= Vector([1.0, 0.0, 0.0]); }
        if self.keys.space { movement += Vector([0.0, 1.0, 0.0]); }
        if self.keys.shift { movement -= Vector([0.0, 1.0, 0.0]); }
        
        self.camera.position += movement.transform(rotate_axes([0, 2], -self.camera.yaw)).scale((self.speed * dt) as f32);
        
        
        
        
        
        self.camera.yaw = ((self.mouse_position.x - self.config.width as f64 * 0.5) * self.sensitivity) as f32;
        self.camera.pitch = ((self.mouse_position.y - self.config.height as f64 * 0.5) * self.sensitivity) as f32;
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[VertexUniforms {
            camera_transform: self.camera.get_transform(),
            model_transform: scale_axes([1.0, 1.0, -1.0, 1.0]),
        }]));
        
        self.fps_text_buffer.set_text(&mut self.font_system, &format!("Fps: {:.1}", 1.0 / self.average_frame_dt), &glyphon::Attrs::new().color(glyphon::Color::rgb(255, 255, 255)).family(glyphon::Family::Name("Luciole")), glyphon::Shaping::Basic);
        
        
        self.text_renderer.prepare(&self.device, &self.queue, &mut self.font_system, &mut self.atlas, &self.text_viewport, [
            glyphon::TextArea {
                buffer: &self.fps_text_buffer,
                left: 10.0 * self.window.scale_factor() as f32,
                top: 10.0 * self.window.scale_factor() as f32,
                scale: self.window.scale_factor() as f32,
                bounds: glyphon::TextBounds {
                    left: (10.0 * self.window.scale_factor()) as i32,
                    top: (10.0 * self.window.scale_factor()) as i32,
                    right: (self.config.width as f64 - 10.0 * self.window.scale_factor()) as i32,
                    bottom: (self.config.height as f64 - 10.0 * self.window.scale_factor()) as i32,
                },
                default_color: glyphon::Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            },
            glyphon::TextArea {
                buffer: &self.controls_text_buffer,
                left: 10.0 * self.window.scale_factor() as f32,
                top: self.config.height as f32 - 34.0 * self.window.scale_factor() as f32,
                scale: self.window.scale_factor() as f32,
                bounds: glyphon::TextBounds {
                    left: (10.0 * self.window.scale_factor()) as i32,
                    top: (self.config.height as f64 - 50.0 * self.window.scale_factor()) as i32,
                    right: (self.config.width as f64 - 10.0 * self.window.scale_factor()) as i32,
                    bottom: (self.config.height as f64 - 10.0 * self.window.scale_factor()) as i32,
                },
                default_color: glyphon::Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            }
        ], &mut self.swash_cache).unwrap();
        
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_bind_group(1, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..(teapot::INDICES.len() as u32 * 3), 0, 0..1);
        
        self.text_renderer.render(&self.atlas, &self.text_viewport, &mut render_pass).unwrap();
        
        drop(render_pass);
        
        self.queue.submit(std::iter::once(encoder.finish()));
        self.window.pre_present_notify();
        output.present();
        
        self.atlas.trim();
        
        let _ = self.device.poll(wgpu::PollType::Wait);
        
        Ok(())
    }
}


pub struct App {
    state: Option<WindowState>,
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<WindowState>>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<WindowState>) -> Self {
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")] proxy: Some(event_loop.create_proxy()),
        }
    }
}

impl ApplicationHandler<WindowState> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();
        
        #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::WindowAttributesExtWebSys;
            window_attributes = window_attributes.with_canvas(Some(canvas::get_canvas()));
        }
        
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        
        #[cfg(not(target_arch = "wasm32"))] {
            self.state = Some(pollster::block_on(WindowState::new(window)).unwrap());
        }
        
        #[cfg(target_arch = "wasm32")]
        if let Some(proxy) = self.proxy.take() {
            wasm_bindgen_futures::spawn_local(async move {
                assert!(proxy.send_event(WindowState::new(window).await.expect("Unable to set up canvas.")).is_ok())
            })
        }
        
        event_loop.set_control_flow(ControlFlow::Poll);
    }
    
    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: WindowState) {
        #[cfg(target_arch = "wasm32")] {
            event.window.request_redraw();
            event.resize(event.window.inner_size());
        }
        
        self.state = Some(event);
    }
    
    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        let state = match &mut self.state { Some(state) => state, None => return };
        
        match event {
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                if state.cursor_grab {
                    state.mouse_position.x += dx;
                    state.mouse_position.y += dy;
                }
            }
            _ => ()
        }
    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let state = match &mut self.state { Some(state) => state, None => return };
        
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            
            WindowEvent::RedrawRequested => {
                state.window.request_redraw();
                
                match state.render() {
                    Ok(_) => (),
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.resize(state.window.inner_size());
                    }
                    Err(e) => log::error!("Render broke uh oh: {e}")
                }
            }
            
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state: s, .. }, ..
            } => match (code, s.is_pressed()) {
                (KeyCode::Escape, true) => (),
                (KeyCode::KeyW, pressed) => state.keys.w = pressed,
                (KeyCode::KeyS, pressed) => state.keys.s = pressed,
                (KeyCode::KeyA, pressed) => state.keys.a = pressed,
                (KeyCode::KeyD, pressed) => state.keys.d = pressed,
                (KeyCode::Space, pressed) => state.keys.space = pressed,
                (KeyCode::ShiftLeft, pressed) => state.keys.shift = pressed,
                (KeyCode::Digit0, true) => { state.config.desired_maximum_frame_latency = 0; state.surface.configure(&state.device, &state.config); }
                (KeyCode::Digit1, true) => { state.config.desired_maximum_frame_latency = 1; state.surface.configure(&state.device, &state.config); }
                (KeyCode::Digit2, true) => { state.config.desired_maximum_frame_latency = 2; state.surface.configure(&state.device, &state.config); }
                (KeyCode::Digit3, true) => { state.config.desired_maximum_frame_latency = 3; state.surface.configure(&state.device, &state.config); }
                (KeyCode::Digit4, true) => { state.config.desired_maximum_frame_latency = 4; state.surface.configure(&state.device, &state.config); }
                _ => ()
            }
            
            WindowEvent::MouseInput { button, state: s, device_id: _ } => {
                match (button, s.is_pressed()) {
                    (MouseButton::Left, true) => {
                        // if !state.cursor_grab {
                        //     state.cursor_grab = true;
                        //     // state.window.set_cursor_grab(winit::window::CursorGrabMode::Locked).unwrap();
                        //     // state.window.set_cursor_grab(winit::window::CursorGrabMode::Confined).unwrap();
                        //     // state.window.set_cursor_visible(false);
                        // }
                    }
                    
                    _ => ()
                }
            }
            
            WindowEvent::CursorMoved { position, device_id: _ } => {
                state.mouse_position = position;
            }
            
            _ => ()
        }
    }
}


#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> Result<()> {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Error).init();
    println!("desktop app started");
    
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new();
    
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run() -> std::result::Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap_throw();
    log::info!("wasm app started");
    
    let event_loop = EventLoop::with_user_event().build().unwrap_throw();
    let app = App::new(&event_loop);
    
    // run_app works on wasm, but winit does something goofy with exceptions in it to keep the same return signature.
    // spawn_app does basically the same thing, but without this silliness, so the JS caller returns gracefully.
    use winit::platform::web::EventLoopExtWebSys;
    event_loop.spawn_app(app);
    Ok(())
}



