use winit::dpi::LogicalPosition;

use crate::*;

// Context-free UI layout structures

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, bytemuck::Zeroable)]
pub struct Color(pub Vector<f32, 4>);

unsafe impl bytemuck::Pod for Color {}

impl Color {
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self(Vector([r, g, b, a]))
    }
    
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self(Vector([r, g, b, 1.0]))
    }
    
    pub const fn gray_alpha(f: f32, a: f32) -> Self {
        Self(Vector([f, f, f, a]))
    }
    
    pub const fn gray(f: f32) -> Self {
        Self(Vector([f, f, f, 1.0]))
    }
}

impl<V: Into<Vector<f32, 4>>> From<V> for Color {
    fn from(value: V) -> Self {
        Self(value.into())
    }
}

impl From<Color> for glyphon::Color {
    fn from(value: Color) -> Self {
        glyphon::Color::rgba(
            (value.0[0] * 255.0) as u8,
            (value.0[1] * 255.0) as u8,
            (value.0[2] * 255.0) as u8,
            (value.0[3] * 255.0) as u8,
        )
    }
}



const RECT_VERTICES: &[Vector<f32, 2>] = &[
    Vector([0.0, 0.0]),
    Vector([1.0, 0.0]),
    Vector([0.0, 1.0]),
    Vector([1.0, 1.0]),
];

/// Defines a rectangle by the coordinates of its edges.
/// Intended to be sent to GPU for UI rendering.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Rect {
    pub position: Vector<f32, 2>,
    pub size: Vector<f32, 2>,
}

impl Rect {
    pub fn offset(&self, amount: Vector<f32, 2>) -> Self {
        Self { position: self.position + amount, size: self.size }
    }
    pub fn inset(&self, margin: f32) -> Self {
        Self { position: self.position + Vector([margin, margin]), size: self.size - Vector([margin * 2.0, margin * 2.0]) }
    }
    pub fn left(&self) -> f32 { self.position[0] }
    pub fn right(&self) -> f32 { self.position[0] + self.size[0] }
    pub fn top(&self) -> f32 { self.position[1] }
    pub fn bottom(&self) -> f32 { self.position[1] + self.size[1] }
    pub fn width(&self) -> f32 { self.size[0] }
    pub fn height(&self) -> f32 { self.size[1] }
    
    pub fn contains_point(&self, point: Vector<f32, 2>) -> bool {
        point[0] >= self.left() && point[0] < self.right() && point[1] >= self.top() && point[1] < self.bottom()
    }
}


#[derive(Debug, Copy, Clone)]
pub struct BoxArea {
    pub rect: Rect,
    pub color: Color,
    pub id: Option<MenuID>,
}

impl BoxArea {
    pub fn new(position: impl Into<Vector<f32, 2>>, size: impl Into<Vector<f32, 2>>, color: impl Into<Color>, id: Option<MenuID>) -> Self {
        Self {
            rect: Rect {
                position: position.into(),
                size: size.into(),
            },
            color: color.into(),
            id,
        }
    }
    
    pub fn new_centered(center: impl Into<Vector<f32, 2>>, size: impl Into<Vector<f32, 2>>, color: impl Into<Color>, id: Option<MenuID>) -> Self {
        let size = size.into();
        Self {
            rect: Rect {
                position: center.into() - size.scale(0.5),
                size,
            },
            color: color.into(),
            id,
        }
    }
}


#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoxAreaInstance {
    pub rect: Rect,
    pub color: Color,
}



// #[derive(Default)]
// pub struct TextProperties {
//     pub color: Option<Color>,
//     pub font: String,
//     pub stretch: glyphon::Stretch,
//     pub style: glyphon::Style,
//     pub weight: glyphon::Weight,
//     pub letter_spacing: Option<glyphon::cosmic_text::LetterSpacing>,
//     pub font_features: glyphon::cosmic_text::FontFeatures,
//     pub align: Option<glyphon::cosmic_text::Align>,
// }

// impl TextProperties {
//     pub fn attrs<'a>(&'a self) -> glyphon::Attrs<'a> {
//         glyphon::Attrs {
//             color_opt: self.color.map(|c| c.into()),
//             family: glyphon::Family::Name(&self.font),
//             stretch: self.stretch,
//             style: self.style,
//             weight: self.weight,
//             letter_spacing_opt: self.letter_spacing,
//             font_features: self.font_features.clone(),
//             metadata: 0,
//             cache_key_flags: glyphon::cosmic_text::CacheKeyFlags::empty(),
//             metrics_opt: None,
//         }
//     }
// }

// pub struct TextArea {
//     pub rect: Rect,
//     pub buffer: glyphon::Buffer,
//     pub properties: TextProperties,
// }

// impl TextArea {
//     pub fn new(font_system: &mut glyphon::FontSystem rect: Rect, font_size: f32, line_height: f32, properties: TextProperties) -> Self {
//         let mut buffer = glyphon::Buffer::new(glyphon::Metrics { font_size, line_height });
//         buffer.set_wrap()
        
//         Self {
//             rect,
//             buffer: ,
//             properties,
//         }
//     }
// }



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MenuUniforms {
    pub scale_factor: Vector<f32, 2>,
}


pub struct MenuSystem {
    pub font_system: glyphon::FontSystem,
    pub swash_cache: glyphon::SwashCache,
    pub text_viewport: glyphon::Viewport,
    pub atlas: glyphon::TextAtlas,
    pub text_renderer: glyphon::TextRenderer,
    
    pub rect_vertex_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub render_pipeline_layout: wgpu::PipelineLayout,
    pub render_pipeline: wgpu::RenderPipeline,
    
    pub uniform_buffer: wgpu::Buffer,
    pub box_area_cache: Vec<BoxArea>,
    pub box_area_buffer: Option<wgpu::Buffer>,
    pub layout_needs_update: bool,
    
    pub default_text_color: Color,
}

impl MenuSystem {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat, default_text_color: impl Into<Color>) -> Self {
        
        let mut font_system = glyphon::FontSystem::new_with_locale_and_db(String::from("en-US"), glyphon::fontdb::Database::new());
        // font_system.db_mut().load_fonts_dir("assets/fonts");
        font_system.db_mut().load_font_data(include_bytes!("../assets/fonts/Luciole-Regular.ttf").to_vec());
        
        let swash_cache = glyphon::SwashCache::new();
        let cache = glyphon::Cache::new(device);
        let text_viewport = glyphon::Viewport::new(device, &cache);
        let mut atlas = glyphon::TextAtlas::new(device, queue, &cache, surface_format);
        let text_renderer = glyphon::TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);
        
        use wgpu::util::DeviceExt;
        let rect_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rect vertex buffer"),
            contents: bytemuck::cast_slice(RECT_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Menu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("menu_shader.wgsl").into())
        });
        
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Menu uniforms"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<MenuUniforms>() as wgpu::BufferAddress,
            mapped_at_creation: false,
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Menu bind group layout"),
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
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Menu bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Menu render pipeline layout"),
            bind_group_layouts: &[
                &bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Menu render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vector<f32, 2>>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<BoxAreaInstance>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: std::mem::size_of::<Vector<f32, 2>>() as wgpu::BufferAddress,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: std::mem::size_of::<Rect>() as wgpu::BufferAddress,
                                shader_location: 3,
                            },
                        ],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
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
            cache: None,
        });
        
        
        Self {
            font_system,
            swash_cache,
            text_viewport,
            atlas,
            text_renderer,
            rect_vertex_buffer,
            box_area_cache: vec![],
            box_area_buffer: None,
            layout_needs_update: true,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            render_pipeline_layout,
            render_pipeline,
            default_text_color: default_text_color.into(),
        }
    }
    
    pub fn render(&self, render_pass: &mut wgpu::RenderPass) -> Result<()> {
        let instance_buffer = match &self.box_area_buffer { Some(b) => b, None => return Ok(()) };
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.rect_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..RECT_VERTICES.len() as u32, 0..self.box_area_cache.len() as u32);
        
        self.text_renderer.render(&self.atlas, &self.text_viewport, render_pass)?;
        
        Ok(())
    }
    
    pub fn find_box_at(&self, position: LogicalPosition<f32>) -> Option<MenuID> {
        for box_area in self.box_area_cache.iter().rev() {
            if box_area.rect.contains_point(Vector([position.x, position.y])) {
                if let Some(id) = box_area.id {
                    return Some(id)
                }
            }
        }
        return None
    }
}

        
        
        // 1: declare elements that exist (struct)
        // 2: define layout function to generate rect list (impl trait)
        // 3: define function to iterate over text buffer &s w/ location info ()
        
        // text area: wrapper over text buffer with all info needed for drawing
        
        
        // cache rects in buffer for rendering
        // remake buffer if layout changes length
        // write to buffer after layout
        
        // system prepare: prepare text buffer list
        
        
        // size update: recalculate all of ui
        // text update: re-render updated text
        
        
        // let num_backends = wgpu::Instance::enabled_backend_features().iter().count();
        // for (i, backend) in wgpu::Instance::enabled_backend_features().iter().enumerate() {
        //     let s = match backend {
        //         wgpu::Backends::NOOP            => "No-op",
        //         wgpu::Backends::VULKAN          => "Vulkan",
        //         wgpu::Backends::METAL           => "Metal",
        //         wgpu::Backends::DX12            => "DirectX 12",
        //         wgpu::Backends::GL              => "GL",
        //         wgpu::Backends::BROWSER_WEBGPU  => "WebGPU",
        //         _ => continue
        //     };
        //     let (box_area, text) = menu_system.new_text_box(
        //         true,
        //         Color::rgba(0.0, 0.0, 0.0, 0.3),
        //         glyphon::Metrics { font_size: 12.0, line_height: 16.0 },
        //         TextProperties { font: "Luciole".to_string(), align: Some(glyphon::cosmic_text::Align::Center), ..Default::default() },
        //         LogicalInsets { left: 10.0, right: 10.0, top: 10.0, bottom: 10.0 },
        //         move |size| Rect { position: Vector([i as f32 / num_backends as f32 + 5.0, 5.0]), size: Vector([size.x() / num_backends as f32 - 10.0, size.y() - 10.0]) },
        //         [],
        //     );
        //     menu_system.set_text(&mut text.borrow_mut(), s);
        //     menu.backends.borrow().children[1].borrow_mut().children.push(box_area);
        // }

