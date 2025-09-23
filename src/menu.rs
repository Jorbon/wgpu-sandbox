use std::{cell::RefCell, rc::Rc};

use winit::dpi::LogicalInsets;

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


#[derive(Default)]
pub struct TextProperties {
    pub color: Option<Color>,
    pub font: String,
    pub stretch: glyphon::Stretch,
    pub style: glyphon::Style,
    pub weight: glyphon::Weight,
    pub letter_spacing: Option<glyphon::cosmic_text::LetterSpacing>,
    pub font_features: glyphon::cosmic_text::FontFeatures,
}

impl TextProperties {
    pub fn attrs<'a>(&'a self) -> glyphon::Attrs<'a> {
        glyphon::Attrs {
            color_opt: self.color.map(|c| c.into()),
            family: glyphon::Family::Name(&self.font),
            stretch: self.stretch,
            style: self.style,
            weight: self.weight,
            letter_spacing_opt: self.letter_spacing,
            font_features: self.font_features.clone(),
            metadata: 0,
            cache_key_flags: glyphon::cosmic_text::CacheKeyFlags::empty(),
            metrics_opt: None,
        }
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
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoxAreaInstance {
    pub rect: Rect,
    pub color: Color,
}



pub struct TextArea {
    pub buffer: glyphon::Buffer,
    pub properties: TextProperties,
    pub margins: LogicalInsets<f32>,
}


pub struct BoxArea {
    pub show: bool,
    pub color: Color,
    pub position_rule: Box<dyn Fn(Vector<f32, 2>) -> Rect>,
    pub text: Option<Rc<RefCell<TextArea>>>,
    pub rect_cache: Option<Rect>,
    pub children: Vec<Rc<RefCell<BoxArea>>>,
}

impl BoxArea {
    pub fn recursive_filter_map_collect<T, F: Fn(&Self) -> Option<T> + 'static>(&self, boxes: &mut Vec<T>, f: &F) {
        if let Some(t) = f(self) { boxes.push(t); }
        for child in &self.children {
            child.borrow().recursive_filter_map_collect(boxes, f);
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MenuUniforms {
    scale_factor: Vector<f32, 2>,
}


pub struct MenuSystem {
    pub font_system: glyphon::FontSystem,
    pub swash_cache: glyphon::SwashCache,
    pub text_viewport: glyphon::Viewport,
    pub atlas: glyphon::TextAtlas,
    pub text_renderer: glyphon::TextRenderer,
    
    pub rect_vertex_buffer: wgpu::Buffer,
    pub instance_buffer: Option<wgpu::Buffer>,
    pub instance_count: usize,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub render_pipeline_layout: wgpu::PipelineLayout,
    pub render_pipeline: wgpu::RenderPipeline,
    
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
            instance_buffer: None,
            instance_count: 0,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            render_pipeline_layout,
            render_pipeline,
            default_text_color: default_text_color.into(),
        }
    }
    
    pub fn new_box(
        &mut self,
        show: bool,
        color: impl Into<Color>,
        position_rule: impl Fn(Vector<f32, 2>) -> Rect + 'static,
        children: impl IntoIterator<Item = Rc<RefCell<BoxArea>>>,
    ) -> Rc<RefCell<BoxArea>> {
        Rc::new(RefCell::new(BoxArea {
            show,
            color: color.into(),
            position_rule: Box::new(position_rule),
            text: None,
            rect_cache: None,
            children: children.into_iter().collect(),
        }))
    }
    
    pub fn new_text_box(
        &mut self,
        show: bool,
        color: impl Into<Color>,
        text_metrics: glyphon::Metrics,
        text_properties: TextProperties,
        text_margins: LogicalInsets<f32>,
        position_rule: impl Fn(Vector<f32, 2>) -> Rect + 'static,
        children: impl IntoIterator<Item = Rc<RefCell<BoxArea>>>,
    ) -> (Rc<RefCell<BoxArea>>, Rc<RefCell<TextArea>>) {
        let text = Rc::new(RefCell::new(TextArea {
            buffer: glyphon::Buffer::new(&mut self.font_system, text_metrics),
            properties: text_properties,
            margins: text_margins,
        }));
        
        (Rc::new(RefCell::new(BoxArea {
            show,
            color: color.into(),
            position_rule: Box::new(position_rule),
            text: Some(Rc::clone(&text)),
            rect_cache: None,
            children: children.into_iter().collect(),
        })), text)
    }
    
    pub fn update_sizes(&mut self, box_area: &mut BoxArea, parent_rect: Rect) {
        let rect = (box_area.position_rule)(parent_rect.size).offset(parent_rect.position);
        box_area.rect_cache = Some(rect);
        if let Some(text) = &mut box_area.text {
            let margins = text.borrow().margins;
            let buffer_width = rect.size.x() - (margins.left + margins.right);
            let buffer_height = rect.size.y() - (margins.top + margins.bottom);
            text.borrow_mut().buffer.set_size(&mut self.font_system, Some(buffer_width), Some(buffer_height));
        }
        for child in &mut box_area.children {
            self.update_sizes(&mut child.borrow_mut(), rect);
        }
    }
    
    pub fn prepare(&mut self, root: Rc<RefCell<BoxArea>>, window: &Arc<Window>, config: &wgpu::SurfaceConfiguration, device: &wgpu::Device, queue: &wgpu::Queue) {
        let resolution = self.text_viewport.resolution();
        let scale_factor = window.scale_factor() as f32;
        
        if config.width != resolution.width || config.height != resolution.height {
            self.text_viewport.update(&queue, glyphon::Resolution { width: config.width, height: config.height });
            
            let logical_size = Vector([config.width as f32 / scale_factor, config.height as f32 / scale_factor]);
            
            self.update_sizes(&mut root.borrow_mut(), Rect { position: Vector([0.0, 0.0]), size: logical_size });
            
            let mut instances = vec![];
            root.borrow().recursive_filter_map_collect(&mut instances, &|b| Some(BoxAreaInstance { rect: b.rect_cache?, color: b.color }));
            
            if instances.len() != self.instance_count || self.instance_buffer.is_none() {
                self.instance_count = instances.len();
                self.instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Menu box instance buffer"),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    size: (self.instance_count * std::mem::size_of::<BoxAreaInstance>()) as wgpu::BufferAddress,
                    mapped_at_creation: false,
                }));
            }
            
            queue.write_buffer(self.instance_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&instances));
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[MenuUniforms {
                scale_factor: Vector([1.0 / logical_size.x(), 1.0 / logical_size.y()]),
            }]));
        }
        
        
        let mut instances = vec![];
        root.borrow().recursive_filter_map_collect(&mut instances, &|b| {
            Some((Rc::clone(b.text.as_ref()?), b.rect_cache?))
        });
        
        let instances = instances.iter().map(|(text, rect)| (text.borrow(), rect)).collect::<Vec<_>>();
        
        self.text_renderer.prepare(&device, &queue, &mut self.font_system, &mut self.atlas, &self.text_viewport, (0..instances.len()).map(|i| {
            let (text, rect) = &instances[i];
            
            glyphon::TextArea {
                buffer: &text.buffer,
                left: (rect.position.x() + text.margins.left) * scale_factor,
                top: (rect.position.y() + text.margins.top) * scale_factor,
                scale: scale_factor,
                bounds: glyphon::TextBounds {
                    left: ((rect.position.x() + text.margins.left) * scale_factor) as i32,
                    top: ((rect.position.y() + text.margins.top) * scale_factor) as i32,
                    right: ((rect.position.x() + rect.size.x() - text.margins.right) * scale_factor) as i32,
                    bottom: ((rect.position.y() + rect.size.y() - text.margins.bottom) * scale_factor) as i32,
                },
                default_color: self.default_text_color.into(),
                custom_glyphs: &[],
            }
        }), &mut self.swash_cache).unwrap();
    }
    
    pub fn render(&self, render_pass: &mut wgpu::RenderPass) -> Result<()> {
        let instance_buffer = match &self.instance_buffer { Some(b) => b, None => return Ok(()) };
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.rect_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..RECT_VERTICES.len() as u32, 0..self.instance_count as u32);
        
        self.text_renderer.render(&self.atlas, &self.text_viewport, render_pass)?;
        
        Ok(())
    }
}




pub struct Menu {
    pub root: Rc<RefCell<BoxArea>>,
    pub fps_text: Rc<RefCell<TextArea>>,
}

impl Menu {
    pub fn new(system: &mut MenuSystem) -> Self {
        
        let (fps_box, fps_text) = system.new_text_box(
            true,
            Color::rgba(0.0, 0.0, 0.0, 0.3),
            glyphon::Metrics { font_size: 24.0, line_height: 24.0 },
            TextProperties::default(),
            LogicalInsets { left: 10.0, right: 10.0, top: 10.0, bottom: 10.0 },
            |_size| Rect { position: Vector([0.0, 0.0]), size: Vector([300.0, 50.0]) },
            [],
        );
        
        let (controls_box, controls_text) = system.new_text_box(
            true,
            Color::rgba(0.0, 0.0, 0.0, 0.3),
            glyphon::Metrics { font_size: 24.0, line_height: 24.0 },
            TextProperties::default(),
            LogicalInsets { left: 10.0, right: 10.0, top: 10.0, bottom: 10.0 },
            |size| Rect { position: Vector([0.0, size.y() - 50.0]), size: Vector([500.0, 50.0]) },
            [],
        );
        
        controls_text.borrow_mut().buffer.set_text(&mut system.font_system, "Controls: WASD + Mouse", &glyphon::Attrs::new().color(glyphon::Color::rgb(255, 255, 255)).family(glyphon::Family::Name("Luciole")), glyphon::Shaping::Basic);
        
        let root = system.new_box(
            true,
            Color::rgba(0.0, 0.0, 0.0, 0.0),
            |size| Rect { position: Vector([0.0, 0.0]), size },
            [fps_box, controls_box],
        );
        
        Self {
            root,
            fps_text,
        }
        
    }
}


