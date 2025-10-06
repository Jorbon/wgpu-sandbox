use crate::*;



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: Vec3<f32>,
    pub color: Vec3<f32>,
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
    
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: Self::ATTRIBUTES,
        }
    }
}

pub type Index = u16;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub model_transform: Mat4x4<f32>,
}

impl Instance {
    const ATTRIBUTES: &[wgpu::VertexAttribute] = &[
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32x4
        },
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
            shader_location: 6,
            format: wgpu::VertexFormat::Float32x4
        },
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
            shader_location: 7,
            format: wgpu::VertexFormat::Float32x4
        },
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
            shader_location: 8,
            format: wgpu::VertexFormat::Float32x4
        },
    ];
    
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: Self::ATTRIBUTES,
        }
    }
}


const VERTICES: &[Vertex] = &[
    Vertex { position: Vector([-0.1,  0.5,  0.0]), color: Vector([1.0, 0.0, 0.0]) },
    Vertex { position: Vector([-0.5,  0.0,  0.0]), color: Vector([0.0, 1.0, 0.0]) },
    Vertex { position: Vector([-0.3, -0.5,  0.0]), color: Vector([0.0, 0.0, 1.0]) },
    Vertex { position: Vector([ 0.4, -0.4,  0.0]), color: Vector([1.0, 0.5, 0.0]) },
    Vertex { position: Vector([ 0.5,  0.2,  0.0]), color: Vector([0.5, 0.0, 1.0]) },
];

const INDICES: &[Index] = &[0, 1, 2, 0, 2, 3, 0, 3, 4];



#[derive(Default)]
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
        // Reversed z, from https://developer.nvidia.com/blog/visualizing-depth-precision/
        let a = self.near_clip / (self.far_clip - self.near_clip);
        scale_axes([-width_scale, width_scale * self.aspect_ratio, 1.0, 1.0]) * Vector([
            Vector([1.0, 0.0, 0.0, 0.0]),
            Vector([0.0, 1.0, 0.0, 0.0]),
            Vector([0.0, 0.0, -a, 1.0]),
            Vector([0.0, 0.0, a*self.far_clip, 0.0]),
        ]) * rotate_axes([0, 1], self.roll) * rotate_axes([1, 2], self.pitch) * rotate_axes([0, 2], self.yaw) * translate_3d(-self.position)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexUniforms {
    pub camera_transform: Mat4x4<f32>,
}



#[derive(Debug, Copy, Clone)]
pub enum MenuID {
    Backend(wgpu::Backend),
    PowerPreference(wgpu::PowerPreference),
    PresentMode(wgpu::PresentMode),
    SurfaceFormat(wgpu::TextureFormat),
    AlphaMode(wgpu::CompositeAlphaMode),
}




pub struct AppState {
    pub graphics_options: GraphicsOptions,
    pub mouse_position: PhysicalPosition<f64>,
    pub keys: TrackedKeys,
    pub camera: Camera,
    pub cursor_grab: bool,
    pub speed: f64,
    pub sensitivity: f64,
    
    pub average_frame_dt: f64,
    #[cfg(not(target_arch = "wasm32"))] pub previous_frame_time: std::time::Instant,
    #[cfg(target_arch = "wasm32")]      pub previous_frame_time: f64,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            graphics_options: GraphicsOptions::default(),
            mouse_position: PhysicalPosition { x: 0.0, y: 0.0 },
            keys: TrackedKeys::default(),
            camera: Camera {
                position: Vector([0.0, 0.0, -2.0]),
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                fov: 90.0,
                aspect_ratio: 1.0,
                near_clip: 0.001,
                far_clip: 100.0,
            },
            speed: 1.0,
            sensitivity: 0.005,
            cursor_grab: false,
            
            average_frame_dt: 0.0,
            #[cfg(not(target_arch = "wasm32"))] previous_frame_time: std::time::Instant::now(),
            #[cfg(target_arch = "wasm32")]      previous_frame_time: 0.0,
        }
    }
    
    pub fn update_graphics_options(&mut self, window_state: &WindowState) {
        self.graphics_options = GraphicsOptions {
            backend: Some(window_state.adapter.get_info().backend),
            power_preference: self.graphics_options.power_preference,
            present_mode: window_state.config.present_mode,
            surface_format: Some(window_state.config.format),
            alpha_mode: Some(window_state.config.alpha_mode),
        };
    }
    
    pub fn layout_rects(&self, window_size: Vector<f32, 2>, window_state: &WindowState) -> Vec<BoxArea> {
        let margin = 10.0;
        
        let fps = BoxArea::new([0.0, 0.0], [300.0, 50.0], Color::gray_alpha(0.0, 0.2), None);
        let controls = BoxArea::new([0.0, window_size.y() - 50.0], [500.0, 50.0], Color::gray_alpha(0.0, 0.2), None);
        let center = BoxArea::new_centered(window_size.scale(0.5), [500.0, 450.0], Color::gray_alpha(0.0, 0.2), None);
        
        let mut boxes = vec![fps, controls, center];
        
        let inner = center.rect.inset(margin);
        let h = 30.0;
        
        
        let mut pos = inner.position;
        
        boxes.push(BoxArea::new(pos, [*inner.size.x(), h], Color::gray_alpha(0.0, 0.0), None));
        pos[1] += h + margin;
        
        let backends = wgpu::Instance::enabled_backend_features().into_iter().collect::<Vec<_>>();
        let w = (inner.width() + margin) / backends.len() as f32 - margin;
        
        for backend in wgpu::Backend::ALL {
            if !backends.contains(&backend.into()) { continue }
            boxes.push(BoxArea::new(pos, [w, h], match Some(backend) == self.graphics_options.backend {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, Some(MenuID::Backend(backend))));
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
        
        boxes.push(BoxArea::new(pos, [*inner.size.x(), h], Color::gray_alpha(0.0, 0.0), None));
        pos[1] += h + margin;
        
        let power_preferences = [
            wgpu::PowerPreference::None,
            wgpu::PowerPreference::LowPower,
            wgpu::PowerPreference::HighPerformance,
        ];
        let w = (inner.width() + margin) / power_preferences.len() as f32 - margin;
        for power_preference in power_preferences {
            boxes.push(BoxArea::new(pos, [w, h], match power_preference == self.graphics_options.power_preference {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, Some(MenuID::PowerPreference(power_preference))));
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
        
        boxes.push(BoxArea::new(pos, [*inner.size.x(), h], Color::gray_alpha(0.0, 0.0), None));
        pos[1] += h + margin;
        
        boxes.push(BoxArea::new(pos, [*inner.size.x(), h], Color::gray_alpha(0.0, 0.0), None));
        pos[1] += h + margin;
        
        let w = (inner.width() + margin) / window_state.surface_caps.present_modes.len() as f32 - margin;
        for present_mode in &window_state.surface_caps.present_modes {
            boxes.push(BoxArea::new(pos, [w, h], match *present_mode == self.graphics_options.present_mode {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, Some(MenuID::PresentMode(*present_mode))));
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
        
        boxes.push(BoxArea::new(pos, [*inner.size.x(), h], Color::gray_alpha(0.0, 0.0), None));
        pos[1] += h + margin;
        
        let w = (inner.width() + margin) / window_state.surface_caps.formats.len() as f32 - margin;
        for surface_format in &window_state.surface_caps.formats {
            boxes.push(BoxArea::new(pos, [w, h], match Some(*surface_format) == self.graphics_options.surface_format {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, Some(MenuID::SurfaceFormat(*surface_format))));
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
        
        boxes.push(BoxArea::new(pos, [*inner.size.x(), h], Color::gray_alpha(0.0, 0.0), None));
        pos[1] += h + margin;
        
        let w = (inner.width() + margin) / window_state.surface_caps.alpha_modes.len() as f32 - margin;
        for alpha_mode in &window_state.surface_caps.alpha_modes {
            boxes.push(BoxArea::new(pos, [w, h], match Some(*alpha_mode) == self.graphics_options.alpha_mode {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, Some(MenuID::AlphaMode(*alpha_mode))));
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
        
        
        
        
        boxes
    }
    
    pub fn on_frame(&mut self, width: u32, height: u32) {
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
        
        // self.menu.fps_text.borrow_mut().buffer.set_text(&mut self.menu_system.font_system, &format!("Fps: {:.1}", 1.0 / self.average_frame_dt), &glyphon::Attrs::new().color(glyphon::Color::rgb(255, 255, 255)).family(glyphon::Family::Name("Luciole")), glyphon::Shaping::Basic);
        
        
        use num_traits::ConstZero;
        let mut movement = Vector::<f32, 3>::ZERO;
        
        if self.keys.w { movement += Vector([0.0, 0.0, 1.0]); }
        if self.keys.s { movement -= Vector([0.0, 0.0, 1.0]); }
        if self.keys.a { movement += Vector([1.0, 0.0, 0.0]); }
        if self.keys.d { movement -= Vector([1.0, 0.0, 0.0]); }
        if self.keys.space { movement += Vector([0.0, 1.0, 0.0]); }
        if self.keys.shift { movement -= Vector([0.0, 1.0, 0.0]); }
        
        self.camera.position += movement.transform(rotate_axes([0, 2], -self.camera.yaw)).scale((self.speed * dt) as f32);
        self.camera.yaw = ((self.mouse_position.x - width as f64 * 0.5) * self.sensitivity) as f32;
        self.camera.pitch = ((self.mouse_position.y - height as f64 * 0.5) * self.sensitivity) as f32;
        
        
    }
}



