use crate::{menu::DEFAULT_TEXT_PROPERTIES, *};

#[derive(Debug, Copy, Clone)]
pub enum WgpuConfigAreaID {
    Backend(wgpu::Backend),
    PowerPreference(wgpu::PowerPreference),
    PresentMode(wgpu::PresentMode),
    SurfaceFormat(wgpu::TextureFormat),
    AlphaMode(wgpu::CompositeAlphaMode),
}


pub struct WgpuConfigMenu {
    pub headers: [TextArea; 5],
    pub backends: Vec<TextArea>,
    pub power_preferences: Vec<TextArea>,
    pub present_modes: Vec<TextArea>,
    pub surface_formats: Vec<TextArea>,
    pub alpha_modes: Vec<TextArea>,
}

const WGPU_TEXT_PROPERTIES: TextPropertiesConst = DEFAULT_TEXT_PROPERTIES.with_align(TextAlign::Center);

impl WgpuConfigMenu {
    pub fn new(font_system: &mut glyphon::FontSystem) -> Self {
        Self {
            headers: [
                TextArea::new(font_system, WGPU_TEXT_PROPERTIES, "Backends"),
                TextArea::new(font_system, WGPU_TEXT_PROPERTIES, "Power Preference"),
                TextArea::new(font_system, WGPU_TEXT_PROPERTIES, "Present Mode"),
                TextArea::new(font_system, WGPU_TEXT_PROPERTIES, "Surface Format"),
                TextArea::new(font_system, WGPU_TEXT_PROPERTIES, "Alpha Mode"),
            ],
            backends: vec![],
            power_preferences: vec![
                TextArea::new(font_system, WGPU_TEXT_PROPERTIES, "Default"),
                TextArea::new(font_system, WGPU_TEXT_PROPERTIES, "Low Power"),
                TextArea::new(font_system, WGPU_TEXT_PROPERTIES, "High Performance"),
            ],
            present_modes: vec![],
            surface_formats: vec![],
            alpha_modes: vec![],
        }
    }
    
    pub fn iter_text_areas(&self) -> impl Iterator<Item = &TextArea> {
        self.headers.iter()
            .chain(self.backends.iter())
            .chain(self.power_preferences.iter())
            .chain(self.present_modes.iter())
            .chain(self.surface_formats.iter())
            .chain(self.alpha_modes.iter())
    }
    
    pub fn layout(
        &mut self,
        font_system: &mut glyphon::FontSystem,
        shapes: &mut Vec<ShapeArea<RootAreaID>>,
        logical_size: Vec2<f32>,
        window_state: &WindowState,
        graphics_options: &GraphicsOptions,
    ) {
        let margin = 10.0;
        
        let center_rect = Rectangle::new_centered(logical_size.scale(0.5), [500.0, 450.0]);
        let center = ShapeArea::new(Shape::Rectangle(center_rect), Color::gray_alpha(0.0, 0.2), RootAreaID::Block);
        shapes.push(center);
        
        let inner = center_rect.inset(margin);
        let mut pos = inner.position;
        let h = 30.0;
        
        let rect = Rectangle::new(pos, [*inner.size.x(), h]);
        shapes.push(ShapeArea::new(Shape::Rectangle(rect), Color::gray_alpha(0.0, 0.0), RootAreaID::Block));
        self.headers[0].set_rect(font_system, rect);
        pos[1] += h + margin;
        
        let backends = wgpu::Instance::enabled_backend_features().into_iter().collect::<Vec<_>>();
        let w = (inner.width() + margin) / backends.len() as f32 - margin;
        self.backends = vec![];
        for backend in wgpu::Backend::ALL {
            if !backends.contains(&backend.into()) { continue }
            let rect = Rectangle::new(pos, [w, h]);
            shapes.push(ShapeArea::new(Shape::Rectangle(rect), match Some(backend) == graphics_options.backend {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, RootAreaID::WgpuConfig(WgpuConfigAreaID::Backend(backend))));
            self.backends.push(TextArea::new_with_rect(font_system, WGPU_TEXT_PROPERTIES, match backend {
                wgpu::Backend::Noop             => "No-op",
                wgpu::Backend::Vulkan           => "Vulkan",
                wgpu::Backend::Metal            => "Metal",
                wgpu::Backend::Dx12             => "DirectX 12",
                wgpu::Backend::Gl               => "GL",
                wgpu::Backend::BrowserWebGpu    => "WebGPU",
            }, rect));
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
        
        let rect = Rectangle::new(pos, [*inner.size.x(), h]);
        shapes.push(ShapeArea::new(Shape::Rectangle(rect), Color::gray_alpha(0.0, 0.0), RootAreaID::Block));
        self.headers[1].set_rect(font_system, rect);
        pos[1] += h + margin;
        
        let power_preferences = [
            wgpu::PowerPreference::None,
            wgpu::PowerPreference::LowPower,
            wgpu::PowerPreference::HighPerformance,
        ];
        let w = (inner.width() + margin) / power_preferences.len() as f32 - margin;
        for (i, power_preference) in power_preferences.into_iter().enumerate() {
            let rect = Rectangle::new(pos, [w, h]);
            shapes.push(ShapeArea::new(Shape::Rectangle(rect), match power_preference == graphics_options.power_preference {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, RootAreaID::WgpuConfig(WgpuConfigAreaID::PowerPreference(power_preference))));
            self.power_preferences[i].set_rect(font_system, rect);
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
        
        let rect = Rectangle::new(pos, [*inner.size.x(), h]);
        shapes.push(ShapeArea::new(Shape::Rectangle(rect), Color::gray_alpha(0.0, 0.0), RootAreaID::Block));
        pos[1] += h + margin;
        
        let rect = Rectangle::new(pos, [*inner.size.x(), h]);
        shapes.push(ShapeArea::new(Shape::Rectangle(rect), Color::gray_alpha(0.0, 0.0), RootAreaID::Block));
        self.headers[2].set_rect(font_system, rect);
        pos[1] += h + margin;
        
        let w = (inner.width() + margin) / window_state.surface_caps.present_modes.len() as f32 - margin;
        self.present_modes = vec![];
        for present_mode in &window_state.surface_caps.present_modes {
            let rect = Rectangle::new(pos, [w, h]);
            shapes.push(ShapeArea::new(Shape::Rectangle(rect), match *present_mode == graphics_options.present_mode {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, RootAreaID::WgpuConfig(WgpuConfigAreaID::PresentMode(*present_mode))));
            self.present_modes.push(TextArea::new_with_rect(font_system, WGPU_TEXT_PROPERTIES, match *present_mode {
                wgpu::PresentMode::AutoVsync    => "Vsync On (Auto)",
                wgpu::PresentMode::AutoNoVsync  => "Vsync Off (Auto)",
                wgpu::PresentMode::Fifo         => "Fifo",
                wgpu::PresentMode::FifoRelaxed  => "Relaxed Fifo",
                wgpu::PresentMode::Immediate    => "Immediate",
                wgpu::PresentMode::Mailbox      => "Mailbox",
            }, rect));
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
        
        let rect = Rectangle::new(pos, [*inner.size.x(), h]);
        shapes.push(ShapeArea::new(Shape::Rectangle(rect), Color::gray_alpha(0.0, 0.0), RootAreaID::Block));
        self.headers[3].set_rect(font_system, rect);
        pos[1] += h + margin;
        
        let w = (inner.width() + margin) / window_state.surface_caps.formats.len() as f32 - margin;
        self.surface_formats = vec![];
        for surface_format in &window_state.surface_caps.formats {
            let rect = Rectangle::new(pos, [w, h]);
            shapes.push(ShapeArea::new(Shape::Rectangle(rect), match Some(*surface_format) == graphics_options.surface_format {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, RootAreaID::WgpuConfig(WgpuConfigAreaID::SurfaceFormat(*surface_format))));
            self.surface_formats.push(TextArea::new_with_rect(font_system, WGPU_TEXT_PROPERTIES, &format!("{:?}", *surface_format), rect));
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
        
        let rect = Rectangle::new(pos, [*inner.size.x(), h]);
        shapes.push(ShapeArea::new(Shape::Rectangle(rect), Color::gray_alpha(0.0, 0.0), RootAreaID::Block));
        self.headers[4].set_rect(font_system, rect);
        pos[1] += h + margin;
        
        let w = (inner.width() + margin) / window_state.surface_caps.alpha_modes.len() as f32 - margin;
        self.alpha_modes = vec![];
        for alpha_mode in &window_state.surface_caps.alpha_modes {
            let rect = Rectangle::new(pos, [w, h]);
            shapes.push(ShapeArea::new(Shape::Rectangle(rect), match Some(*alpha_mode) == graphics_options.alpha_mode {
                true  => Color::gray_alpha(0.0, 0.6),
                false => Color::gray_alpha(0.0, 0.3),
            }, RootAreaID::WgpuConfig(WgpuConfigAreaID::AlphaMode(*alpha_mode))));
            self.alpha_modes.push(TextArea::new_with_rect(font_system, WGPU_TEXT_PROPERTIES, match *alpha_mode {
                wgpu::CompositeAlphaMode::Auto              => "Auto",
                wgpu::CompositeAlphaMode::Opaque            => "Opaque",
                wgpu::CompositeAlphaMode::PreMultiplied     => "Pre-Multiplied",
                wgpu::CompositeAlphaMode::PostMultiplied    => "Post-Multiplied",
                wgpu::CompositeAlphaMode::Inherit           => "Inherit",
            }, rect));
            pos[0] += w + margin;
        }
        pos[0] = inner.position[0];
        pos[1] += h + margin;
    }
}

