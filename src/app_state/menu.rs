use crate::*;

#[derive(Debug, Copy, Clone)]
pub enum MenuID {
    Backend(wgpu::Backend),
    PowerPreference(wgpu::PowerPreference),
    PresentMode(wgpu::PresentMode),
    SurfaceFormat(wgpu::TextureFormat),
    AlphaMode(wgpu::CompositeAlphaMode),
}


pub struct GraphicsMenu {
    pub headers: [TextArea; 5],
    pub backends: Vec<TextArea>,
    pub power_preferences: Vec<TextArea>,
    pub present_modes: Vec<TextArea>,
    pub surface_formats: Vec<TextArea>,
    pub alpha_modes: Vec<TextArea>,
}

impl GraphicsMenu {
    pub fn new(font_system: &mut glyphon::FontSystem) -> Self {
        Self {
            headers: [
                TextArea::new(font_system, "Backends"),
                TextArea::new(font_system, "Power Preference"),
                TextArea::new(font_system, "Present Mode"),
                TextArea::new(font_system, "Surface Format"),
                TextArea::new(font_system, "Alpha Mode"),
            ],
            backends: vec![],
            power_preferences: vec![
                TextArea::new(font_system, "Default"),
                TextArea::new(font_system, "Low Power"),
                TextArea::new(font_system, "High Performance"),
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
}

