use crate::*;

pub mod wgpu_config;
pub use wgpu_config::*;


#[derive(Debug, Copy, Clone)]
pub enum RootAreaID {
    Block,
    Pass,
    WgpuConfig(WgpuConfigAreaID),
}




const DEFAULT_TEXT_PROPERTIES: TextPropertiesConst = TextPropertiesConst {
    size: 20.0,
    line_height: 20.0,
    font: "Luciole",
    color: Color::WHITE,
    align: TextAlign::Left,
    stretch: TextStretch::Normal,
    style: TextStyle::Normal,
    weight: TextWeight::NORMAL,
    letter_spacing: None,
    font_features: FontFeatures { features: vec![] },
};


pub struct RootMenu {
    pub wgpu_open: bool,
    pub fps_text: TextArea,
    pub controls_text: TextArea,
    pub wgpu_config: WgpuConfigMenu,
}

impl RootMenu {
    pub fn new(font_system: &mut glyphon::FontSystem) -> Self {
        Self {
            wgpu_open: true,
            fps_text: TextArea::new(font_system, DEFAULT_TEXT_PROPERTIES, "Fps: No information"),
            controls_text: TextArea::new(font_system, DEFAULT_TEXT_PROPERTIES, "Controls: Mouse, WASD, Space, Shift"),
            wgpu_config: WgpuConfigMenu::new(font_system),
        }
    }
    
    pub fn iter_text_areas(&self) -> impl Iterator<Item = &TextArea> {
        [&self.fps_text, &self.controls_text].into_iter()
            .chain(iter_if(self.wgpu_open, self.wgpu_config.iter_text_areas()))
    }
    
    pub fn layout(
        &mut self,
        font_system: &mut glyphon::FontSystem,
        shapes: &mut Vec<ShapeArea<RootAreaID>>,
        logical_size: Vec2<f32>,
        window_state: &WindowState,
        graphics_options: &GraphicsOptions,
    ) {
        let fps_rect = Rectangle::new([0.0, 0.0], [300.0, 40.0]);
        let fps = ShapeArea::new(Shape::Rectangle(fps_rect), Color::gray_alpha(0.0, 0.2), RootAreaID::Pass);
        self.fps_text.set_rect(font_system, fps_rect);
        
        let controls_rect = Rectangle::new([0.0, logical_size.y() - 40.0], [500.0, 40.0]);
        let controls = ShapeArea::new(Shape::Rectangle(controls_rect), Color::gray_alpha(0.0, 0.2), RootAreaID::Pass);
        self.controls_text.set_rect(font_system, controls_rect);
        
        if self.wgpu_open {
            self.wgpu_config.layout(font_system, shapes, logical_size, window_state, graphics_options);
        }
        
        shapes.append(&mut vec![fps, controls]);
    }
}

