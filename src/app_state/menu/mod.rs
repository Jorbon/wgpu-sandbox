use crate::*;

pub mod graphics;
pub use graphics::*;


#[derive(Debug, Copy, Clone)]
pub enum MenuID {
    Block,
    Pass,
    Graphics(GraphicsMenuID),
}


pub struct Menu {
    pub fps_text: TextArea,
    pub controls_text: TextArea,
    pub graphics: GraphicsMenu,
}

impl MenuHaver for Menu {
    const TEXT_PROPERTIES: TextPropertiesConst = TextPropertiesConst {
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
    
    fn iter_text_areas(&self) -> impl Iterator<Item = &TextArea> {
        [&self.fps_text, &self.controls_text].into_iter()
            .chain(self.graphics.iter_text_areas())
    }
}

impl Menu {
    pub fn new(font_system: &mut glyphon::FontSystem) -> Self {
        Self {
            fps_text: Self::text_area(font_system, "Fps: No information"),
            controls_text: Self::text_area(font_system, "Controls: Mouse, WASD, Space, Shift"),
            graphics: GraphicsMenu::new(font_system),
        }
    }
    
    pub fn layout(&mut self, font_system: &mut glyphon::FontSystem, boxes: &mut Vec<BoxArea>, logical_size: Vec2<f32>, window_state: &WindowState, graphics_options: &GraphicsOptions) {
        let fps = BoxArea::new(Rect::from([0.0, 0.0], [300.0, 40.0]), Color::gray_alpha(0.0, 0.2), MenuID::Pass);
        self.fps_text.set_rect(font_system, fps.rect);
        let controls = BoxArea::new(Rect::from([0.0, logical_size.y() - 40.0], [500.0, 40.0]), Color::gray_alpha(0.0, 0.2), MenuID::Pass);
        self.controls_text.set_rect(font_system, controls.rect);
        self.graphics.layout(font_system, boxes, logical_size, window_state, graphics_options);
        
        boxes.append(&mut vec![fps, controls]);
    }
}

