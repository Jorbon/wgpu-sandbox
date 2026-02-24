use crate::Vector;


#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Color(pub Vector<f32, 4>);

impl Color {
    pub const BLACK  : Self = Self::gray(0.0);
    pub const WHITE  : Self = Self::gray(1.0);
    pub const RED    : Self = Self::rgb(1.0, 0.0, 0.0);
    pub const GREEN  : Self = Self::rgb(0.0, 1.0, 0.0);
    pub const BLUE   : Self = Self::rgb(0.0, 0.0, 1.0);
    pub const YELLOW : Self = Self::rgb(1.0, 1.0, 0.0);
    pub const CYAN   : Self = Self::rgb(0.0, 1.0, 1.0);
    pub const MAGENTA: Self = Self::rgb(1.0, 0.0, 1.0);
    
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

impl Default for Color {
    fn default() -> Self {
        Self::BLACK
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

