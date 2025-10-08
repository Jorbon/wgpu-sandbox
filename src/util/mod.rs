pub mod vector; #[allow(unused_imports)] pub use vector::*;

#[allow(unused_imports)] use crate::*;

// Every crate does this so I guess I will too
pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, Error>;


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



pub const RECT_VERTICES: &[Vector<f32, 2>] = &[
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
    pub const fn new(position: Vector<f32, 2>, size: Vector<f32, 2>) -> Self {
        Self { position, size }
    }
    pub fn from(position: impl Into<Vector<f32, 2>>, size: impl Into<Vector<f32, 2>>) -> Self {
        Self::new(position.into(), size.into())
    }
    pub const fn offset(&self, amount: Vector<f32, 2>) -> Self {
        Self { position: Vector([*self.position.x() + *amount.x(), *self.position.y() + *amount.y()]), size: self.size }
    }
    pub const fn inset(&self, margin: f32) -> Self {
        Self { position: Vector([*self.position.x() + margin, *self.position.y() + margin]), size: Vector([*self.size.x() - margin * 2.0, *self.size.y() - margin * 2.0]) }
    }
    pub const fn left(&self) -> f32 { *self.position.x() }
    pub const fn right(&self) -> f32 { *self.position.x() + *self.size.x() }
    pub const fn top(&self) -> f32 { *self.position.y() }
    pub const fn bottom(&self) -> f32 { *self.position.y() + *self.size.y() }
    pub const fn width(&self) -> f32 { *self.size.x() }
    pub const fn height(&self) -> f32 { *self.size.y() }
    
    pub const fn contains_point(&self, point: Vector<f32, 2>) -> bool {
        *point.x() >= self.left() && *point.x() < self.right() && *point.y() >= self.top() && *point.y() < self.bottom()
    }
}


#[derive(Debug, Copy, Clone)]
pub struct BoxArea {
    pub rect: Rect,
    pub color: Color,
    pub id: Option<MenuID>,
}

impl BoxArea {
    pub fn new(rect: impl Into<Rect>, color: impl Into<Color>, id: Option<MenuID>) -> Self {
        Self { rect: rect.into(), color: color.into(), id }
    }
    
    pub fn new_centered(center: impl Into<Vector<f32, 2>>, size: impl Into<Vector<f32, 2>>, color: impl Into<Color>, id: Option<MenuID>) -> Self {
        let size = size.into();
        Self {
            rect: Rect::new(center.into() - size.scale(0.5), size),
            color: color.into(),
            id,
        }
    }
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

pub struct TextArea {
    pub buffer: glyphon::Buffer,
    // pub properties: TextProperties,
    pub rect: Rect,
}

impl TextArea {
    pub fn new_with_rect(font_system: &mut glyphon::FontSystem, text: &str, rect: Rect) -> Self {
        let mut buffer = glyphon::Buffer::new_empty(glyphon::Metrics::new(20.0, 20.0));
        buffer.set_wrap(font_system, glyphon::Wrap::None);
        buffer.set_text(font_system, text, &glyphon::Attrs {
            color_opt: None,
            family: glyphon::Family::Name("Luciole"),
            stretch: glyphon::Stretch::Normal,
            style: glyphon::Style::Normal,
            weight: glyphon::Weight::NORMAL,
            letter_spacing_opt: None,
            font_features: glyphon::cosmic_text::FontFeatures::new(),
            metadata: 0,
            cache_key_flags: glyphon::cosmic_text::CacheKeyFlags::empty(),
            metrics_opt: None,
        }, glyphon::Shaping::Basic);
        
        Self { buffer, rect }
    }
    
    pub fn new(font_system: &mut glyphon::FontSystem, text: &str) -> Self {
        Self::new_with_rect(font_system, text, Rect::new(Vector([0.0, 0.0]), Vector([0.0, 0.0])))
    }
}


