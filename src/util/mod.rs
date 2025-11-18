use crate::*;

pub mod vector;
use glyphon::Attrs;
pub use vector::*;

use glyphon::TextBounds;


// Every crate does this so I guess I will too
pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, Error>;


#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, bytemuck::Zeroable)]
pub struct Color(pub Vector<f32, 4>);

unsafe impl bytemuck::Pod for Color {}

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
    pub id: MenuID,
}

impl BoxArea {
    pub fn new(rect: impl Into<Rect>, color: impl Into<Color>, id: MenuID) -> Self {
        Self { rect: rect.into(), color: color.into(), id }
    }
    
    pub fn new_centered(center: impl Into<Vector<f32, 2>>, size: impl Into<Vector<f32, 2>>, color: impl Into<Color>, id: MenuID) -> Self {
        let size = size.into();
        Self {
            rect: Rect::new(center.into() - size.scale(0.5), size),
            color: color.into(),
            id,
        }
    }
}


pub type TextAlign = glyphon::cosmic_text::Align;
pub type TextStyle = glyphon::cosmic_text::Style;
pub type TextWeight = glyphon::cosmic_text::Weight;
pub type TextStretch = glyphon::cosmic_text::Stretch;
pub type TextMetrics = glyphon::Metrics;
pub use glyphon::cosmic_text::LetterSpacing;
pub use glyphon::cosmic_text::FontFeatures;

pub struct TextProperties {
    pub size: f32,
    pub line_height: f32,
    pub font: String,
    pub color: Color,
    pub align: TextAlign,
    pub stretch: TextStretch,
    pub style: TextStyle,
    pub weight: TextWeight,
    pub letter_spacing: Option<LetterSpacing>,
    pub font_features: FontFeatures,
}

impl Default for TextProperties {
    fn default() -> Self { Self {
        size: 12.0,
        line_height: 12.0,
        font: "Luciole".to_owned(),
        color: Color::WHITE,
        align: TextAlign::Left,
        stretch: TextStretch::Normal,
        style: TextStyle::Normal,
        weight: TextWeight::NORMAL,
        letter_spacing: None,
        font_features: FontFeatures::new(),
    } }
}

impl TextProperties {
    pub fn attrs<'a>(&'a self) -> glyphon::Attrs<'a> {
        glyphon::Attrs {
            color_opt: Some(self.color.into()),
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
    
    pub fn metrics(&self) -> TextMetrics {
        TextMetrics {
            font_size: self.size,
            line_height: self.line_height,
        }
    }
}

pub struct TextPropertiesConst {
    pub size: f32,
    pub line_height: f32,
    pub font: &'static str,
    pub color: Color,
    pub align: TextAlign,
    pub stretch: TextStretch,
    pub style: TextStyle,
    pub weight: TextWeight,
    pub letter_spacing: Option<LetterSpacing>,
    pub font_features: FontFeatures,
}

impl From<TextPropertiesConst> for TextProperties {
    fn from(value: TextPropertiesConst) -> Self {
        Self {
            size: value.size,
            line_height: value.line_height,
            font: value.font.to_owned(),
            color: value.color,
            align: value.align,
            stretch: value.stretch,
            style: value.style,
            weight: value.weight,
            letter_spacing: value.letter_spacing,
            font_features: value.font_features,
        }
    }
}

impl TextPropertiesConst {
    pub const fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }
    pub const fn with_align(mut self, align: TextAlign) -> Self {
        self.align = align;
        self
    }
}



pub struct TextArea {
    buffer: glyphon::Buffer,
    pub text: String,
    pub properties: TextProperties,
    pub rect: Rect,
    pub text_width: f32,
}

impl TextArea {
    pub fn new_with_rect(font_system: &mut glyphon::FontSystem, properties: TextProperties, text: &str, rect: Rect) -> Self {
        let mut buffer = glyphon::Buffer::new_empty(properties.metrics());
        buffer.set_wrap(font_system, glyphon::Wrap::None);
        buffer.set_size(font_system, Some(rect.width()), None);
        buffer.set_rich_text(font_system, std::iter::once((text, properties.attrs())), &Attrs::new(), glyphon::Shaping::Basic, Some(properties.align));
        Self { buffer, text: text.to_owned(), properties, rect, text_width: 0.0 }
    }
    
    pub fn new(font_system: &mut glyphon::FontSystem, properties: TextProperties, text: &str) -> Self {
        Self::new_with_rect(font_system, properties, text, Rect::new(Vector([0.0, 0.0]), Vector([0.0, 0.0])))
    }
    
    pub fn set_rect(&mut self, font_system: &mut glyphon::FontSystem, rect: Rect) {
        self.rect = rect;
        self.buffer.set_size(font_system, Some(rect.width()), Some(rect.height()));
    }
    
    pub fn set_text(&mut self, font_system: &mut glyphon::FontSystem, text: &str) {
        self.text = text.to_owned();
        self.buffer.set_rich_text(font_system, std::iter::once((self.text.as_str(), self.properties.attrs())), &Attrs::new(), glyphon::Shaping::Basic, Some(self.properties.align));
        self.text_width = self.buffer.layout_runs().map(|run| run.line_w).reduce(f32::max).unwrap_or(0.0);
    }
    
    pub fn set_properties(&mut self, font_system: &mut glyphon::FontSystem, properties: TextProperties) {
        self.properties = properties;
        self.buffer.set_metrics(font_system, self.properties.metrics());
        self.buffer.set_rich_text(font_system, std::iter::once((self.text.as_str(), self.properties.attrs())), &Attrs::new(), glyphon::Shaping::Basic, Some(self.properties.align));
        self.text_width = self.buffer.layout_runs().map(|run| run.line_w).reduce(f32::max).unwrap_or(0.0);
    }
    
    pub fn get_render_object(&self, scale_factor: f32) -> glyphon::TextArea<'_> {
        let left   = self.rect.left  () * scale_factor;
        let right  = self.rect.right () * scale_factor;
        let top    = self.rect.top   () * scale_factor;
        let bottom = self.rect.bottom() * scale_factor;
        glyphon::TextArea {
            buffer: &self.buffer,
            left,
            top,
            scale: scale_factor,
            bounds: TextBounds {
                left: left as i32,
                top: top as i32,
                right: right as i32,
                bottom: bottom as i32,
            },
            default_color: Color::BLACK.into(),
            custom_glyphs: &[],
        }
    }
}


pub trait MenuHaver {
    const TEXT_PROPERTIES: TextPropertiesConst;
    fn iter_text_areas(&self) -> impl Iterator<Item = &TextArea>;
    
    fn text_area(font_system: &mut glyphon::FontSystem, text: &str) -> TextArea {
        TextArea::new(font_system, Self::TEXT_PROPERTIES.into(), text)
    }
    fn text_area_with_rect(font_system: &mut glyphon::FontSystem, text: &str, rect: Rect) -> TextArea {
        TextArea::new_with_rect(font_system, Self::TEXT_PROPERTIES.into(), text, rect)
    }
}


