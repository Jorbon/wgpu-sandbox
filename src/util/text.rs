use glyphon::{Attrs, TextBounds};

use crate::{Color, Rectangle, Vector};


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
    pub rectangle: Rectangle,
    pub text_width: f32,
}

impl TextArea {
    pub fn new_with_rect(font_system: &mut glyphon::FontSystem, properties: impl Into<TextProperties>, text: &str, rectangle: Rectangle) -> Self {
        let properties = properties.into();
        let mut buffer = glyphon::Buffer::new_empty(properties.metrics());
        buffer.set_wrap(font_system, glyphon::Wrap::None);
        buffer.set_size(font_system, Some(rectangle.width()), None);
        buffer.set_rich_text(font_system, std::iter::once((text, properties.attrs())), &Attrs::new(), glyphon::Shaping::Basic, Some(properties.align));
        Self {
            buffer,
            text: text.to_owned(),
            properties,
            rectangle,
            text_width: 0.0,
        }
    }
    
    pub fn new(font_system: &mut glyphon::FontSystem, properties: impl Into<TextProperties>, text: &str) -> Self {
        Self::new_with_rect(font_system, properties, text, Rectangle::new(Vector([0.0, 0.0]), Vector([0.0, 0.0])))
    }
    
    pub fn set_rect(&mut self, font_system: &mut glyphon::FontSystem, rect: Rectangle) {
        self.rectangle = rect;
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
        let left   = self.rectangle.left  () * scale_factor;
        let right  = self.rectangle.right () * scale_factor;
        let top    = self.rectangle.top   () * scale_factor;
        let bottom = self.rectangle.bottom() * scale_factor;
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

