use crate::{Color, Vector};


#[derive(Debug, Copy, Clone)]
pub struct Rectangle {
    pub position: Vector<f32, 2>,
    pub size: Vector<f32, 2>,
}

impl Rectangle {
    pub fn new(position: impl Into<Vector<f32, 2>>, size: impl Into<Vector<f32, 2>>) -> Self {
        Self {
            position: position.into(),
            size: size.into(),
        }
    }
    
    pub fn new_centered(center: impl Into<Vector<f32, 2>>, size: impl Into<Vector<f32, 2>>) -> Self {
        let size = size.into();
        Self {
            position: center.into() - size * 0.5,
            size, 
        }
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


#[derive(Debug, Clone)]
pub enum Shape {
    Rectangle(Rectangle),
}

impl Shape {
    pub fn contains_point(&self, point: Vector<f32, 2>) -> bool {
        match self {
            Self::Rectangle(rect) => rect.contains_point(point),
        }
    }
}


#[derive(Debug, Clone)]
pub struct ShapeArea<AreaID> {
    pub shape: Shape,
    pub color: Color,
    pub id: AreaID,
}


impl<AreaID> ShapeArea<AreaID> {
    pub fn new(shape: impl Into<Shape>, color: impl Into<Color>, id: AreaID) -> Self {
        Self {
            shape: shape.into(),
            color: color.into(),
            id,
        }
    }
}

