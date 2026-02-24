pub mod vector;
// pub mod matrix;
pub mod color;
pub mod shape;
pub mod text;

pub use vector::*;
// pub use matrix::*;
pub use color::*;
pub use shape::*;
pub use text::*;


// Every crate does this so I guess I will too
pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, Error>;



// pub enum IterLazy<T: Iterator, F: FnOnce() -> T> {
//     Asleep(F),
//     Active(T),
// }

// impl<T: Iterator, F: FnOnce() -> T> IterLazy<T, F> {
//     pub fn (&mut self) -> &mut T {
//         match self {
//             IterLazy::Asleep(f) => {
                
//             }
//         }
//     }
// }

// impl<T: Iterator, F: FnOnce() -> T> Iterator for IterLazy<T, F> {
//     type Item = T::Item;
//     fn next(&mut self) -> Option<Self::Item> {
//         match self {
//             IterLazy::Asleep(f) => {
//                 let mut iter = f();
//                 let next = iter.next();
//                 *self = IterLazy::Active(iter);
//                 next
//             }
//             IterLazy::Active(iter) => iter.next(),
//         }
//     }
// }


pub struct IterIf<T: Iterator> {
    pub condition: bool,
    pub iterator: T,
}

impl<T: Iterator> Iterator for IterIf<T> {
    type Item = T::Item;
    fn next(&mut self) -> Option<Self::Item> {
        match self.condition {
            true => self.iterator.next(),
            false => None,
        }
    }
}

pub fn iter_if<T: Iterator>(condition: bool, iterator: T) -> IterIf<T> {
    IterIf { condition, iterator }
}


