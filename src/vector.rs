#[allow(unused_imports)]
use std::{iter::Sum, ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign}};

use num_traits::{ConstOne, ConstZero, Float, One, Zero};


#[repr(C)]
#[derive(Debug, Copy, Clone, Eq, PartialEq, bytemuck::Zeroable)]
pub struct Vector<T, const N: usize>(pub [T; N]);

pub type Matrix<T, const M: usize, const N: usize> = Vector<Vector<T, N>, M>;


// Standard library extensions

impl<T, const N: usize> Vector<T, N> {
    #[inline] pub fn iter(&self) -> core::slice::Iter<T> { self.0.iter() }
    #[inline] pub fn iter_mut(&mut self) -> core::slice::IterMut<T> { self.0.iter_mut() }
    #[inline] pub fn each_ref(&self) -> Vector<&T, N> { Vector(self.0.each_ref()) }
    #[inline] pub fn each_mut(&mut self) -> Vector<&mut T, N> { Vector(self.0.each_mut()) }
    #[inline] pub fn map<F, U>(self, f: F) -> Vector<U, N> where F: FnMut(T) -> U { Vector(self.0.map(f)) }
    
    #[inline]
    pub fn map_with<F, U, V>(self, rhs: Vector<U, N>, mut f: F) -> Vector<V, N>
    where F: FnMut(T, U) -> V {
        let mut self_iter = self.into_iter();
        let mut rhs_iter = rhs.into_iter();
        Vector(core::array::from_fn(|_| f(self_iter.next().unwrap(), rhs_iter.next().unwrap())))
    }
    
    #[inline]
    pub fn transform<U, const M: usize>(self, rhs: Matrix<U, N, M>) -> Vector<<U as Mul<T>>::Output, M>
    where
        T: Copy,
        U: Copy + Mul<T>,
        <U as Mul<T>>::Output: Sum,
    {
        let Vector([value]) = rhs * self.as_column();
        value
    }
}

impl<T, const N: usize, I> Index<I> for Vector<T, N> where [T; N]: Index<I> {
    type Output = <[T; N] as Index<I>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, const N: usize, I> IndexMut<I> for Vector<T, N> where [T; N]: IndexMut<I> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<T, const N: usize> IntoIterator for Vector<T, N> {
    type Item = T;
    type IntoIter = <[T; N] as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}




// Common access methods

impl<T> Vector<T, 2> {
    #[inline] pub const fn x(&self) -> &T { &self.0[0] }
    #[inline] pub const fn y(&self) -> &T { &self.0[1] }
}

impl<T> Vector<T, 3> {
    #[inline] pub const fn x(&self) -> &T { &self.0[0] }
    #[inline] pub const fn y(&self) -> &T { &self.0[1] }
    #[inline] pub const fn z(&self) -> &T { &self.0[2] }
}

impl<T> Vector<T, 4> {
    #[inline] pub const fn x(&self) -> &T { &self.0[0] }
    #[inline] pub const fn y(&self) -> &T { &self.0[1] }
    #[inline] pub const fn z(&self) -> &T { &self.0[2] }
    #[inline] pub const fn w(&self) -> &T { &self.0[3] }
}


impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    pub fn column(&self, index: usize) -> Vector<&T, N> {
        self.index(index).each_ref()
    }
    
    pub fn column_mut(&mut self, index: usize) -> Vector<&mut T, N> {
        self.index_mut(index).each_mut()
    }
    
    pub fn row(&self, index: usize) -> Vector<&T, M> {
        self.each_ref().map(|v| v.index(index))
    }
    
    pub fn row_mut(&mut self, index: usize) -> Vector<&mut T, M> {
        self.each_mut().map(|v| v.index_mut(index))
    }
}

impl<T, const N: usize> Vector<T, N> {
    pub fn as_column(self) -> Matrix<T, 1, N> {
        Vector([self])
    }
    
    pub fn as_row(self) -> Matrix<T, N, 1> {
        self.map(|t| Vector([t]))
    }
}

// impl<T, const N: usize> Matrix<T, 1, N> {
//     pub fn as_vector(self) -> Vector<T, N> {
//         let Vector([value]) = self;
//         value
//     }
// }

impl<T, const N: usize> Matrix<T, N, 1> {
    pub fn as_vector(self) -> Vector<T, N> {
        self.map(|v| {
            let Vector([value]) = v;
            value
        })
    }
}


// Basic trait impls

unsafe impl<T: bytemuck::Pod, const N: usize> bytemuck::Pod for Vector<T, N> {}

impl<T, const N: usize> Default for Vector<T, N>
where [T; N]: Default {
    fn default() -> Self {
        Self(<[T; N]>::default())
    }
}


impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T> From<(T, T)> for Vector<T, 2> { fn from(value: (T, T)) -> Self { Self(value.into()) } }
impl<T> From<(T, T, T)> for Vector<T, 3> { fn from(value: (T, T, T)) -> Self { Self(value.into()) } }
impl<T> From<(T, T, T, T)> for Vector<T, 4> { fn from(value: (T, T, T, T)) -> Self { Self(value.into()) } }


// Operations

impl<T: Neg, const N: usize> Neg for Vector<T, N> { type Output = Vector<<T as Neg>::Output, N>; fn neg(self) -> Self::Output { self.map(|t| -t) } }
impl<T: Not, const N: usize> Not for Vector<T, N> { type Output = Vector<<T as Not>::Output, N>; fn not(self) -> Self::Output { self.map(|t| !t) } }

impl<T: Add   <U>, U, const N: usize> Add   <Vector<U, N>> for Vector<T, N> { type Output = Vector<<T as Add   <U>>::Output, N>; fn add   (self, rhs: Vector<U, N>) -> Self::Output { self.map_with(rhs, T::add   ) } }
impl<T: Sub   <U>, U, const N: usize> Sub   <Vector<U, N>> for Vector<T, N> { type Output = Vector<<T as Sub   <U>>::Output, N>; fn sub   (self, rhs: Vector<U, N>) -> Self::Output { self.map_with(rhs, T::sub   ) } }
impl<T: BitAnd<U>, U, const N: usize> BitAnd<Vector<U, N>> for Vector<T, N> { type Output = Vector<<T as BitAnd<U>>::Output, N>; fn bitand(self, rhs: Vector<U, N>) -> Self::Output { self.map_with(rhs, T::bitand) } }
impl<T: BitOr <U>, U, const N: usize> BitOr <Vector<U, N>> for Vector<T, N> { type Output = Vector<<T as BitOr <U>>::Output, N>; fn bitor (self, rhs: Vector<U, N>) -> Self::Output { self.map_with(rhs, T::bitor ) } }
impl<T: BitXor<U>, U, const N: usize> BitXor<Vector<U, N>> for Vector<T, N> { type Output = Vector<<T as BitXor<U>>::Output, N>; fn bitxor(self, rhs: Vector<U, N>) -> Self::Output { self.map_with(rhs, T::bitxor) } }
impl<T: Shl   <U>, U, const N: usize> Shl   <Vector<U, N>> for Vector<T, N> { type Output = Vector<<T as Shl   <U>>::Output, N>; fn shl   (self, rhs: Vector<U, N>) -> Self::Output { self.map_with(rhs, T::shl   ) } }
impl<T: Shr   <U>, U, const N: usize> Shr   <Vector<U, N>> for Vector<T, N> { type Output = Vector<<T as Shr   <U>>::Output, N>; fn shr   (self, rhs: Vector<U, N>) -> Self::Output { self.map_with(rhs, T::shr   ) } }

// impl<T: Mul<U>, U: Copy, const N: usize> Mul<U> for Vector<T, N> { type Output = Vector<<T as Mul<U>>::Output, N>; fn mul(self, rhs: U) -> Self::Output { self.map(|t| t.mul(rhs)) } }
// impl<T: Div<U>, U: Copy, const N: usize> Div<U> for Vector<T, N> { type Output = Vector<<T as Div<U>>::Output, N>; fn div(self, rhs: U) -> Self::Output { self.map(|t| t.div(rhs)) } }
// impl<T: Rem<U>, U: Copy, const N: usize> Rem<U> for Vector<T, N> { type Output = Vector<<T as Rem<U>>::Output, N>; fn rem(self, rhs: U) -> Self::Output { self.map(|t| t.rem(rhs)) } }

impl<T: AddAssign   <U>, U, const N: usize> AddAssign   <Vector<U, N>> for Vector<T, N> { fn add_assign   (&mut self, rhs: Vector<U, N>) { self.iter_mut().zip(rhs).for_each(|(t, u)| t.add_assign   (u)) } }
impl<T: SubAssign   <U>, U, const N: usize> SubAssign   <Vector<U, N>> for Vector<T, N> { fn sub_assign   (&mut self, rhs: Vector<U, N>) { self.iter_mut().zip(rhs).for_each(|(t, u)| t.sub_assign   (u)) } }
impl<T: BitAndAssign<U>, U, const N: usize> BitAndAssign<Vector<U, N>> for Vector<T, N> { fn bitand_assign(&mut self, rhs: Vector<U, N>) { self.iter_mut().zip(rhs).for_each(|(t, u)| t.bitand_assign(u)) } }
impl<T: BitOrAssign <U>, U, const N: usize> BitOrAssign <Vector<U, N>> for Vector<T, N> { fn bitor_assign (&mut self, rhs: Vector<U, N>) { self.iter_mut().zip(rhs).for_each(|(t, u)| t.bitor_assign (u)) } }
impl<T: BitXorAssign<U>, U, const N: usize> BitXorAssign<Vector<U, N>> for Vector<T, N> { fn bitxor_assign(&mut self, rhs: Vector<U, N>) { self.iter_mut().zip(rhs).for_each(|(t, u)| t.bitxor_assign(u)) } }
impl<T: ShlAssign   <U>, U, const N: usize> ShlAssign   <Vector<U, N>> for Vector<T, N> { fn shl_assign   (&mut self, rhs: Vector<U, N>) { self.iter_mut().zip(rhs).for_each(|(t, u)| t.shl_assign   (u)) } }
impl<T: ShrAssign   <U>, U, const N: usize> ShrAssign   <Vector<U, N>> for Vector<T, N> { fn shr_assign   (&mut self, rhs: Vector<U, N>) { self.iter_mut().zip(rhs).for_each(|(t, u)| t.shr_assign   (u)) } }

// impl<T: MulAssign<U>, U: Copy, const N: usize> MulAssign<U> for Vector<T, N> { fn mul_assign(&mut self, rhs: U) { self.iter_mut().for_each(|t| t.mul_assign(rhs)) } }
// impl<T: DivAssign<U>, U: Copy, const N: usize> DivAssign<U> for Vector<T, N> { fn div_assign(&mut self, rhs: U) { self.iter_mut().for_each(|t| t.div_assign(rhs)) } }
// impl<T: RemAssign<U>, U: Copy, const N: usize> RemAssign<U> for Vector<T, N> { fn rem_assign(&mut self, rhs: U) { self.iter_mut().for_each(|t| t.rem_assign(rhs)) } }


// Other arithmetic methods

impl<T, const N: usize> Vector<T, N> {
    pub fn scale<U>(self, rhs: U) -> Vector<<T as Mul<U>>::Output, N>
    where
        T: Mul<U>,
        U: Copy,
    {
        self.map(|t| t * rhs)
    }
    
    pub fn scale_axes<U, V>(self, rhs: V) -> Vector<<T as Mul<U>>::Output, N>
    where
        T: Mul<U>,
        V: Into<Vector<U, N>>,
    {
        self.map_with(rhs.into(), T::mul)
    }
}


// Dot and cross products

impl<T, const N: usize> Vector<T, N> {
    pub fn dot<U>(self, rhs: Vector<U, N>) -> <T as Mul<U>>::Output
    where
        T: Mul<U>,
        <T as Mul<U>>::Output: Sum,
    {
        self.map_with(rhs, T::mul).into_iter().sum()
    }
}

impl<T> Vector<T, 3> {
    pub fn cross<U>(self, rhs: Vector<U, 3>) -> Vector<<<T as Mul<U>>::Output as Sub>::Output, 3>
    where
        T: Copy + Mul<U>,
        U: Copy,
        <T as Mul<U>>::Output: Sub,
    {
        Vector([
            *self.y() * *rhs.z() - *self.z() * *rhs.y(),
            *self.z() * *rhs.x() - *self.x() * *rhs.z(),
            *self.x() * *rhs.y() - *self.y() * *rhs.x(),
        ])
    }
}


// Matrix multiplication

impl<T, U, const L: usize, const M:usize, const N: usize> Mul<Matrix<U, L, M>> for Matrix<T, M, N>
where
    T: Copy + Mul<U>,
    U: Copy,
    T: Mul<U>,
    <T as Mul<U>>::Output: Sum,
{
    type Output = Matrix<<T as Mul<U>>::Output, L, N>;
    fn mul(self, rhs: Matrix<U, L, M>) -> Self::Output {
        Vector(core::array::from_fn(|i|
            Vector(core::array::from_fn(|j|
                self.row(j).map_with(rhs.column(i), |t, u| *t * *u).into_iter().sum()
            ))
        ))
    }
}


// Zero and One traits

impl<T: Zero, const N: usize> Zero for Vector<T, N> {
    fn zero() -> Self {
        Self(core::array::from_fn(|_i| T::zero()))
    }
    fn is_zero(&self) -> bool {
        self.iter().all(|t| t.is_zero())
    }
}

impl<T: ConstZero, const N: usize> ConstZero for Vector<T, N> {
    const ZERO: Self = Self([T::ZERO; N]);
}


impl<T: Copy + Zero + One + Sum, const N: usize> One for Matrix<T, N, N> {
    fn one() -> Self {
        Vector(core::array::from_fn(|i|
            Vector(core::array::from_fn(|j|
                match i == j {
                    true => T::one(),
                    false => T::zero(),
                }
            ))
        ))
    }
}

impl<T: Copy + ConstZero + ConstOne + Sum, const N: usize> ConstOne for Matrix<T, N, N> {
    const ONE: Self = {
        let mut result: Self = unsafe { *std::mem::MaybeUninit::uninit().as_mut_ptr() };
        
        let mut i = 0;
        let mut j = 0;
        while i < N {
            while j < N {
                result.0[i].0[j] = match i == j {
                    true => T::ONE,
                    false => T::ZERO,
                };
                j += 1;
            }
            
            j = 0;
            i += 1;
        }
        
        result
    };
}


// Matrix transforms

impl<T: Copy + Sum + Zero + One, const N: usize> Matrix<T, N, N> {
    pub fn in_axes<const L: usize>(self, axes: [usize; N]) -> Matrix<T, L, L> {
        let mut m = Matrix::<T, L, L>::one();
        for i in 0..N {
            if axes[i] >= L { continue }
            for j in 0..N {
                if axes[j] >= L { continue }
                m[axes[i]][axes[j]] = self[i][j];
            }
        }
        m
    }
}

pub fn rotate_transform<T: Float>(angle: T) -> Matrix<T, 2, 2> {
    let c = angle.cos();
    let s = angle.sin();
    Vector([
        Vector([c, -s]),
        Vector([s,  c]),
    ])
}

pub fn rotate_axes<T: Float + Sum, const N: usize>(axes: [usize; 2], angle: T) -> Matrix<T, N, N> {
    rotate_transform(angle).in_axes(axes)
}

pub fn scale_axes<T, V, const N: usize>(scale: V) -> Matrix<T, N, N>
where
    T: Copy + Zero,
    V: Into<Vector<T, N>>,
{
    let v = scale.into();
    Vector(core::array::from_fn(|i|
        Vector(core::array::from_fn(|j|
            match i == j {
                true => v[i],
                false => T::zero(),
            }
        ))
    ))
}

pub fn translate_2d<T, V>(offset: V) -> Matrix<T, 3, 3>
where
    T: Copy + Sum + Zero + One + AddAssign,
    V: Into<Vector<T, 2>>,
{
    let mut m = Matrix::one();
    let v = offset.into();
    for i in 0..2 {
        m[2][i] += v[i];
    }
    m
}

pub fn translate_3d<T, V>(offset: V) -> Matrix<T, 4, 4>
where
    T: Copy + Sum + Zero + One + AddAssign,
    V: Into<Vector<T, 3>>,
{
    let mut m = Matrix::one();
    let v = offset.into();
    for i in 0..3 {
        m[3][i] += v[i];
    }
    m
}




pub type Vec2<T> = Vector<T, 2>;
pub type Vec3<T> = Vector<T, 3>;
pub type Vec4<T> = Vector<T, 4>;
pub type Mat2x2<T> = Matrix<T, 2, 2>;
pub type Mat2x3<T> = Matrix<T, 2, 3>;
pub type Mat2x4<T> = Matrix<T, 2, 4>;
pub type Mat3x2<T> = Matrix<T, 3, 2>;
pub type Mat3x3<T> = Matrix<T, 3, 3>;
pub type Mat3x4<T> = Matrix<T, 3, 4>;
pub type Mat4x2<T> = Matrix<T, 4, 2>;
pub type Mat4x3<T> = Matrix<T, 4, 3>;
pub type Mat4x4<T> = Matrix<T, 4, 4>;
