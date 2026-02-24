use std::marker::PhantomData as PD;
#[allow(unused_imports)]
use std::{iter::Sum, ops::*};
use num_traits::*;


pub trait CountTrait {
    const VALUE: usize;
}

pub struct Count0;
pub struct Count<Inner: CountTrait>(PD<Inner>);

impl CountTrait for Count0 {
    const VALUE: usize = 0;
}
impl<Inner: CountTrait> CountTrait for Count<Inner> {
    const VALUE: usize = Inner::VALUE + 1;
}

pub trait ConstMinus<C: CountTrait>: CountTrait {
    type Difference: CountTrait;
}

impl<C: CountTrait> ConstMinus<Count0> for C {
    type Difference = C;
}

impl<LeftInner, RightInner> ConstMinus<Count<RightInner>> for Count<LeftInner>
where
    LeftInner: CountTrait + ConstMinus<RightInner>,
    RightInner: CountTrait,
{
    type Difference = <LeftInner as ConstMinus<RightInner>>::Difference;
}

pub type Count1 = Count<Count0>;
pub type Count2 = Count<Count1>;
pub type Count3 = Count<Count2>;
pub type Count4 = Count<Count3>;



pub trait ConstIterator {
    type T;
    type Count: CountTrait;
    const LENGTH: usize = Self::Count::VALUE;
    fn iter(&self) -> impl Iterator<Item = &Self::T>;
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::T>;
    fn iter_owned(self) -> impl Iterator<Item = Self::T>;
}

pub trait ConstIndexFromEnd<C: CountTrait>: ConstIterator {
    fn index_from_end(&self) -> &Self::T;
    fn index_from_end_mut(&mut self) -> &mut Self::T;
    fn index_from_end_owned(self) -> Self::T;
}

pub trait ConstIndex<C>:
Sized + ConstIterator<Count = Count<Self::LengthMinusOne>> + ConstIndexFromEnd<<Self::LengthMinusOne as ConstMinus<C>>::Difference>
where C: CountTrait
{
    type LengthMinusOne: CountTrait + ConstMinus<C>;
    fn index(&self) -> &Self::T {
        <Self as ConstIndexFromEnd<<Self::LengthMinusOne as ConstMinus<C>>::Difference>>::index_from_end(self)
    }
    fn index_mut(&mut self) -> &mut Self::T {
        <Self as ConstIndexFromEnd<<Self::LengthMinusOne as ConstMinus<C>>::Difference>>::index_from_end_mut(self)
    }
    fn index_owned(self) -> Self::T {
        <Self as ConstIndexFromEnd<<Self::LengthMinusOne as ConstMinus<C>>::Difference>>::index_from_end_owned(self)
    }
}



// Core

pub trait VectorTrait: ConstIterator {}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Vec0<T>(PD<T>);
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Vector<Inner: VectorTrait>(pub Inner, pub Inner::T);

impl<T> ConstIterator for Vec0<T> {
    type T = T;
    type Count = Count0;
    fn iter(&self) -> impl Iterator<Item = &Self::T> {
        std::iter::empty()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::T> {
        std::iter::empty()
    }
    fn iter_owned(self) -> impl Iterator<Item = Self::T> {
        std::iter::empty()
    }
}
impl<Inner: VectorTrait> ConstIterator for Vector<Inner> {
    type T = Inner::T;
    type Count = Count<Inner::Count>;
    fn iter(&self) -> impl Iterator<Item = &Self::T> {
        self.0.iter().chain(std::iter::once(&self.1))
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::T> {
        self.0.iter_mut().chain(std::iter::once(&mut self.1))
    }
    fn iter_owned(self) -> impl Iterator<Item = Self::T> {
        self.0.iter_owned().chain(std::iter::once(self.1))
    }
}

impl<T> VectorTrait for Vec0<T> {}
impl<Inner: VectorTrait> VectorTrait for Vector<Inner> {}


impl<Inner: VectorTrait> ConstIndexFromEnd<Count0> for Vector<Inner> {
    fn index_from_end(&self) -> &Self::T {
        &self.1
    }
    fn index_from_end_mut(&mut self) -> &mut Self::T {
        &mut self.1
    }
    fn index_from_end_owned(self) -> Self::T {
        self.1
    }
}
impl<VectorInner, CountInner> ConstIndexFromEnd<Count<CountInner>> for Vector<VectorInner>
where
    VectorInner: VectorTrait + ConstIndexFromEnd<CountInner>,
    CountInner: CountTrait,
{
    fn index_from_end(&self) -> &Self::T {
        <VectorInner as ConstIndexFromEnd<CountInner>>::index_from_end(&self.0)
    }
    fn index_from_end_mut(&mut self) -> &mut Self::T {
        <VectorInner as ConstIndexFromEnd<CountInner>>::index_from_end_mut(&mut self.0)
    }
    fn index_from_end_owned(self) -> Self::T {
        <VectorInner as ConstIndexFromEnd<CountInner>>::index_from_end_owned(self.0)
    }
}

impl<Inner, C> ConstIndex<C> for Vector<Inner>
where
    Inner: VectorTrait,
    C: CountTrait,
    Inner::Count: ConstMinus<C>,
    Self: ConstIterator<Count = Count<Inner::Count>> + ConstIndexFromEnd<<Inner::Count as ConstMinus<C>>::Difference>,
{
    type LengthMinusOne = Inner::Count;
}

impl<Inner: VectorTrait> Vector<Inner> {
    pub fn get_ref<C>(&self) -> &<Self as ConstIterator>::T
    where
        C: CountTrait,
        Self: ConstIndex<C>,
    {
        ConstIndex::<C>::index(self)
    }
    
    pub fn get_mut<C>(&mut self) -> &mut <Self as ConstIterator>::T
    where
        C: CountTrait,
        Self: ConstIndex<C>,
    {
        ConstIndex::<C>::index_mut(self)
    }
    
    pub fn get<C>(self) -> <Self as ConstIterator>::T
    where
        C: CountTrait,
        Self: ConstIndex<C>,
    {
        ConstIndex::<C>::index_owned(self)
    }
    
    pub fn x(self) -> <Self as ConstIterator>::T
    where Self: ConstIndex<Count0>
    {
        self.get::<Count0>()
    }
    
    pub fn y(self) -> <Self as ConstIterator>::T
    where Self: ConstIndex<Count1>
    {
        self.get::<Count1>()
    }
    
    pub fn z(self) -> <Self as ConstIterator>::T
    where Self: ConstIndex<Count2>
    {
        self.get::<Count2>()
    }
    
    pub fn w(self) -> <Self as ConstIterator>::T
    where Self: ConstIndex<Count3>
    {
        self.get::<Count3>()
    }
}


// Length-specific

pub type Vec1<T> = Vector<Vec0<T>>;
pub type Vec2<T> = Vector<Vec1<T>>;
pub type Vec3<T> = Vector<Vec2<T>>;
pub type Vec4<T> = Vector<Vec3<T>>;

pub const fn vec1<T>(x: T) -> Vec1<T> {
    Vector(Vec0(PD), x)
}

pub const fn vec2<T>(x: T, y: T) -> Vec2<T> {
    Vector(vec1(x), y)
}

pub const fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vector(vec2(x, y), z)
}

pub const fn vec4<T>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    Vector(vec3(x, y, z), w)
}



// Print
impl<Inner> std::fmt::Debug for Vector<Inner>
where
    Inner: VectorTrait,
    Inner::T: std::fmt::Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec{}( ", Self::LENGTH)?;
        for (i, value) in self.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:?}", value)?;
        }
        write!(f, " )")?;
        Ok(())
    }
}

impl<Inner> std::fmt::Display for Vector<Inner>
where
    Inner: VectorTrait,
    Inner::T: std::fmt::Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec{}( ", Self::LENGTH)?;
        for (i, value) in self.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", value)?;
        }
        write!(f, " )")?;
        Ok(())
    }
}


// Props

impl<T> Zero for Vec0<T>
where T: Zero
{
    fn zero() -> Self {
        Self(PD)
    }
    fn is_zero(&self) -> bool {
        true
    }
}
impl<Inner> Zero for Vector<Inner>
where
    Inner: VectorTrait + Zero,
    Inner::T: Zero,
{
    fn zero() -> Self {
        Self(Inner::zero(), Inner::T::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

impl<T> ConstZero for Vec0<T>
where T: ConstZero
{
    const ZERO: Self = Self(PD);
}
impl<Inner> ConstZero for Vector<Inner>
where
    Inner: VectorTrait + ConstZero,
    Inner::T: ConstZero,
{
    const ZERO: Self = Vector(Inner::ZERO, Inner::T::ZERO);
}


impl<T> From<()> for Vec0<T> {
    #[allow(unused_variables)]
    fn from(value: ()) -> Self {
        Self(PD)
    }
}
impl<Inner, I, T> From<(I, T)> for Vector<Inner>
where
    Inner: VectorTrait,
    I: Into<Inner>,
    T: Into<Inner::T>,
{
    fn from(value: (I, T)) -> Self {
        Self(value.0.into(), value.1.into())
    }
}


// Ops

impl<T> Neg for Vec0<T>
where T: Neg
{
    type Output = Vec0<<T as Neg>::Output>;
    fn neg(self) -> Self::Output {
        Vec0(PD)
    }
}
impl<Inner> Neg for Vector<Inner>
where
    Inner: VectorTrait + Neg,
    <Inner as Neg>::Output: VectorTrait,
    Inner::T: Neg<Output = <<Inner as Neg>::Output as ConstIterator>::T>,
{
    type Output = Vector<<Inner as Neg>::Output>;
    fn neg(self) -> Self::Output {
        Vector(-self.0, -self.1)
    }
}

impl<T> Not for Vec0<T>
where T: Not
{
    type Output = Vec0<<T as Not>::Output>;
    fn not(self) -> Self::Output {
        Vec0(PD)
    }
}
impl<Inner> Not for Vector<Inner>
where
    Inner: VectorTrait + Not,
    <Inner as Not>::Output: VectorTrait,
    Inner::T: Not<Output = <<Inner as Not>::Output as ConstIterator>::T>,
{
    type Output = Vector<<Inner as Not>::Output>;
    fn not(self) -> Self::Output {
        Vector(!self.0, !self.1)
    }
}


impl<LeftT, RightT> Add<Vec0<RightT>> for Vec0<LeftT>
where LeftT: Add<RightT>
{
    type Output = Vec0<<LeftT as Add<RightT>>::Output>;
    #[allow(unused_variables)]
    fn add(self, rhs: Vec0<RightT>) -> Self::Output {
        Vec0(PD)
    }
}
impl<LeftInner, RightInner> Add<Vector<RightInner>> for Vector<LeftInner>
where
    LeftInner: VectorTrait + Add<RightInner>,
    RightInner: VectorTrait,
    <LeftInner as Add<RightInner>>::Output: VectorTrait,
    LeftInner::T: Add<RightInner::T, Output = <<LeftInner as Add<RightInner>>::Output as ConstIterator>::T>,
{
    type Output = Vector<<LeftInner as Add<RightInner>>::Output>;
    fn add(self, rhs: Vector<RightInner>) -> Self::Output {
        Vector(self.0 + rhs.0, self.1 + rhs.1)
    }
}


impl<LeftT, RightT> Mul<RightT> for Vec0<LeftT>
where LeftT: Mul<RightT>
{
    type Output = Vec0<<LeftT as Mul<RightT>>::Output>;
    #[allow(unused_variables)]
    fn mul(self, rhs: RightT) -> Self::Output {
        Vec0(PD)
    }
}
impl<LeftInner, RightT> Mul<RightT> for Vector<LeftInner>
where
    LeftInner: VectorTrait + Mul<RightT>,
    <LeftInner as Mul<RightT>>::Output: VectorTrait,
    LeftInner::T: Mul<RightT, Output = <<LeftInner as Mul<RightT>>::Output as ConstIterator>::T>,
    RightT: Copy,
{
    type Output = Vector<<LeftInner as Mul<RightT>>::Output>;
    fn mul(self, rhs: RightT) -> Self::Output {
        Vector(self.0 * rhs, self.1 * rhs)
    }
}


// Custom ops

pub trait Dot<Rhs = Self> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}
impl<LeftT, RightT> Dot<Vec1<RightT>> for Vec1<LeftT>
where LeftT: Mul<RightT>
{
    type Output = <LeftT as Mul<RightT>>::Output;
    fn dot(self, rhs: Vec1<RightT>) -> Self::Output {
        self.1 * rhs.1
    }
}
impl<LeftInner, RightInner> Dot<Vector<Vector<RightInner>>> for Vector<Vector<LeftInner>>
where
    LeftInner: VectorTrait,
    RightInner: VectorTrait,
    Vector<LeftInner>: Dot<Vector<RightInner>>,
    LeftInner::T: Mul<RightInner::T>,
    <Vector<LeftInner> as Dot<Vector<RightInner>>>::Output: Add<<LeftInner::T as Mul<RightInner::T>>::Output>,
{
    type Output = <<Vector<LeftInner> as Dot<Vector<RightInner>>>::Output as Add<<LeftInner::T as Mul<RightInner::T>>::Output>>::Output;
    fn dot(self, rhs: Vector<Vector<RightInner>>) -> Self::Output {
        self.0.dot(rhs.0) + self.1 * rhs.1
    }
}


impl<LeftT> Vec3<LeftT> {
    pub fn cross<RightT>(self, rhs: Vec3<RightT>) -> Vec3<<<LeftT as Mul<RightT>>::Output as Sub<<LeftT as Mul<RightT>>::Output>>::Output>
    where
        LeftT: Copy + Mul<RightT>,
        RightT: Copy,
        <LeftT as Mul<RightT>>::Output: Sub<<LeftT as Mul<RightT>>::Output>,
        // <<LeftT as Mul<RightT>>::Output as Add<<LeftT as Mul<RightT>>::Output>>::Output: Add<<LeftT as Mul<RightT>>::Output>,
    {
        vec3(
            self.y() * rhs.z() - self.z() * rhs.y(),
            self.z() * rhs.x() - self.x() * rhs.z(),
            self.x() * rhs.y() - self.y() * rhs.x(),
        )
    }
}




pub trait MatrixTrait: ConstIterator {}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Matrix0<Row: VectorTrait>(PD<Row>);
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Matrix<Inner: MatrixTrait>(pub Inner, pub Inner::T);

impl<Row: VectorTrait> ConstIterator for Matrix0<Row> {
    type T = Row;
    type Count = Count0;
    fn iter(&self) -> impl Iterator<Item = &Self::T> {
        std::iter::empty()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::T> {
        std::iter::empty()
    }
    fn iter_owned(self) -> impl Iterator<Item = Self::T> {
        std::iter::empty()
    }
}
impl<Inner: MatrixTrait> ConstIterator for Matrix<Inner> {
    type T = Inner::T;
    type Count = Count<Inner::Count>;
    fn iter(&self) -> impl Iterator<Item = &Self::T> {
        self.0.iter().chain(std::iter::once(&self.1))
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::T> {
        self.0.iter_mut().chain(std::iter::once(&mut self.1))
    }
    fn iter_owned(self) -> impl Iterator<Item = Self::T> {
        self.0.iter_owned().chain(std::iter::once(self.1))
    }
}

impl<Row: VectorTrait> MatrixTrait for Matrix0<Row> {}
impl<Inner: MatrixTrait> MatrixTrait for Matrix<Inner> {}


pub trait MatrixColumnTrait {}
// pub struct MatrixColumn0






// Tests

const fn c<C1, C2>() -> <C1 as ConstMinus<C2>>::Difference
where C1: CountTrait + ConstMinus<C2>, C2: CountTrait
{
    todo!()
}

fn test() {
    let _b = c::<Count<Count<Count<Count0>>>, Count<Count<Count0>>>();
    
    let _a: Vec3<f32> = Vector(Vector(vec1(1.0), 2.0), 3.0);
    dbg!(ConstIndexFromEnd::<Count0>::index_from_end(&_a));
    dbg!(ConstIndexFromEnd::<Count<Count0>>::index_from_end(&_a));
    dbg!(ConstIndexFromEnd::<Count<Count<Count0>>>::index_from_end(&_a));
    // dbg!(ConstIndexFromEnd::<Count<Count<Count<Count0>>>>::index_from_end(&_a)); -- doesn't compile
    
    dbg!(ConstIndex::<Count0>::index(&_a));
    dbg!(ConstIndex::<Count<Count0>>::index(&_a));
    dbg!(ConstIndex::<Count<Count<Count0>>>::index(&_a));
    // dbg!(ConstIndex::<Count<Count<Count<Count0>>>>::index(&_a)); -- doesn't compile
}


