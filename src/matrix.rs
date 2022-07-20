//! Provides a Matrix type to organize data into bidimensional dynamically sized matrices.
//! This is meant to be used only internally.

use std::{ops::{Index, IndexMut}, borrow::Borrow};

/// Dynamically sized, Vec allocated, simple matrix
#[derive(Clone)]
pub(crate) struct Matrix<T> {
    /// Heigth of the matrix
    rows: usize,
    /// Width of the matrix
    cols: usize,
    /// Row-major flattened contents of the matrix
    data: Vec<T>
}

impl<T: Clone> Matrix<T> {
    /// Creates a `Matrix<T>` from the given raw data in flattened, row-major order.
    /// 
    /// # Panics
    /// 
    /// Panics if `raw.len()` differs from `rows*cols`.
    pub fn from_raw_data(rows: usize, cols: usize, raw: Vec<T>) -> Self {
        if raw.len() != rows * cols {
            panic!("Matrix can not be built from data of invalid length");
        }

        Self {
            rows,
            cols,
            data: raw
        }
    }

    /// Create a `Matrix<T>` from the given statically sized matrix
    pub fn from_static<const R: usize, const C: usize>(raw: impl Borrow<[[T; C]; R]>) -> Self {
        Self {
            rows: R,
            cols: C,
            data: raw.borrow().iter().flatten().cloned().collect()
        }
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, row: usize) -> &Self::Output {
        // Exploit the fact that the matrix inner data is stored in row-major order
        &self.data[(self.cols*row)..(self.cols*row + self.cols)]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        &mut self.data[(self.cols*row)..(self.cols*row + self.cols)]
    }
}

impl<T: Clone, const R: usize, const C: usize> From<&[[T; C]; R]> for Matrix<T> {
    fn from(slice: &[[T; C]; R]) -> Self {
        Matrix::from_static(slice)
    }
}

impl<T: Clone, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(arr: [[T; C]; R]) -> Self {
        Matrix::from_static(arr)
    }
}
