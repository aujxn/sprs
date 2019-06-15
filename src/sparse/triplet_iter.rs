//! A structure for iterating over the non-zero values of any kind of
//! sparse matrix.

use num_traits::Num;

use indexing::SpIndex;
use sparse::{CsMatI, TriMatIter};
use crate::CompressedStorage;

impl<'a, N, I, RI, CI, DI> Iterator for TriMatIter<RI, CI, DI>
where
    I: 'a + SpIndex,
    N: 'a,
    RI: Iterator<Item = &'a I>,
    CI: Iterator<Item = &'a I>,
    DI: Iterator<Item = &'a N>,
{
    type Item = (&'a N, (I, I));

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match (self.row_inds.next(), self.col_inds.next(), self.data.next()) {
            (Some(row), Some(col), Some(val)) => Some((val, (*row, *col))),
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.row_inds.size_hint() // FIXME merge hints?
    }
}

impl<'a, N, I, RI, CI, DI> TriMatIter<RI, CI, DI>
where
    I: 'a + SpIndex,
    N: 'a,
    RI: Iterator<Item = &'a I>,
    CI: Iterator<Item = &'a I>,
    DI: Iterator<Item = &'a N>,
{
    /// Create a new `TriMatIter` from iterators
    pub fn new(
        shape: (usize, usize),
        nnz: usize,
        row_inds: RI,
        col_inds: CI,
        data: DI,
    ) -> Self {
        Self {
            rows: shape.0,
            cols: shape.1,
            nnz,
            row_inds,
            col_inds,
            data,
        }
    }

    /// The number of rows of the matrix
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of cols of the matrix
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The shape of the matrix, as a `(rows, cols)` tuple
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub fn into_row_inds(self) -> RI {
        self.row_inds
    }

    pub fn into_col_inds(self) -> CI {
        self.col_inds
    }

    pub fn into_data(self) -> DI {
        self.data
    }

    pub fn transpose_into(self) -> TriMatIter<CI, RI, DI> {
        TriMatIter {
            rows: self.cols,
            cols: self.rows,
            nnz: self.nnz,
            row_inds: self.col_inds,
            col_inds: self.row_inds,
            data: self.data,
        }
    }
}

impl<'a, N, I, RI, CI, DI> TriMatIter<RI, CI, DI>
where
    I: 'a + SpIndex,
    N: 'a + Clone,
    RI: Clone + Iterator<Item = &'a I>,
    CI: Clone + Iterator<Item = &'a I>,
    DI: Clone + Iterator<Item = &'a N>,
{

    /// Consume TriMatIter and produce a CSC matrix
    pub fn into_csc(self) -> CsMatI<N, I> 
    where
        N: Num
    {
        self.into_cs(CompressedStorage::CSC)
    }

    /// Consume TriMatIter and produce a CSR matrix
    pub fn into_csr(self) -> CsMatI<N, I> 
    where
        N: Num
    {
        self.into_cs(CompressedStorage::CSR)
    }

    /// Consume TriMatIter and produce a CsMat matrix with the chosen storage
    pub fn into_cs(self, storage: crate::CompressedStorage) -> CsMatI<N, I>
    where
        N: Num
    {

        // (i,j, input position, output position)
        let mut rc: Vec<(I, I, usize, usize)> = Vec::new();

        let mut nnz_max = 0;
        for (_, (i, j)) in self.clone() {
            rc.push((i,j, nnz_max, 0));
            nnz_max += 1;
        }

        match storage {
            CompressedStorage::CSR => {
                rc.sort_by_key(|i| (i.0, i.1));
            },
            CompressedStorage::CSC => {
                rc.sort_by_key(|i| (i.1, i.0));
            },
        }

        let outer_idx = |idx: &(I,I, usize, usize)| {
            match storage {
                CompressedStorage::CSR => idx.0,
                CompressedStorage::CSC => idx.1,
            }
        };

        let mut slot = 0;
        let mut indptr = vec![I::zero()];
        let mut cur_outer = I::zero();

        for rec in 0 .. nnz_max {
            if rec > 0 {
                if rc[rec - 1].0 == rc[rec].0 && rc[rec - 1].1 == rc[rec].1 {
                    // got a duplicate
                } else {
                    slot += 1;
                }
            }

            rc[rec].3 = slot;

            let new_outer = outer_idx(&rc[rec]);
            while new_outer > cur_outer {
                indptr.push(I::from_usize(slot));
                cur_outer += I::one();
            }
        }

        slot += 1;
        indptr.push(I::from_usize(slot));

        rc.sort_by_key(|i| i.2);

        let mut data: Vec<N> = vec![N::zero(); slot];
        let mut indices: Vec<I> = vec![I::zero(); slot];

        for ((v, (i, j)), (i2, j2, pos, slot)) in self.clone().into_iter().zip(rc.into_iter()) {

            assert_eq!(i, i2);
            assert_eq!(j, j2);

            assert!({
                let outer = outer_idx(&(i2,j2, pos, slot));
                slot >= indptr[outer.index()].index() && slot < indptr[outer.index() + 1].index()
            });

            data[slot] = data[slot].clone() + v.clone();

            match storage {
                CompressedStorage::CSR => { indices[slot] = j },
                CompressedStorage::CSC => { indices[slot] = i },
            }
        }

        CsMatI {
            storage,
            nrows: self.rows,
            ncols: self.cols,
            indptr,
            indices,
            data,
        }
    }
}
