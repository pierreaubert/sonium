//! Pool-backed scratch reuse for mixed-precision GMRES.

// Rust guideline compliant 2026-02-21
use crate::pool::Buffer;
use crate::{Backend, BufferElement, LowerPrecision, MemoryPool};
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

/// Retain mixed-precision device scratch across restarts and repeated solves.
///
/// The workspace owns a clone of its memory pool and uses that pool's backend.
/// Scratch grows on demand to the maximum simultaneous demand encountered.
/// Changing dimension releases cached vectors. Dropping the workspace returns
/// all scratch to the pool quota. Host Hessenberg and diagnostic arrays are
/// still per-call allocations; backend reduction caches have separate ownership.
pub struct GmresIrWorkspace<B: Backend, T: LowerPrecision> {
    pub(super) pool: MemoryPool<B>,
    pub(super) outer: Cache<B, T>,
    pub(super) inner: Cache<B, T::Lower>,
    dimension: Option<usize>,
}

impl<B: Backend, T: LowerPrecision> std::fmt::Debug for GmresIrWorkspace<B, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GmresIrWorkspace")
            .field("dimension", &self.dimension)
            .field("outer_cached_vectors", &self.outer.buffers.borrow().len())
            .field("inner_cached_vectors", &self.inner.buffers.borrow().len())
            .finish_non_exhaustive()
    }
}

impl<B: Backend, T: LowerPrecision> GmresIrWorkspace<B, T> {
    /// Create an empty workspace without allocating device storage.
    pub fn new(pool: &MemoryPool<B>) -> Self {
        Self {
            pool: pool.clone(),
            outer: Cache {
                buffers: RefCell::new(Vec::new()),
            },
            inner: Cache {
                buffers: RefCell::new(Vec::new()),
            },
            dimension: None,
        }
    }

    pub(super) fn clear(&mut self) {
        self.outer.buffers.get_mut().clear();
        self.inner.buffers.get_mut().clear();
        self.dimension = None;
    }

    pub(super) fn prepare(&mut self, dimension: usize) {
        if self.dimension != Some(dimension) {
            self.outer.buffers.get_mut().clear();
            self.inner.buffers.get_mut().clear();
            self.dimension = Some(dimension);
        }
    }
}

pub(super) struct Cache<B: Backend, T: BufferElement> {
    buffers: RefCell<Vec<Buffer<T, B>>>,
}

pub(super) struct Allocator<'a, B: Backend, T: BufferElement> {
    backend: &'a B,
    pooled: Option<(&'a MemoryPool<B>, &'a Cache<B, T>)>,
}

impl<'a, B: Backend, T: BufferElement> Allocator<'a, B, T> {
    pub(super) fn raw(backend: &'a B) -> Self {
        Self {
            backend,
            pooled: None,
        }
    }

    pub(super) fn pooled(pool: &'a MemoryPool<B>, cache: &'a Cache<B, T>) -> Self {
        Self {
            backend: pool.backend(),
            pooled: Some((pool, cache)),
        }
    }

    pub(super) fn alloc(&self, n: usize) -> Result<Scratch<'a, B, T>, crate::OomError> {
        if let Some((pool, cache)) = self.pooled {
            let buffer = cache.buffers.borrow_mut().pop();
            let buffer = match buffer {
                Some(buffer) => buffer,
                None => pool.try_alloc(n)?,
            };
            debug_assert_eq!(buffer.len(), n);
            Ok(Scratch::Pooled {
                buffer: Some(buffer),
                cache,
            })
        } else {
            Ok(Scratch::Raw(self.backend.raw_alloc(n)?))
        }
    }
}

pub(super) enum Scratch<'a, B: Backend, T: BufferElement> {
    Raw(B::Buffer<T>),
    Pooled {
        buffer: Option<Buffer<T, B>>,
        cache: &'a Cache<B, T>,
    },
}

impl<B: Backend, T: BufferElement> Deref for Scratch<'_, B, T> {
    type Target = B::Buffer<T>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Raw(buffer) => buffer,
            Self::Pooled { buffer, .. } => buffer.as_ref().expect("owned scratch").raw(),
        }
    }
}
impl<B: Backend, T: BufferElement> DerefMut for Scratch<'_, B, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Raw(buffer) => buffer,
            Self::Pooled { buffer, .. } => buffer.as_mut().expect("owned scratch").raw_mut(),
        }
    }
}
impl<B: Backend, T: BufferElement> Drop for Scratch<'_, B, T> {
    fn drop(&mut self) {
        if let Self::Pooled { buffer, cache } = self
            && let Some(buffer) = buffer.take()
        {
            cache.buffers.borrow_mut().push(buffer);
        }
    }
}
