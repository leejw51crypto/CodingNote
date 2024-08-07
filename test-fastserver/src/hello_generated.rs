// automatically generated by the FlatBuffers compiler, do not modify


// @generated

use core::mem;
use core::cmp::Ordering;

extern crate flatbuffers;
use self::flatbuffers::{EndianScalar, Follow};

pub enum HelloWorldOffset {}
#[derive(Copy, Clone, PartialEq)]

pub struct HelloWorld<'a> {
  pub _tab: flatbuffers::Table<'a>,
}

impl<'a> flatbuffers::Follow<'a> for HelloWorld<'a> {
  type Inner = HelloWorld<'a>;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    Self { _tab: flatbuffers::Table::new(buf, loc) }
  }
}

impl<'a> HelloWorld<'a> {
  pub const VT_NAME: flatbuffers::VOffsetT = 4;
  pub const VT_DATA: flatbuffers::VOffsetT = 6;

  #[inline]
  pub unsafe fn init_from_table(table: flatbuffers::Table<'a>) -> Self {
    HelloWorld { _tab: table }
  }
  #[allow(unused_mut)]
  pub fn create<'bldr: 'args, 'args: 'mut_bldr, 'mut_bldr>(
    _fbb: &'mut_bldr mut flatbuffers::FlatBufferBuilder<'bldr>,
    args: &'args HelloWorldArgs<'args>
  ) -> flatbuffers::WIPOffset<HelloWorld<'bldr>> {
    let mut builder = HelloWorldBuilder::new(_fbb);
    if let Some(x) = args.data { builder.add_data(x); }
    if let Some(x) = args.name { builder.add_name(x); }
    builder.finish()
  }


  #[inline]
  pub fn name(&self) -> Option<&'a str> {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<flatbuffers::ForwardsUOffset<&str>>(HelloWorld::VT_NAME, None)}
  }
  #[inline]
  pub fn data(&self) -> Option<flatbuffers::Vector<'a, u8>> {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'a, u8>>>(HelloWorld::VT_DATA, None)}
  }
}

impl flatbuffers::Verifiable for HelloWorld<'_> {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.visit_table(pos)?
     .visit_field::<flatbuffers::ForwardsUOffset<&str>>("name", Self::VT_NAME, false)?
     .visit_field::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'_, u8>>>("data", Self::VT_DATA, false)?
     .finish();
    Ok(())
  }
}
pub struct HelloWorldArgs<'a> {
    pub name: Option<flatbuffers::WIPOffset<&'a str>>,
    pub data: Option<flatbuffers::WIPOffset<flatbuffers::Vector<'a, u8>>>,
}
impl<'a> Default for HelloWorldArgs<'a> {
  #[inline]
  fn default() -> Self {
    HelloWorldArgs {
      name: None,
      data: None,
    }
  }
}

pub struct HelloWorldBuilder<'a: 'b, 'b> {
  fbb_: &'b mut flatbuffers::FlatBufferBuilder<'a>,
  start_: flatbuffers::WIPOffset<flatbuffers::TableUnfinishedWIPOffset>,
}
impl<'a: 'b, 'b> HelloWorldBuilder<'a, 'b> {
  #[inline]
  pub fn add_name(&mut self, name: flatbuffers::WIPOffset<&'b  str>) {
    self.fbb_.push_slot_always::<flatbuffers::WIPOffset<_>>(HelloWorld::VT_NAME, name);
  }
  #[inline]
  pub fn add_data(&mut self, data: flatbuffers::WIPOffset<flatbuffers::Vector<'b , u8>>) {
    self.fbb_.push_slot_always::<flatbuffers::WIPOffset<_>>(HelloWorld::VT_DATA, data);
  }
  #[inline]
  pub fn new(_fbb: &'b mut flatbuffers::FlatBufferBuilder<'a>) -> HelloWorldBuilder<'a, 'b> {
    let start = _fbb.start_table();
    HelloWorldBuilder {
      fbb_: _fbb,
      start_: start,
    }
  }
  #[inline]
  pub fn finish(self) -> flatbuffers::WIPOffset<HelloWorld<'a>> {
    let o = self.fbb_.end_table(self.start_);
    flatbuffers::WIPOffset::new(o.value())
  }
}

impl core::fmt::Debug for HelloWorld<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut ds = f.debug_struct("HelloWorld");
      ds.field("name", &self.name());
      ds.field("data", &self.data());
      ds.finish()
  }
}
#[inline]
/// Verifies that a buffer of bytes contains a `HelloWorld`
/// and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_hello_world_unchecked`.
pub fn root_as_hello_world(buf: &[u8]) -> Result<HelloWorld, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root::<HelloWorld>(buf)
}
#[inline]
/// Verifies that a buffer of bytes contains a size prefixed
/// `HelloWorld` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `size_prefixed_root_as_hello_world_unchecked`.
pub fn size_prefixed_root_as_hello_world(buf: &[u8]) -> Result<HelloWorld, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root::<HelloWorld>(buf)
}
#[inline]
/// Verifies, with the given options, that a buffer of bytes
/// contains a `HelloWorld` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_hello_world_unchecked`.
pub fn root_as_hello_world_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<HelloWorld<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root_with_opts::<HelloWorld<'b>>(opts, buf)
}
#[inline]
/// Verifies, with the given verifier options, that a buffer of
/// bytes contains a size prefixed `HelloWorld` and returns
/// it. Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_hello_world_unchecked`.
pub fn size_prefixed_root_as_hello_world_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<HelloWorld<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root_with_opts::<HelloWorld<'b>>(opts, buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a HelloWorld and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid `HelloWorld`.
pub unsafe fn root_as_hello_world_unchecked(buf: &[u8]) -> HelloWorld {
  flatbuffers::root_unchecked::<HelloWorld>(buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a size prefixed HelloWorld and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid size prefixed `HelloWorld`.
pub unsafe fn size_prefixed_root_as_hello_world_unchecked(buf: &[u8]) -> HelloWorld {
  flatbuffers::size_prefixed_root_unchecked::<HelloWorld>(buf)
}
#[inline]
pub fn finish_hello_world_buffer<'a, 'b>(
    fbb: &'b mut flatbuffers::FlatBufferBuilder<'a>,
    root: flatbuffers::WIPOffset<HelloWorld<'a>>) {
  fbb.finish(root, None);
}

#[inline]
pub fn finish_size_prefixed_hello_world_buffer<'a, 'b>(fbb: &'b mut flatbuffers::FlatBufferBuilder<'a>, root: flatbuffers::WIPOffset<HelloWorld<'a>>) {
  fbb.finish_size_prefixed(root, None);
}
