//! Experimental Linux/Zstandard comparison for existing LIVWIN02 streams.
use std::{
    error::Error,
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    ptr::NonNull,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const RAW_MAGIC: &[u8; 8] = b"LIVWIN02";
const MAGIC: &[u8; 8] = b"LIVZST01";
const RAW: usize = 73;
const HEADER: u64 = 24;
const ENTRY: u64 = 12;
const MAX_BLOCK: usize = 4096;
const COMPRESSION_LEVEL: c_int = 100;
const CHECKSUM_FLAG: c_int = 201;

#[link(name = "libzstd.so.1", kind = "dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    fn ZSTD_versionNumber() -> c_uint;
    fn ZSTD_createCCtx() -> *mut c_void;
    fn ZSTD_freeCCtx(ctx: *mut c_void) -> usize;
    fn ZSTD_CCtx_setParameter(ctx: *mut c_void, parameter: c_int, value: c_int) -> usize;
    fn ZSTD_compressBound(size: usize) -> usize;
    fn ZSTD_compress2(
        ctx: *mut c_void,
        dst: *mut u8,
        cap: usize,
        src: *const u8,
        len: usize,
    ) -> usize;
    fn ZSTD_decompress(dst: *mut u8, cap: usize, src: *const u8, len: usize) -> usize;
    fn ZSTD_isError(code: usize) -> c_uint;
    fn ZSTD_getErrorName(code: usize) -> *const c_char;
}

fn zstd_result(code: usize) -> Result<usize> {
    // Zstandard supplies a stable, NUL-terminated error string for its own result codes.
    if unsafe { ZSTD_isError(code) } == 0 {
        return Ok(code);
    }
    let error = unsafe { CStr::from_ptr(ZSTD_getErrorName(code)) };
    Err(error.to_string_lossy().into_owned().into())
}

struct Compressor(NonNull<c_void>);

impl Compressor {
    fn new() -> Result<Self> {
        // The context is owned exclusively by this value, including on parameter errors.
        let ctx = Self(NonNull::new(unsafe { ZSTD_createCCtx() }).ok_or("no Zstd context")?);
        for (parameter, value) in [(COMPRESSION_LEVEL, 3), (CHECKSUM_FLAG, 1)] {
            zstd_result(unsafe { ZSTD_CCtx_setParameter(ctx.0.as_ptr(), parameter, value) })?;
        }
        Ok(ctx)
    }

    fn compress(&mut self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        // Distinct slices provide the exact readable input and writable output limits.
        zstd_result(unsafe {
            ZSTD_compress2(
                self.0.as_ptr(),
                output.as_mut_ptr(),
                output.len(),
                input.as_ptr(),
                input.len(),
            )
        })
    }
}

impl Drop for Compressor {
    fn drop(&mut self) {
        unsafe { ZSTD_freeCCtx(self.0.as_ptr()) };
    }
}

#[derive(Clone, Copy)]
enum Layout {
    Raw = 0,
    Narrow = 1,
}

impl Layout {
    fn width(self) -> usize {
        match self {
            Self::Raw => RAW,
            Self::Narrow => 28,
        }
    }

    fn pack(self, raw: &[u8], packed: &mut Vec<u8>) -> Result<()> {
        packed.clear();
        if matches!(self, Self::Raw) {
            packed.extend_from_slice(raw);
        } else {
            for row in raw.as_chunks::<RAW>().0 {
                packed.push(row[0]);
                for pair in row[1..].as_chunks::<8>().0 {
                    let id = u32::from_le_bytes(pair[..4].try_into()?);
                    let bytes = u32::from_le_bytes(pair[4..].try_into()?);
                    packed.extend_from_slice(&u16::try_from(id)?.to_le_bytes());
                    packed.push(u8::try_from(bytes)?);
                }
            }
        }
        Ok(())
    }

    fn unpack(self, packed: &[u8], raw: &mut [u8; RAW]) {
        if matches!(self, Self::Raw) {
            raw.copy_from_slice(packed);
        } else {
            *raw = [0; RAW];
            raw[0] = packed[0];
            for (src, dst) in packed[1..]
                .as_chunks::<3>()
                .0
                .iter()
                .zip(raw[1..].as_chunks_mut::<8>().0)
            {
                dst[..2].copy_from_slice(&src[..2]);
                dst[4] = src[2];
            }
        }
    }
}

fn read<const N: usize>(input: &mut File) -> Result<[u8; N]> {
    let mut data = [0; N];
    input.read_exact(&mut data)?;
    Ok(data)
}

fn raw_rows(input: &mut File) -> Result<u64> {
    input.rewind()?;
    if &read::<8>(input)? != RAW_MAGIC {
        return Err("incorrect raw stream version".into());
    }
    let bytes = input
        .metadata()?
        .len()
        .checked_sub(8)
        .ok_or("truncated raw stream")?;
    if !bytes.is_multiple_of(RAW as u64) {
        return Err("truncated raw token record".into());
    }
    Ok(bytes / RAW as u64)
}

struct Format {
    rows: u64,
    block_rows: u32,
    layout: Layout,
}

impl Format {
    fn new(rows: u64, block_rows: u32, layout: Layout) -> Result<Self> {
        if rows == 0 || !(1..=MAX_BLOCK as u32).contains(&block_rows) {
            return Err("invalid rows or block size".into());
        }
        Ok(Self {
            rows,
            block_rows,
            layout,
        })
    }

    fn blocks(&self) -> u64 {
        self.rows.div_ceil(u64::from(self.block_rows))
    }

    fn data_start(&self) -> Result<u64> {
        self.blocks()
            .checked_mul(ENTRY)
            .and_then(|len| HEADER.checked_add(len))
            .ok_or_else(|| "index size overflow".into())
    }

    fn rows_in(&self, block: u64) -> usize {
        let width = u64::from(self.block_rows);
        (self.rows - block * width).min(width) as usize
    }

    fn read(input: &mut File) -> Result<Self> {
        if &read::<8>(input)? != MAGIC {
            return Err("incorrect compact stream version".into());
        }
        let rows = u64::from_le_bytes(read(input)?);
        let block_rows = u32::from_le_bytes(read(input)?);
        let layout = match read::<4>(input)? {
            [0, 0, 0, 0] => Layout::Raw,
            [1, 0, 0, 0] => Layout::Narrow,
            _ => return Err("unsupported compact layout".into()),
        };
        Self::new(rows, block_rows, layout)
    }

    fn write(&self, output: &mut File) -> Result<()> {
        output.write_all(MAGIC)?;
        output.write_all(&self.rows.to_le_bytes())?;
        output.write_all(&self.block_rows.to_le_bytes())?;
        output.write_all(&[self.layout as u8, 0, 0, 0])?;
        Ok(())
    }
}

fn encode(input: &mut File, path: &Path, format: &Format) -> Result<()> {
    let mut output = OpenOptions::new().create_new(true).write(true).open(path)?;
    format.write(&mut output)?;
    let mut offset = format.data_start()?;
    let reserve = format.block_rows as usize;
    let mut input_bytes = vec![0; reserve * RAW];
    let mut packed = Vec::with_capacity(reserve * format.layout.width());
    let mut compressed = vec![0; unsafe { ZSTD_compressBound(reserve * format.layout.width()) }];
    let mut compressor = Compressor::new()?;
    for block in 0..format.blocks() {
        let raw = &mut input_bytes[..format.rows_in(block) * RAW];
        input.read_exact(raw)?;
        format.layout.pack(raw, &mut packed)?;
        let len = compressor.compress(&packed, &mut compressed)?;
        output.seek(SeekFrom::Start(HEADER + block * ENTRY))?;
        output.write_all(&offset.to_le_bytes())?;
        output.write_all(&u32::try_from(len)?.to_le_bytes())?;
        output.seek(SeekFrom::Start(offset))?;
        output.write_all(&compressed[..len])?;
        offset += len as u64;
    }
    output.sync_all()?;
    Ok(())
}

struct Reader {
    file: File,
    format: Format,
    bytes: u64,
    data_start: u64,
    block: Option<u64>,
    end: u64,
    compressed: Vec<u8>,
    decoded: Vec<u8>,
}

impl Reader {
    fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let format = Format::read(&mut file)?;
        let bytes = file.metadata()?.len();
        let data_start = format.data_start()?;
        if data_start >= bytes {
            return Err("truncated compact seek table or payload".into());
        }
        let decoded_len = format.block_rows as usize * format.layout.width();
        Ok(Self {
            file,
            format,
            bytes,
            data_start,
            block: None,
            end: data_start,
            compressed: vec![0; unsafe { ZSTD_compressBound(decoded_len) }],
            decoded: vec![0; decoded_len],
        })
    }

    fn load(&mut self, block: u64, sequential: bool) -> Result<()> {
        if block >= self.format.blocks() {
            return Err("block out of range".into());
        }
        self.file.seek(SeekFrom::Start(HEADER + block * ENTRY))?;
        let start = u64::from_le_bytes(read(&mut self.file)?);
        let len = u32::from_le_bytes(read(&mut self.file)?) as usize;
        let end = start
            .checked_add(len as u64)
            .ok_or("overflow in block extent")?;
        if start < self.data_start
            || end > self.bytes
            || len == 0
            || len > self.compressed.len()
            || (sequential && start != self.end)
        {
            return Err("invalid block extent".into());
        }
        self.file.seek(SeekFrom::Start(start))?;
        self.file.read_exact(&mut self.compressed[..len])?;
        let expected = self.format.rows_in(block) * self.format.layout.width();
        // The writable slice is sized from the validated block size, never from the Zstd frame.
        let actual = zstd_result(unsafe {
            ZSTD_decompress(
                self.decoded.as_mut_ptr(),
                expected,
                self.compressed.as_ptr(),
                len,
            )
        })?;
        if actual != expected {
            return Err("decompressed block size mismatch".into());
        }
        self.block = Some(block);
        self.end = end;
        Ok(())
    }

    fn row(&mut self, row: u64) -> Result<[u8; RAW]> {
        // The benchmark calls compare over the entire container before timing random reads.
        if row >= self.format.rows {
            return Err("row out of range".into());
        }
        let block_rows = u64::from(self.format.block_rows);
        let block = row / block_rows;
        if self.block != Some(block) {
            self.load(block, false)?;
        }
        let mut raw = [0; RAW];
        let width = self.format.layout.width();
        let offset = (row % block_rows) as usize * width;
        self.format
            .layout
            .unpack(&self.decoded[offset..offset + width], &mut raw);
        Ok(raw)
    }
}

fn compare(input: &mut File, reader: &mut Reader) -> Result<()> {
    if raw_rows(input)? != reader.format.rows {
        return Err("window count mismatch".into());
    }
    reader.end = reader.data_start;
    let mut raw = vec![0; reader.format.block_rows as usize * RAW];
    let mut unpacked = [0; RAW];
    let width = reader.format.layout.width();
    for block in 0..reader.format.blocks() {
        reader.load(block, true)?;
        let rows = reader.format.rows_in(block);
        input.read_exact(&mut raw[..rows * RAW])?;
        for (expected, packed) in raw[..rows * RAW]
            .as_chunks::<RAW>()
            .0
            .iter()
            .zip(reader.decoded[..rows * width].chunks_exact(width))
        {
            reader.format.layout.unpack(packed, &mut unpacked);
            if *expected != unpacked {
                return Err("byte-exact token record verification failed".into());
            }
        }
    }
    if reader.end != reader.bytes {
        return Err("trailing compact stream data".into());
    }
    Ok(())
}

fn raw_row(file: &mut File, position: u64) -> Result<[u8; RAW]> {
    file.seek(SeekFrom::Start(8 + position * RAW as u64))?;
    read(file)
}

fn benchmark(input_path: &Path, output_path: &Path, layout: Layout, block_rows: u32) -> Result<()> {
    let mut input = File::open(input_path)?;
    let format = Format::new(raw_rows(&mut input)?, block_rows, layout)?;
    let start = Instant::now();
    encode(&mut input, output_path, &format)?;
    let pack = start.elapsed();
    let mut reader = Reader::open(output_path)?;
    let start = Instant::now();
    compare(&mut input, &mut reader)?;
    let verify = start.elapsed();
    let mut random = 19u64;
    let positions: Vec<_> = (0..20_000)
        .map(|_| {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random % format.rows
        })
        .collect();
    let mut expected = vec![[0; RAW]; positions.len()];
    let mut actual = vec![[0; RAW]; positions.len()];
    let start = Instant::now();
    for (&position, row) in positions.iter().zip(&mut expected) {
        *row = raw_row(&mut input, position)?;
    }
    let raw_read = start.elapsed();
    let start = Instant::now();
    for (&position, row) in positions.iter().zip(&mut actual) {
        *row = reader.row(position)?;
    }
    let packed_read = start.elapsed();
    if actual != expected {
        return Err("random window mismatch".into());
    }
    let (input_bytes, output_bytes) = (input.metadata()?.len(), reader.bytes);
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.3}\t{:.3}",
        unsafe { ZSTD_versionNumber() },
        layout.width(),
        block_rows,
        format.rows,
        input_bytes,
        output_bytes,
        pack.as_secs_f64(),
        verify.as_secs_f64(),
        raw_read.as_secs_f64() * 1e6 / positions.len() as f64,
        packed_read.as_secs_f64() * 1e6 / positions.len() as f64,
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [input, output, layout, size] = args.as_slice() else {
        return Err("usage: compact-stream INPUT OUTPUT raw|narrow BLOCK_ROWS".into());
    };
    let layout = match layout.to_str() {
        Some("raw") => Layout::Raw,
        Some("narrow") => Layout::Narrow,
        _ => return Err("unsupported layout".into()),
    };
    benchmark(
        Path::new(input),
        Path::new(output),
        layout,
        size.to_str().ok_or("invalid block size")?.parse()?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_flags_context_and_high_bytes() -> Result<()> {
        let mut row = [0; RAW];
        row[0] = 0b10001001;
        row[1..5].copy_from_slice(&65535u32.to_le_bytes());
        row[5..9].copy_from_slice(&255u32.to_le_bytes());
        let mut packed = Vec::new();
        Layout::Narrow.pack(&row, &mut packed)?;
        let mut actual = [0; RAW];
        Layout::Narrow.unpack(&packed, &mut actual);
        assert_eq!(actual, row);
        row[3] = 1;
        assert!(Layout::Narrow.pack(&row, &mut packed).is_err());
        Ok(())
    }

    #[test]
    fn bad_checksum_fails_project_decoder() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "packtok-m7-zstd-{}-{}.trial",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)?;
        let raw = [5; RAW];
        let format = Format::new(1, 1, Layout::Raw)?;
        format.write(&mut file)?;
        file.write_all(&format.data_start()?.to_le_bytes())?;
        let mut zipped = vec![0; unsafe { ZSTD_compressBound(RAW) }];
        let size = Compressor::new()?.compress(&raw, &mut zipped)?;
        file.write_all(&(size as u32).to_le_bytes())?;
        file.write_all(&zipped[..size])?;
        assert_eq!(Reader::open(&path)?.row(0)?, raw);
        file.seek(SeekFrom::End(-1))?;
        file.write_all(&[zipped[size - 1] ^ 1])?;
        assert!(Reader::open(&path)?.row(0).is_err());
        file.set_len(HEADER + ENTRY)?;
        assert!(Reader::open(&path).is_err());
        std::fs::remove_file(path)?;
        Ok(())
    }
}
