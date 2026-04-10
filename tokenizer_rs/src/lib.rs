/*!
 * kmamba_tokenizer - BPE Tokenizer FFI for k-mamba
 * 
 * Uses tiktoken-rs (GPT-2 tokenizer) for encoding/decoding.
 * Exposes C-compatible interface for integration with k-mamba C/CUDA.
 */

use std::ffi::{CStr, CString, c_char};
use std::os::raw::c_uint;
use std::slice;
use tiktoken_rs::cl100k_base_singleton;

// 32,768 vocab size constant matching k-mamba
const MAX_VOCAB_SIZE: usize = 32768;

/// Initialize the tokenizer (called automatically on first use)
fn get_tokenizer() -> std::sync::Arc<parking_lot::Mutex<tiktoken_rs::CoreBPE>> {
    cl100k_base_singleton()
}

/// Encode UTF-8 text into token IDs
/// 
/// # Safety
/// - text must be a valid null-terminated C string
/// - out_len must be a valid pointer to write the length
/// 
/// Returns: pointer to heap-allocated array of uint32_t tokens
/// Caller must free with kmamba_free_tokens()
#[no_mangle]
pub unsafe extern "C" fn kmamba_encode(text: *const c_char, out_len: *mut usize) -> *mut c_uint {
    if text.is_null() || out_len.is_null() {
        return std::ptr::null_mut();
    }
    
    // Convert C string to Rust string
    let c_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    // Tokenize using GPT-2 tokenizer
    let tokenizer = get_tokenizer();
    let tokens = tokenizer.lock().encode_with_special_tokens(c_str);
    
    // Clamp to max vocab size (modulo for safety)
    let clamped_tokens: Vec<c_uint> = tokens
        .into_iter()
        .map(|t| t % MAX_VOCAB_SIZE as c_uint)
        .collect();
    
    let len = clamped_tokens.len();
    
    // Allocate memory for C caller
    let layout = std::alloc::Layout::array::<c_uint>(len).unwrap();
    let ptr = std::alloc::alloc(layout) as *mut c_uint;
    
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    // Copy data
    std::ptr::copy_nonoverlapping(clamped_tokens.as_ptr(), ptr, len);
    
    // Write length
    *out_len = len;
    
    ptr
}

/// Decode token IDs back to UTF-8 text
/// 
/// # Safety
/// - tokens must be a valid pointer to array of uint32_t
/// - len is the length of the array
/// 
/// Returns: pointer to heap-allocated C string
/// Caller must free with kmamba_free_string()
#[no_mangle]
pub unsafe extern "C" fn kmamba_decode(tokens: *const c_uint, len: usize) -> *mut c_char {
    if tokens.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    
    // Convert to Rust slice
    let token_slice = slice::from_raw_parts(tokens, len);
    
    // Decode
    let tokenizer = get_tokenizer();
    let text = match tokenizer.lock().decode(token_slice.to_vec()) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    // Convert to C string
    match CString::new(text) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free memory allocated by kmamba_encode
/// 
/// # Safety
/// - ptr must have been allocated by kmamba_encode
/// - len must be the original length from kmamba_encode
#[no_mangle]
pub unsafe extern "C" fn kmamba_free_tokens(ptr: *mut c_uint, len: usize) {
    if !ptr.is_null() && len > 0 {
        let layout = std::alloc::Layout::array::<c_uint>(len).unwrap();
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

/// Free memory allocated by kmamba_decode
/// 
/// # Safety
/// - ptr must have been allocated by kmamba_decode
#[no_mangle]
pub unsafe extern "C" fn kmamba_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        // Reclaim ownership and drop
        let _ = CString::from_raw(ptr);
    }
}

/// Get the vocab size constant
#[no_mangle]
pub extern "C" fn kmamba_vocab_size() -> usize {
    MAX_VOCAB_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_decode() {
        let text = "Hello, world!";
        let cstring = CString::new(text).unwrap();
        
        unsafe {
            let mut len: usize = 0;
            let tokens = kmamba_encode(cstring.as_ptr(), &mut len);
            assert!(!tokens.is_null());
            assert!(len > 0);
            
            let decoded = kmamba_decode(tokens, len);
            assert!(!decoded.is_null());
            
            // Cleanup
            kmamba_free_tokens(tokens);
            kmamba_free_string(decoded);
        }
    }
}
