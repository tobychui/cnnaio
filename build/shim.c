/*
 * shim.c — fills two gaps when linking the prebuilt ncnn wasm archive against
 * wasi-sdk's wasi-libc + (no-exceptions) libc++:
 *
 *  1) newlib-style integer printf variants. LLVM at -O2 rewrites fprintf/printf
 *     calls that have no floating-point conversions into iprintf/fiprintf/etc.
 *     wasi-libc (musl) doesn't ship those, so we forward them to the real ones.
 *
 *  2) Itanium C++ ABI exception entry points. wasi-sdk's libc++ is built with
 *     exceptions disabled, so __cxa_throw & friends are undefined. ncnn doesn't
 *     throw on the inference happy-path; if it ever did, we abort loudly.
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stddef.h>

/* ---- integer printf family ---- */
int iprintf(const char* fmt, ...)              { va_list a; va_start(a, fmt); int r = vprintf(fmt, a);          va_end(a); return r; }
int fiprintf(FILE* f, const char* fmt, ...)    { va_list a; va_start(a, fmt); int r = vfprintf(f, fmt, a);      va_end(a); return r; }
int siprintf(char* s, const char* fmt, ...)    { va_list a; va_start(a, fmt); int r = vsprintf(s, fmt, a);      va_end(a); return r; }
int sniprintf(char* s, size_t n, const char* fmt, ...) { va_list a; va_start(a, fmt); int r = vsnprintf(s, n, fmt, a); va_end(a); return r; }

int __small_printf(const char* fmt, ...)           { va_list a; va_start(a, fmt); int r = vprintf(fmt, a);     va_end(a); return r; }
int __small_fprintf(FILE* f, const char* fmt, ...) { va_list a; va_start(a, fmt); int r = vfprintf(f, fmt, a); va_end(a); return r; }
int __small_sprintf(char* s, const char* fmt, ...) { va_list a; va_start(a, fmt); int r = vsprintf(s, fmt, a); va_end(a); return r; }
int __small_snprintf(char* s, size_t n, const char* fmt, ...) { va_list a; va_start(a, fmt); int r = vsnprintf(s, n, fmt, a); va_end(a); return r; }

/* ---- C++ exception ABI stubs (weak so real libc++abi wins if present) ---- */
__attribute__((weak)) void* __cxa_allocate_exception(size_t size) { return malloc(size); }
__attribute__((weak)) void  __cxa_free_exception(void* p)         { free(p); }
__attribute__((weak)) void  __cxa_throw(void* thrown, void* tinfo, void (*dest)(void*)) {
    (void)thrown; (void)tinfo; (void)dest;
    fprintf(stderr, "[shim] C++ exception thrown inside ncnn wasm — aborting\n");
    abort();
}
__attribute__((weak)) void  __cxa_rethrow(void)        { abort(); }
__attribute__((weak)) void* __cxa_begin_catch(void* p) { return p; }
__attribute__((weak)) void  __cxa_end_catch(void)      { }

/* ---- emscripten cpu-detection calls baked into ncnn's cpu.cpp (THREADS=0) ---- */
__attribute__((weak)) int emscripten_has_threading_support(void) { return 0; }
__attribute__((weak)) int emscripten_num_logical_cores(void)     { return 1; }
