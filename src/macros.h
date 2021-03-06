#ifndef MACROS_H_5JXBGRAD
#define MACROS_H_5JXBGRAD


/// Defines Macro fro cuda callable functions
#ifdef __CUDACC__
#define CUCALL __host__ __device__
#define LIB thrust
#else
#define CUCALL
#define LIB std
#endif

#define OUT ::std::cout

/// control Inlining
#define ALWAYS_INLINE __attribute__((always_inline))

#endif /* end of include guard: MACROS_H_5JXBGRAD */
