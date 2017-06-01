#ifndef MACROS_H_5JXBGRAD
#define MACROS_H_5JXBGRAD


/// Defines Macro fro cuda callable functions
#ifdef __CUDACC__
#define CUCALL __host__ __device__
#else
#define CUCALL 
#endif


#endif /* end of include guard: MACROS_H_5JXBGRAD */
